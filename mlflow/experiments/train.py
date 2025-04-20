import os
import gc
import torch
import mlflow
import torch.utils.data
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from NGG.utils.utils import preprocess_dataset, linear_beta_schedule
from NGG.train_utils.load_or_not_deepset import load_or_not_deepset
from NGG.train_utils.load_or_not_stat_model import load_or_not_stat_model
from NGG.train_utils.load_autoencoder import load_autoencoder
from NGG.train_utils.train_autoencoder import train_autoencoder_mlflow
from NGG.train_utils.denoiser_train import train_denoise_mlflow
from NGG.train_utils.check_results import check_results
from NGG.denoiser.denoise_model import DenoiseNN
from NGG.autoencoders.autoencoder_base import VariationalAutoEncoder
from NGG.autoencoders.autoencoder_concat import VariationalAutoEncoder_concat
from NGG.autoencoders.autoencoder_GMVAE import GMVAE


def run_training(args, device):
    """
    Core training function that takes an args object as input and performs training, evaluation, 
    and logging to MLFlow.
    """
    # Check if MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD exist
    if os.getenv("MLFLOW_TRACKING_USERNAME") and os.getenv("MLFLOW_TRACKING_PASSWORD"):
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://user-lstepien-mlflow.user.lab.sspcloud.fr")
    else:
        mlflow_tracking_uri = "http://localhost:5000"

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Map VAE types
    VAE_mapper = {
        "base": VariationalAutoEncoder,
        "concat": VariationalAutoEncoder_concat,
        "features": "NotImplemented",
        "GMVAE": GMVAE,
    }
    VAE_class = VAE_mapper[args.AE]

    # Start MLFlow run
    with mlflow.start_run(run_name=args.name):
        # Log model configuration and hyperparameters
        mlflow.log_params(vars(args))

        # Preprocess datasets
        if args.labelize:
            trainset, kmeans = preprocess_dataset(
                "train",
                args.n_max_nodes,
                args.spectral_emb_dim,
                args.normalize,
                args.labelize,
                args.additional,
                n_clusters=args.n_clusters,
            )
            validset, _ = preprocess_dataset(
                "valid",
                args.n_max_nodes,
                args.spectral_emb_dim,
                args.normalize,
                args.labelize,
                args.additional,
                kmeans=kmeans,
            )
            testset, _ = preprocess_dataset(
                "test",
                args.n_max_nodes,
                args.spectral_emb_dim,
                args.normalize,
                args.labelize,
                args.additional,
                kmeans=kmeans,
            )
        else:
            trainset = preprocess_dataset(
                "train",
                args.n_max_nodes,
                args.spectral_emb_dim,
                args.normalize,
                args.labelize,
                args.additional,
            )
            validset = preprocess_dataset(
                "valid",
                args.n_max_nodes,
                args.spectral_emb_dim,
                args.normalize,
                args.labelize,
                args.additional,
            )
            testset = preprocess_dataset(
                "test",
                args.n_max_nodes,
                args.spectral_emb_dim,
                args.normalize,
                args.labelize,
                args.additional,
            )
            kmeans = None

        if hasattr(args, "cv_folds") and args.cv_folds > 1:
            cv_folds = args.cv_folds
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            mse_folds = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(trainset), start=1):
                print(f"Début du fold {fold}/{cv_folds}")
                cv_trainset = torch.utils.data.Subset(trainset, train_idx)
                cv_validset = torch.utils.data.Subset(trainset, val_idx)
                train_loader = DataLoader(cv_trainset, batch_size=args.batch_size, shuffle=True)
                val_loader = DataLoader(cv_validset, batch_size=args.batch_size, shuffle=False)

                args.node_feature_dimension = cv_trainset[0].x.shape[1]

                # Charger les composants du modèle pour ce fold
                deepsets = load_or_not_deepset(args, device)
                stat_model = load_or_not_stat_model(args, train_loader, device)
                to_labels = stat_model if stat_model is not None else None
                autoencoder = load_autoencoder(args, VAE_class, args.AE, to_labels, device, deepsets)

                optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

                autoencoder, _ = train_autoencoder_mlflow(
                    args, autoencoder, train_loader, val_loader, device, optimizer, scheduler
                )

                # Entraînement du modèle de débruitage
                betas = linear_beta_schedule(timesteps=args.timesteps)
                denoise_model = DenoiseNN(
                    input_dim=args.latent_dim,
                    hidden_dim=args.hidden_dim_denoise,
                    n_layers=args.n_layers_denoise,
                    n_cond=args.n_condition,
                    d_cond=args.dim_condition,
                ).to(device)
                optimizer_den = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
                scheduler_den = torch.optim.lr_scheduler.StepLR(optimizer_den, step_size=500, gamma=0.1)

                denoise_model, _ = train_denoise_mlflow(
                    args,
                    denoise_model,
                    autoencoder,
                    optimizer_den,
                    scheduler_den,
                    train_loader,
                    val_loader,
                    device,
                    torch.sqrt(torch.cumprod(1.0 - betas, axis=0)),
                    torch.sqrt(1.0 - torch.cumprod(1.0 - betas, axis=0)),
                )

                denoise_model.eval()

                # Évaluer sur le fold de validation
                val_loader_eval = DataLoader(cv_validset, batch_size=args.batch_size, shuffle=False)
                metrics = check_results(
                    args, device, autoencoder, denoise_model, val_loader_eval, cv_validset, betas
                )
                print(f"Fold {fold} MSE: {metrics['mse_all_features']}")
                mse_folds.append(metrics["mse_all_features"])

                # Nettoyage
                del train_loader, val_loader, val_loader_eval
                torch.cuda.empty_cache()
                gc.collect()

            avg_mse = sum(mse_folds) / len(mse_folds)
            print(f"MSE moyen en validation croisée: {avg_mse}")

            # Entraînement final sur l'ensemble complet full_train avant évaluation sur testset
            train_loader_final = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
            val_loader_final = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
            deepsets = load_or_not_deepset(args, device)
            stat_model = load_or_not_stat_model(args, train_loader_final, device)
            to_labels = stat_model if stat_model is not None else None
            autoencoder = load_autoencoder(args, VAE_class, args.AE, to_labels, device, deepsets)

            optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

            autoencoder, training_history_autoencoder = train_autoencoder_mlflow(
                args, autoencoder, train_loader_final, val_loader_final, device, optimizer, scheduler
            )

            for epoch, (train_loss, val_loss) in enumerate(
                zip(
                    training_history_autoencoder["train_loss_autoencoder"],
                    training_history_autoencoder["val_loss_autoencoder"],
                ),
                start=1,
            ):
                mlflow.log_metric("train_loss_autoencoder", train_loss, step=epoch)
                mlflow.log_metric("val_loss_autoencoder", val_loss, step=epoch)

            betas = linear_beta_schedule(timesteps=args.timesteps)
            denoise_model = DenoiseNN(
                input_dim=args.latent_dim,
                hidden_dim=args.hidden_dim_denoise,
                n_layers=args.n_layers_denoise,
                n_cond=args.n_condition,
                d_cond=args.dim_condition,
            ).to(device)
            optimizer_den = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
            scheduler_den = torch.optim.lr_scheduler.StepLR(optimizer_den, step_size=500, gamma=0.1)

            denoise_model, training_history_denoiser = train_denoise_mlflow(
                args,
                denoise_model,
                autoencoder,
                optimizer_den,
                scheduler_den,
                train_loader_final,
                val_loader_final,
                device,
                torch.sqrt(torch.cumprod(1.0 - betas, axis=0)),
                torch.sqrt(1.0 - torch.cumprod(1.0 - betas, axis=0)),
            )

            for epoch, (train_loss, val_loss) in enumerate(
                zip(
                    training_history_denoiser["train_loss_denoise"],
                    training_history_denoiser["val_loss_denoise"],
                ),
                start=1,
            ):
                mlflow.log_metric("train_loss_denoise", train_loss, step=epoch)
                mlflow.log_metric("val_loss_denoise", val_loss, step=epoch)

            return autoencoder, denoise_model, avg_mse
        else:
            # Set node feature dimension
            args.node_feature_dimension = trainset[0].x.shape[1]

            # Initialize data loaders
            train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

            # Load models
            deepsets = load_or_not_deepset(args, device)
            stat_model = load_or_not_stat_model(args, train_loader, device)
            to_labels = stat_model if stat_model is not None else kmeans
            autoencoder = load_autoencoder(
                args, VAE_class, args.AE, to_labels, device, deepsets
            )

            # Train autoencoder
            optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

            autoencoder, training_history_autoencoder = train_autoencoder_mlflow(
                args, autoencoder, train_loader, val_loader, device, optimizer, scheduler
            )

            # Log autoencoder training and validation losses per epoch
            for epoch, (train_loss, val_loss) in enumerate(
                zip(
                    training_history_autoencoder["train_loss_autoencoder"],
                    training_history_autoencoder["val_loss_autoencoder"],
                ),
                start=1,
            ):
                mlflow.log_metric("train_loss_autoencoder", train_loss, step=epoch)
                mlflow.log_metric("val_loss_autoencoder", val_loss, step=epoch)

            torch.cuda.empty_cache()
            gc.collect()

            # Define beta schedule
            betas = linear_beta_schedule(timesteps=args.timesteps)

            # Initialize denoising model
            denoise_model = DenoiseNN(
                input_dim=args.latent_dim,
                hidden_dim=args.hidden_dim_denoise,
                n_layers=args.n_layers_denoise,
                n_cond=args.n_condition,
                d_cond=args.dim_condition,
            ).to(device)
            optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

            denoise_model, training_history_denoiser = train_denoise_mlflow(
                args,
                denoise_model,
                autoencoder,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                device,
                torch.sqrt(torch.cumprod(1.0 - betas, axis=0)),
                torch.sqrt(1.0 - torch.cumprod(1.0 - betas, axis=0)),
            )

            # Log denoiser training and validation losses per epoch
            for epoch, (train_loss, val_loss) in enumerate(
                zip(
                    training_history_denoiser["train_loss_denoise"],
                    training_history_denoiser["val_loss_denoise"],
                ),
                start=1,
            ):
                mlflow.log_metric("train_loss_denoise", train_loss, step=epoch)
                mlflow.log_metric("val_loss_denoise", val_loss, step=epoch)

            denoise_model.eval()

            del train_loader, val_loader

        # Evaluate and log results
        metrics = check_results(
            args, device, autoencoder, denoise_model, test_loader, testset, betas
        )
        if metrics is not None:
            mlflow.log_metrics(metrics)
        else:
            print("Warning: Metrics returned by check_results are None. Skipping logging.")

    return autoencoder, denoise_model, metrics["mse_all_features"]
