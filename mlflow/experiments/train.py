import os
import gc
import torch
import mlflow
import mlflow.pytorch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from NGG.utils.utils import preprocess_dataset, linear_beta_schedule, generate_args_from_config
from NGG.train_utils.load_or_not_deepset import load_or_not_deepset
from NGG.train_utils.load_or_not_stat_model import load_or_not_stat_model
from NGG.train_utils.load_autoencoder import load_autoencoder
from NGG.train_utils.train_autoencoder import train_autoencoder
from NGG.train_utils.denoiser_train import train_denoise
from NGG.train_utils.check_results import check_results
from NGG.denoiser.denoise_model import DenoiseNN
from NGG.autoencoders.autoencoder_base import VariationalAutoEncoder
from NGG.autoencoders.autoencoder_concat import VariationalAutoEncoder_concat
from NGG.autoencoders.autoencoder_GMVAE import GMVAE
import yaml

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load configuration from params.yaml
with open(os.path.join(script_dir, "../config/params.yaml"), "r") as f:
    config = yaml.safe_load(f)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Map VAE types
VAE_mapper = {
    "base": VariationalAutoEncoder,
    "concat": VariationalAutoEncoder_concat,
    "features": "NotImplemented",
    "GMVAE": GMVAE,
}
VAE_class = VAE_mapper[config["model_config"]["AE"]]

# Generate args-like objects from config
args_list = generate_args_from_config(config)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("ngg_experiment")

# Iterate over each combination of hyperparameters
for args in args_list:
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

        # TODO: change train_autoencoder to return losses
        autoencoder = train_autoencoder(
            args, autoencoder, train_loader, val_loader, device, optimizer, scheduler
        )

        # Log autoencoder model
        input_example_autoencoder = trainset[0].x.unsqueeze(0).to(device).cpu().numpy()  # Convert to numpy
        mlflow.pytorch.log_model(autoencoder, "autoencoder", input_example=input_example_autoencoder)

        torch.cuda.empty_cache()
        gc.collect()

        # Define beta schedule
        betas = linear_beta_schedule(timesteps=args.timesteps)

        # Define alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

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

        # TODO: change train_denoise to return losses
        denoise_model = train_denoise(
            args,
            denoise_model,
            autoencoder,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            device,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
        )

        # Log denoising model
        input_example_denoise = torch.randn(1, args.latent_dim).to(device).cpu().numpy()  # Convert to numpy
        mlflow.pytorch.log_model(denoise_model, "denoise_model", input_example=input_example_denoise)

        denoise_model.eval()

        del train_loader, val_loader

        # Evaluate and log results
        metrics = check_results(
            args, device, autoencoder, denoise_model, test_loader, testset, betas
        )
        if metrics is not None:  # Ensure metrics is not None
            mlflow.log_metrics(metrics)
        else:
            print("Warning: Metrics returned by check_results are None. Skipping logging.")
