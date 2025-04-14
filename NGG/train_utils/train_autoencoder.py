import torch
from datetime import datetime
import numpy as np

def train_autoencoder(args, autoencoder, train_loader, val_loader, device, optimizer, scheduler):
    # Train VGAE model
    if args.train_autoencoder:
        best_val_loss = np.inf
        early_stop_counter = 0
        for epoch in range(1, args.epochs_autoencoder+1):
            autoencoder.train()
            train_loss_trackers = {}
            val_loss_trackers = {}
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()

                # Call loss function
                if args.feature_concat:
                    loss_dict = autoencoder.loss(
                        data,
                        args.beta,
                        args.contrastive_hyperparameters,
                        args.penalization_hyperparameters,
                        )
                else:
                    loss_dict = autoencoder.loss(
                        data,
                        *args.gmvae_loss_parameters,

                        )
                
                # Aggregate loss values dynamically
                for key, value in loss_dict.items():
                    if key not in train_loss_trackers:
                        train_loss_trackers[key] = 0.0
                    train_loss_trackers[key] += value.item()

                # Backpropagation
                loss_dict["loss"].backward()
                optimizer.step()

            # Calculate averages for the epoch
            train_count = len(train_loader.dataset)
            for key in train_loss_trackers:
                train_loss_trackers[key] /= train_count

            # Validation
            autoencoder.eval()
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)

                    # Call loss function
                    if args.AE!='GMVAE':
                        loss_dict = autoencoder.loss(
                            data,
                            args.beta,
                            args.contrastive_hyperparameters,
                            args.penalization_hyperparameters,
                            )
                    else:
                        loss_dict = autoencoder.loss(
                            data,
                            *args.gmvae_loss_parameters,
                            )

                    # Aggregate validation loss values
                    for key, value in loss_dict.items():
                        if key not in val_loss_trackers:
                            val_loss_trackers[key] = 0.0
                        val_loss_trackers[key] += value.item()

            # Calculate averages for validation
            val_count = len(val_loader.dataset)
            for key in val_loss_trackers:
                val_loss_trackers[key] /= val_count

            # Print losses dynamically
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            loss_str = ", ".join(
                [f"{key.capitalize()} Loss: {value:.5f}" for key, value in train_loss_trackers.items()]
            )
            print(f"{dt_t} Epoch: {epoch:04d}, Train {loss_str}")
            loss_str = ", ".join(
                [f"{key.capitalize()} Loss: {value:.5f}" for key, value in val_loss_trackers.items()]
            )
            print(f"{dt_t} Epoch: {epoch:04d}, Val {loss_str}")

            scheduler.step()

            if best_val_loss >= val_loss_trackers["loss"]:
                best_val_loss = val_loss_trackers["loss"]
                torch.save({
                    'state_dict': autoencoder.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, f'autoencoder_{args.AE}.pth.tar')
                
            if early_stop_counter >= 20:
                break
            
            if epoch > 20 and best_val_loss < val_loss_trackers["loss"]:
                early_stop_counter += int(args.early_stopping)

        torch.save({
                    'state_dict': autoencoder.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, f'autoencoder_{args.AE}_final.pth.tar')

    else:
        checkpoint = torch.load(f'autoencoder_{args.AE}.pth.tar')
        autoencoder.load_state_dict(checkpoint['state_dict'])
        
    autoencoder.eval()
    
    return autoencoder

def train_autoencoder_mlflow(args, autoencoder, train_loader, val_loader, device, optimizer, scheduler):
    """
    Train the autoencoder while tracking training and validation losses across epochs.
    Does not save the model to disk.
    """
    if args.train_autoencoder:
        best_val_loss = np.inf
        early_stop_counter = 0
        training_history = {"train_loss_autoencoder": [], "val_loss_autoencoder": []}

        for epoch in range(1, args.epochs_autoencoder + 1):
            autoencoder.train()
            train_loss_trackers = {}
            val_loss_trackers = {}

            # Training loop
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()

                # Call loss function
                if args.feature_concat:
                    loss_dict = autoencoder.loss(
                        data,
                        args.beta,
                        args.contrastive_hyperparameters,
                        args.penalization_hyperparameters,
                    )
                else:
                    loss_dict = autoencoder.loss(
                        data,
                        *args.gmvae_loss_parameters,
                    )

                # Aggregate loss values dynamically
                for key, value in loss_dict.items():
                    if key not in train_loss_trackers:
                        train_loss_trackers[key] = 0.0
                    train_loss_trackers[key] += value.item()

                # Backpropagation
                loss_dict["loss"].backward()
                optimizer.step()

            # Calculate averages for the epoch
            train_count = len(train_loader.dataset)
            for key in train_loss_trackers:
                train_loss_trackers[key] /= train_count

            # Validation loop
            autoencoder.eval()
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)

                    # Call loss function
                    if args.AE != "GMVAE":
                        loss_dict = autoencoder.loss(
                            data,
                            args.beta,
                            args.contrastive_hyperparameters,
                            args.penalization_hyperparameters,
                        )
                    else:
                        loss_dict = autoencoder.loss(
                            data,
                            *args.gmvae_loss_parameters,
                        )

                    # Aggregate validation loss values
                    for key, value in loss_dict.items():
                        if key not in val_loss_trackers:
                            val_loss_trackers[key] = 0.0
                        val_loss_trackers[key] += value.item()

            # Calculate averages for validation
            val_count = len(val_loader.dataset)
            for key in val_loss_trackers:
                val_loss_trackers[key] /= val_count

            # Append losses to history
            training_history["train_loss_autoencoder"].append(train_loss_trackers["loss"])
            training_history["val_loss_autoencoder"].append(val_loss_trackers["loss"])

            # Print losses dynamically
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            loss_str = ", ".join(
                [f"{key.capitalize()} Loss: {value:.5f}" for key, value in train_loss_trackers.items()]
            )
            print(f"{dt_t} Epoch: {epoch:04d}, Train {loss_str}")
            loss_str = ", ".join(
                [f"{key.capitalize()} Loss: {value:.5f}" for key, value in val_loss_trackers.items()]
            )
            print(f"{dt_t} Epoch: {epoch:04d}, Val {loss_str}")

            scheduler.step()

            # Early stopping logic
            if best_val_loss >= val_loss_trackers["loss"]:
                best_val_loss = val_loss_trackers["loss"]

            if early_stop_counter >= 20:
                break

            if epoch > 20 and best_val_loss < val_loss_trackers["loss"]:
                early_stop_counter += int(args.early_stopping)

    autoencoder.eval()
    return autoencoder, training_history