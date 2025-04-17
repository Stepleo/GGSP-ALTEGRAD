import os
import sys
import yaml
import mlflow
import torch
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from NGG.utils.utils import generate_args_from_config
from train import run_training

# Set logging level to DEBUG
logging.basicConfig(level=logging.DEBUG)

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load configuration from params.yaml
with open(os.path.join(script_dir, "../config/params.yaml"), "r") as f:
    config = yaml.safe_load(f)

# Generate args-like objects from config
args_list = generate_args_from_config(config)

mlflow.set_tracking_uri("https://user-lstepien-mlflow.user.lab.sspcloud.fr")  # Updated MLflow tracking URI
mlflow.set_experiment(config["model_config"]["name"])

# Track the best models and their MSE
best_mse = float("inf")
best_autoencoder = None
best_denoise_model = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Iterate over each combination of hyperparameters and call run_training
for args in args_list:
    autoencoder, denoise_model, mse = run_training(args, device)

    # Keep track of the best models based on MSE
    if mse < best_mse:
        best_mse = mse
        best_autoencoder = autoencoder
        best_denoise_model = denoise_model
        best_args = args

# Save the best models to MLFlow
with mlflow.start_run(run_name="best_models") as best_run:
    # Log the best MSE
    mlflow.log_metric("best_mse_all_features", best_mse)
    mlflow.log_params(vars(best_args))

    mlflow.pytorch.log_model(
        best_autoencoder,
        "best_autoencoder",
    )

    mlflow.pytorch.log_model(
        best_denoise_model,
        "best_denoise_model",
    )

    # Save the run_id of the best_models run to a file
    with open("/app/mlflow/experiments/best_run_id.txt", "w") as f:
        f.write(best_run.info.run_id)

print(f"Best MSE on all features: {best_mse}")
print(f"Best run_id: {best_run.info.run_id}")
