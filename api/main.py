import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse  # Import HTMLResponse
from argparse import Namespace  # Import Namespace for args generation
import logging  # Import logging for debugging

# Add the /app directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .models import CheckResultsRequest, CheckResultsResponse  # Use relative imports
import mlflow
import mlflow.pytorch
import torch
from NGG.train_utils.check_results import check_results
from NGG.utils.utils import preprocess_dataset
from torch_geometric.loader import DataLoader
from torch.serialization import add_safe_globals
from torch_geometric.data.data import DataEdgeAttr

# Allowlist the DataEdgeAttr class
add_safe_globals([DataEdgeAttr])

# Dynamically load the run_id from the file
RUN_ID_FILE = "/app/mlflow/experiments/best_run_id.txt"
try:
    with open(RUN_ID_FILE, "r") as f:
        DEFAULT_RUN_ID = f.read().strip()
except FileNotFoundError:
    raise RuntimeError(f"Run ID file not found at {RUN_ID_FILE}. Ensure evaluate.py has been run.")

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Initialize the FastAPI app
app = FastAPI(
    title="Graph Generation and Evaluation API",
    description=(
        "This API allows you to generate and evaluate graphs using machine learning models. "
        "It provides endpoints for running experiments and visualizing metrics. 🚀"
        "<br><br><img src=\"https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Graph_theory.svg/1200px-Graph_theory.svg.png\" width=\"200\">"
    ),
    version="1.0.0",
)

# Define a type mapping based on parser.py
TYPE_MAPPING = {
    "n_max_nodes": int,
    "spectral_emb_dim": int,
    "batch_size": int,
    "timesteps": int,
    "hidden_dim_encoder": int,
    "hidden_dim_decoder": int,
    "latent_dim": int,
    "n_layers_encoder": int,
    "n_layers_decoder": int,
    "hidden_dim_denoise": int,
    "n_layers_denoise": int,
    "dim_condition": int,
    "n_condition": int,
    "n_clusters": int,
    "lr": float,
    "dropout": float,
    "beta": float,
    "normalize": lambda x: x == "True",
    "labelize": lambda x: x == "True",
    "additional": lambda x: x == "True",
    "deepsets": lambda x: x == "True",
    "stats_model": lambda x: x == "True",
    "feature_concat": lambda x: x == "True",
    "constrain_denoiser": lambda x: x == "True",
    "early_stopping": lambda x: x == "True",
}

@app.get("/", tags=["Welcome"])
def show_welcome_page():
    """
    Show welcome page with API details.
    """
    return {
        "Message": "Welcome to the Graph Generation and Evaluation API!",
        "API_Title": "Graph Generation and Evaluation API",
        "Version": "1.0.0",
        "Description": "Use this API to generate and evaluate graphs using machine learning models.",
    }

@app.get("/check_results", response_class=HTMLResponse, tags=["Metrics"])
def show_default_metrics():
    """
    Display metrics for a default run_id.
    """
    try:
        mlflow.set_tracking_uri("http://172.17.0.1:5000")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Use the default run_id
        run_id = DEFAULT_RUN_ID
        autoencoder_path = f"runs:/{run_id}/best_autoencoder"
        denoise_model_path = f"runs:/{run_id}/best_denoise_model"

        # Load models with the allowlist
        autoencoder = mlflow.pytorch.load_model(autoencoder_path, map_location=device, weights_only=False)
        denoise_model = mlflow.pytorch.load_model(denoise_model_path, map_location=device, weights_only=False)
        params = mlflow.get_run(run_id).data.params

        # Convert params dictionary to Namespace with correct types
        typed_params = {
            key: TYPE_MAPPING.get(key, str)(value) if value != 'None' else None
            for key, value in params.items()
        }
        args = Namespace(**typed_params)

        autoencoder.to(device)
        denoise_model.to(device)

        # Preprocess the dataset using parameters
        testset = preprocess_dataset(
            dataset='test',
            n_max_nodes=args.n_max_nodes,
            spectral_emb_dim=args.spectral_emb_dim,
            normalize=args.normalize,
            labelize=args.labelize,
            additional_features_bool=args.additional,
        )
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

        # Define beta schedule
        betas = torch.linspace(0.0001, 0.02, args.timesteps)

        # Run check_results
        metrics = check_results(
            args=args,
            device=device,
            autoencoder=autoencoder,
            denoise_model=denoise_model,
            test_loader=test_loader,
            testset=testset,
            betas=betas,
        )

        # Format the metrics dictionary into an HTML table
        html_metrics = """
        <html>
        <head>
            <title>Metrics</title>
        </head>
        <body>
            <h1>Metrics for Default Run ID</h1>
            <table border='1'>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """
        for key, value in metrics.items():
            html_metrics += f"<tr><td>{key}</td><td>{value}</td></tr>"
        html_metrics += """
            </table>
        </body>
        </html>
        """

        return HTMLResponse(content=html_metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

logging.basicConfig(level=logging.DEBUG)

@app.post("/check_results", response_class=HTMLResponse, tags=["Metrics"])
def run_check_results(request: CheckResultsRequest):
    """
    Run the check_results function and display metrics in an HTML table.
    """
    try:
        logging.debug("Starting check_results...")
        mlflow.set_tracking_uri("http://172.17.0.1:5000")
        # Load models and parameters using the run_id
        run_id = request.run_id
        logging.debug(f"Using run_id: {run_id}")

        autoencoder_path = f"runs:/{run_id}/best_autoencoder"
        denoise_model_path = f"runs:/{run_id}/best_denoise_model"
        logging.debug(f"Loading models from: {autoencoder_path}, {denoise_model_path}")

        autoencoder = mlflow.pytorch.load_model(autoencoder_path)
        denoise_model = mlflow.pytorch.load_model(denoise_model_path)
        logging.debug("Models loaded successfully.")

        params = mlflow.get_run(run_id).data.params
        logging.debug(f"Parameters loaded: {params}")

        # Convert params dictionary to Namespace with correct types
        typed_params = {
            key: TYPE_MAPPING.get(key, str)(value) if value != 'None' else None
            for key, value in params.items()
        }
        args = Namespace(**typed_params)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        autoencoder.to(device)
        denoise_model.to(device)

        logging.debug("Preprocessing dataset...")
        # Preprocess the dataset using parameters
        testset = preprocess_dataset(
            dataset='test',
            n_max_nodes=args.n_max_nodes,
            spectral_emb_dim=args.spectral_emb_dim,
            normalize=args.normalize,
            labelize=args.labelize,
            additional_features_bool=args.additional,
        )
        logging.debug("Dataset preprocessed.")

        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
        logging.debug("DataLoader initialized.")

        # Define beta schedule
        betas = torch.linspace(0.0001, 0.02, args.timesteps)

        logging.debug("Running check_results...")
        # Run check_results
        metrics = check_results(
            args=args,
            device=device,
            autoencoder=autoencoder,
            denoise_model=denoise_model,
            test_loader=test_loader,
            testset=testset,
            betas=betas,
        )
        logging.debug(f"Metrics computed: {metrics}")

        # Format the metrics dictionary into an HTML table
        html_metrics = """
        <html>
        <head>
            <title>Metrics</title>
        </head>
        <body>
            <h1>Metrics</h1>
            <table border='1'>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """
        for key, value in metrics.items():
            html_metrics += f"<tr><td>{key}</td><td>{value}</td></tr>"
        html_metrics += """
            </table>
        </body>
        </html>
        """

        return HTMLResponse(content=html_metrics)
    except Exception as e:
        logging.error(f"Error in check_results: {e}")
        raise HTTPException(status_code=500, detail=str(e))
