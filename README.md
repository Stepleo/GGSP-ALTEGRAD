# Neural Graph Generator (NGG) Project

## Overview

This project revolves around a module named **NGG** (Neural Graph Generator), which contains models designed to generate graphs based on 7 key graph properties:

1. **Number of nodes**
2. **Number of edges**
3. **Number of triangles**
4. **Average degree of the graph nodes**
5. **Graph max k-core**
6. **Global clustering coefficient**
7. **Number of communities**

For more details about these models, please refer to the **Project-report.pdf** file and the code in the `NGG` folder.

---

## Workflow Description

### **Training and Evaluation**
The workflow is centered around the file `mlflow/experiments/evaluate.py`. Here's how it works:
1. **Hyperparameter Configuration**:
   - A set of hyperparameters for training the models is defined in `mlflow/config/params.yaml`.
2. **Training**:
   - For each combination of hyperparameters, models are trained using cross-validation on a dataset of **9000 graphs**.
   - The dataset is accessible [here](https://minio.lab.sspcloud.fr/lstepien/NGG/data.tar.gz).
3. **Logging**:
   - Training losses and metrics are logged to **MLflow**.
4. **Model Selection**:
   - The best model is selected based on its performance on a test set of **1000 graphs**.
   - The best model is saved to MLflow for future use.

### **Try It Yourself**
To try the workflow locally:
1. Clone the repository.
2. Start by setting up the environment by runnning:

``` bash
pip install -r requirements.txt
```

setup the module NGG by using 

``` bash
pip install -e .
```
3. Run
```bash
python mlflow/experiments/evaluate.py
```
 To train the models from the module manually you can use NGG/main.py with this command 

``` bash
python NGG/main.py --n-layers-decoder 6 --n-layers-encoder 4 --n-layers-denoise 4 \
--epochs-denoise 200 --epochs-autoencoder 200 --AE concat (or GMVAE) \
--name $Name_of_experiment --timesteps 1000 --additional
```

- ```--penalization-hyperparameters 1 ``` #to add MSE losses for n_nodes, n_triagles and n_edges
- ```--normalize ``` #adds self loops to the adj matrices
You can modify the configurations in `mlflow/config/params.yaml` to experiment with different hyperparameter settings.

## Integrated Workflow with GitHub Actions and Kubernetes

To automate the workflow, we set up **GitHub Actions** and **Kubernetes** on SSPCloud. Here's how it works:

### **GitHub Actions**:
- A GitHub Actions workflow builds a Docker container for the project.
- The container is pushed to Docker Hub.
- The workflow then launches a Kubernetes cluster on SSPCloud.

### **Kubernetes Cluster**:
- The cluster uses the Docker container to run `evaluate.py` on SSPCloud servers.
- Metrics and models are logged to MLflow at: [https://user-lstepien-mlflow.user.lab.sspcloud.fr](https://user-lstepien-mlflow.user.lab.sspcloud.fr).

### **FastAPI for Model Serving**:
- After training, the best model saved by `evaluate.py` is served using **FastAPI**.
- The API is accessible at: [https://conditionned-graph-generation.lab.sspcloud.fr/](https://conditionned-graph-generation.lab.sspcloud.fr/).

---

## **API Details**

The FastAPI application currently supports one endpoint:

- **`/check_results`**:
  - This endpoint runs the best model on the test set and displays metrics regarding how well the generated graphs match the desired properties.

To access this endpoint, append `/check_results` to the API URL:  
[https://conditionned-graph-generation.lab.sspcloud.fr/check_results](https://conditionned-graph-generation.lab.sspcloud.fr/check_results).

---

## **Summary**

This project provides an end-to-end solution for training, evaluating, and serving graph generation models:

- **Training**: Models are trained and evaluated using `evaluate.py`, with metrics logged to MLflow.
- **Automation**: GitHub Actions and Kubernetes automate the workflow, running the training process on SSPCloud.
- **Serving**: The best model is served via FastAPI, allowing users to evaluate its performance interactively.

Feel free to explore the repository, modify configurations, and experiment with the models!


## Folder structure

```
.Repo
├── NGG                    # Main module folder for the NGG project
│   ├── main.py            # Entry point or table of contents for the project
│   ├── train_utils        # Utilities and scripts for training models
│   │   ├── parser.py               # Script for parsing input arguments
│   │   ├── load_or_not_deepset.py  # Logic for loading or skipping DeepSets models
│   │   ├── load_or_not_stat_model.py # Logic for loading or skipping statistical models
│   │   ├── load_autoencoder.py     # Script for loading autoencoder models
│   │   ├── train_autoencoder.py    # Script for training autoencoder models
│   │   ├── denoiser_train.py       # Script for training denoiser models
│   │   └── check_results.py        # Script for checking training results
│   ├── utils              # Utility functions and scripts for general use
│   │   ├── extract_feats.py        # Script for extracting features
│   │   ├── utils.py                # General utility functions
│   │   ├── verify_dataset_distribution.py # Script to verify dataset distribution
│   │   └── verify_graph_features.py       # Script to verify graph features
│   ├── autoencoders       # Folder containing autoencoder implementations
│   │   ├── autoencoder_base.py      # Base class for autoencoders
│   │   ├── autoencoder_concat.py    # Concatenation-based autoencoder
│   │   ├── autoencoder_GMVAE.py     # Gaussian Mixture VAE implementation
│   │   ├── autoencoder_GMVAEv2.py   # Version 2 of Gaussian Mixture VAE
│   │   ├── components               # Sub-folder for components used in autoencoders
│   │   │   ├── deepsets.py            # DeepSets component
│   │   │   ├── encoders               # Sub-folder for encoder components
│   │   │   │   ├── GIN_base.py          # Base GIN encoder
│   │   │   │   └── GIN_concat.py        # GIN encoder with concatenation
│   │   │   ├── decoders               # Sub-folder for decoder components
│   │   │   │   ├── decoder_base.py      # Base decoder
│   │   │   │   └── decoder_norm.py      # Normalization-based decoder
│   ├── denoisers          # Folder for denoiser models
│   │   └── denoise_model.py      # Denoiser model implementation
│   └── commands.txt       # File containing various command line commands for reference
├── api                    # FastAPI application for serving the best model
│   ├── main.py            # FastAPI entry point
│   ├── models.py          # Request and response models for the API
├── mlflow                 # MLflow-related scripts and configurations
│   ├── experiments        # Folder for training and evaluation scripts
│   │   ├── evaluate.py    # Script for training and saving the best models
│   │   ├── train.py       # Script for training models
│   ├── config             # Configuration files for MLflow
│   │   └── params.yaml    # Hyperparameter configurations
├── data                   # Folder to store datasets
├── deployment             # Kubernetes deployment files
│   ├── deployment.yaml    # Deployment configuration for Kubernetes
│   ├── service.yaml       # Service configuration for FastAPI
│   ├── mlflow-service.yaml # Service configuration for MLflow
│   ├── ingress.yaml       # Ingress configuration for routing
├── progression_archive    # Folder for archiving progression and old versions
├── model_weights          # Folder to store model weights
├── run_pipeline.sh        # Script to run the pipeline (evaluate.py and FastAPI)
├── Dockerfile             # Dockerfile for building the project container
├── requirements.txt       # Python dependencies for the project
└── setup.py               # Setup script for installing dependencies and packages
```


