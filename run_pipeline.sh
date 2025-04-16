#!/bin/bash

# Ensure the script is executable and in Unix format
set -e

# Step 1: Run evaluate.py to train and save the best models (optional)
# echo "Running evaluate.py to train and save the best models..."
# python /app/mlflow/experiments/evaluate.py

# Step 2: Launch the FastAPI application
echo "Launching the FastAPI application..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000