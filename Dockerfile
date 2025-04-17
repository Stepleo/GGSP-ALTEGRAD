# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Copy the mlflow directory
COPY ./mlflow /app/mlflow

# Copy the api directory
COPY ./api /app/api

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Install the module
RUN pip install -e .

# Copy the run_pipeline.sh script
COPY run_pipeline.sh /app/run_pipeline.sh

# Make the script executable
RUN chmod +x /app/run_pipeline.sh

# Ensure the script is in Unix format
RUN apt-get update && apt-get install -y dos2unix && dos2unix /app/run_pipeline.sh

# Expose the port FastAPI will run on
EXPOSE 8000

# Use ENTRYPOINT for script execution
ENTRYPOINT ["/bin/bash", "-c", "exec /app/run_pipeline.sh"]
