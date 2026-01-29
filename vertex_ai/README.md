# D-FINE Vehicle Detection Training on Vertex AI

This guide explains how to train your D-FINE vehicle detection model on Google Cloud Vertex AI.

## Prerequisites

1. **Google Cloud CLI** installed and configured
2. **Docker** installed for building container images
3. **Appropriate permissions** in the `production-paladium` project
4. **Dataset prepared** and ready to upload to GCS

## Step-by-Step Instructions

### 1. Upload Dataset to Google Cloud Storage

First, upload your prepared COCO dataset to GCS:

```bash
# Create bucket directory if it doesn't exist
gsutil mb -p production-paladium gs://cv_models_dir/vehicle_detection/ 2>/dev/null || true

# Upload your dataset
gsutil -m cp -r data/vehicle_full_coco gs://cv_models_dir/vehicle_detection/data/

# Verify upload
gsutil ls gs://cv_models_dir/vehicle_detection/data/vehicle_full_coco/
```

### 2. Build Docker Image

Build the Docker container with all D-FINE dependencies:

```bash
# Navigate to D-FINE root directory
cd /Users/gabrielnascimento/ml_only/D-FINE

# Build the Docker image
docker build \
  -f vertex_ai/Dockerfile \
  -t us-central1-docker.pkg.dev/production-paladium/cv-models-envs/dfine-vehicle-detection:latest \
  --platform linux/amd64 \
  .
```

### 3. Push Docker Image to Artifact Registry

Authenticate and push the image:

```bash
# Configure Docker for GCP Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Push the image
docker push us-central1-docker.pkg.dev/production-paladium/cv-models-envs/dfine-vehicle-detection:latest
```

### 4. Configure Training Parameters

Edit `vertex_ai/vertex.yaml` if you need to adjust:
- Machine type (default: `n1-highmem-8`)
- GPU type (default: `NVIDIA_TESLA_V100`)
- Training arguments (epochs, batch size, etc.)
- Output directory in GCS

### 5. Authenticate with Google Cloud

```bash
# Login to Google Cloud
gcloud auth login

# Set default project
gcloud config set project production-paladium

# Create application default credentials
gcloud auth application-default login
```

### 6. (Optional) Set up WandB for Experiment Tracking

If you want to use Weights & Biases for tracking:

```bash
export WANDB_API_KEY=your_wandb_api_key_here
```

### 7. Submit Training Job to Vertex AI

Run the training job:

```bash
# Basic training
python vertex_ai/run_on_vertex_ai.py

# Training with custom parameters
python vertex_ai/run_on_vertex_ai.py --epochs 100 --batch-size 64 --checkpoint-freq 5

# Evaluation only (if you have a checkpoint)
python vertex_ai/run_on_vertex_ai.py --test-only
```

