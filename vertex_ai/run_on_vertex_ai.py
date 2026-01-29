#!/usr/bin/env python3
"""
Script to run D-FINE training on Google Vertex AI
"""

import os
import sys
import yaml
from typing import Optional
from google.cloud import aiplatform
from google.cloud.aiplatform import CustomJob
from google.cloud.aiplatform_v1.types import (
    ContainerSpec,
    EnvVar,
    MachineSpec,
    WorkerPoolSpec,
)


def load_vertex_config(config_path: str = "vertex_ai/vertex.yaml"):
    """Load Vertex AI configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_vertex_ai_job(
    project_id: str,
    location: str = "us-central1",
    display_name: str = "dfine-training",
    machine_type: str = "n1-standard-8",
    accelerator_type: str = "NVIDIA_TESLA_T4",
    accelerator_count: int = 1,
    replica_count: int = 1,
    service_account: Optional[str] = None,
    labels: Optional[dict] = None,
    container_image_uri: Optional[str] = None,
    args: Optional[list] = None,
    env: Optional[list] = None,
    gcs_data_path: Optional[str] = None,
):
    """
    Create and run a custom job on Vertex AI

    Args:
        project_id: Google Cloud project ID
        location: Vertex AI region
        display_name: Name for the job
        machine_type: Machine type for training
        accelerator_type: GPU accelerator type
        accelerator_count: Number of accelerators
        replica_count: Number of replicas
        service_account: Service account email to run the job
        labels: Dictionary of labels to attach to the job
        container_image_uri: Container image URI
        args: Arguments to pass to the training script
        env: Environment variables
        gcs_data_path: GCS path to dataset
    """

    # Initialize Vertex AI
    aiplatform.init(
        project=project_id,
        location=location,
        staging_bucket="gs://cv_models_dir/"
    )

    # Configure machine spec
    machine_spec = MachineSpec(
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
    )

    # Build environment variables
    env_vars = env or []
    if gcs_data_path:
        env_vars.append(EnvVar(name="GCS_DATA_PATH", value=gcs_data_path))

    # Configure container spec
    container_spec = ContainerSpec(
        image_uri=container_image_uri,
        args=args or [],
        env=env_vars if env_vars else None,
    )

    # Configure worker pool
    worker_pool_specs = [
        WorkerPoolSpec(
            machine_spec=machine_spec,
            replica_count=replica_count,
            container_spec=container_spec,
        )
    ]

    # Create the custom job
    job = CustomJob(
        display_name=display_name,
        worker_pool_specs=worker_pool_specs,
        project=project_id,
        location=location,
        labels=labels,
    )

    if service_account:
        job._gca_resource.job_spec.service_account = service_account

    print(f"Starting job: {display_name}")
    print(f"Container image: {container_image_uri}")
    print(f"Machine type: {machine_type}")
    print(f"GPU: {accelerator_type} x {accelerator_count}")
    print(f"Training args: {' '.join(args or [])}")

    job.submit()

    return job


def main():
    # Load configuration
    config = load_vertex_config()

    # Allow environment variable overrides
    project_id = os.getenv("GCP_PROJECT", config["project_id"])

    # Check for WandB API key (optional)
    wandb_key = os.getenv("WANDB_API_KEY", config.get("wandb_api_key"))

    # Build environment variables
    env_vars = []
    if wandb_key:
        env_vars.append(EnvVar(name="WANDB_API_KEY", value=wandb_key))
        print("✓ WandB API key configured")

    # Check for container image
    if not config.get("container_image_uri"):
        print("❌ Error: container_image_uri not set in vertex.yaml")
        print("Please build and push the Docker image first:")
        print("  1. Build: docker build -f vertex_ai/Dockerfile -t <image_uri> .")
        print("  2. Push: docker push <image_uri>")
        print("  3. Update vertex.yaml with the image URI")
        sys.exit(1)

    # Parse command line arguments for overrides
    import argparse
    parser = argparse.ArgumentParser(description="Run D-FINE training on Vertex AI")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--checkpoint-freq", type=int, help="Checkpoint frequency")
    parser.add_argument("--test-only", action="store_true", help="Run evaluation only")
    cli_args = parser.parse_args()

    # Build training arguments
    training_args = config.get("args", []).copy()

    # Add CLI overrides
    if cli_args.epochs:
        training_args.extend(["--update", f"epochs={cli_args.epochs}"])
    if cli_args.batch_size:
        training_args.extend(["--update", f"train_dataloader.total_batch_size={cli_args.batch_size}"])
    if cli_args.checkpoint_freq:
        training_args.extend(["--update", f"checkpoint_freq={cli_args.checkpoint_freq}"])
    if cli_args.test_only:
        training_args.append("--test-only")

    try:
        job = create_vertex_ai_job(
            project_id=project_id,
            location=config["location"],
            display_name=config["display_name"],
            machine_type=config["machine_type"],
            accelerator_type=config["accelerator_type"],
            accelerator_count=config["accelerator_count"],
            replica_count=config["replica_count"],
            service_account=config.get("service_account"),
            container_image_uri=config["container_image_uri"],
            labels=config.get("labels", {}),
            args=training_args,
            env=env_vars,
            gcs_data_path=config.get("gcs_data_path"),
        )

        print("\n✓ Job submitted successfully!")
        print(f"  Job name: {job.name}")
        print(f"  Display name: {job.display_name}")
        print(f"  State: {job.state}")
        print(f"\nMonitor progress at:")
        print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project_id}")

    except Exception as e:
        print(f"\n❌ Error creating/running job: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
