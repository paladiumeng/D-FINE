#!/usr/bin/env python3
"""
Download data from Google Cloud Storage to local directory.
Replaces gsutil CLI with Python implementation.
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage
from google.cloud.exceptions import NotFound
from tqdm import tqdm


def parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """Parse GCS path into bucket and prefix.

    Args:
        gcs_path: GCS path like gs://bucket/path/to/data

    Returns:
        Tuple of (bucket_name, prefix)
    """
    if not gcs_path.startswith('gs://'):
        raise ValueError(f"Invalid GCS path: {gcs_path}. Must start with gs://")

    # Remove gs:// prefix and split
    path_without_prefix = gcs_path[5:]
    if '/' in path_without_prefix:
        bucket_name, prefix = path_without_prefix.split('/', 1)
        # Remove trailing /* if present (common pattern)
        if prefix.endswith('/*'):
            prefix = prefix[:-2]
        # Ensure prefix ends with / for proper directory listing
        if prefix and not prefix.endswith('/'):
            prefix = prefix + '/'
    else:
        bucket_name = path_without_prefix
        prefix = ''

    return bucket_name, prefix


def download_blob(blob, local_base_path: Path, prefix: str) -> str:
    """Download a single blob to local filesystem.

    Args:
        blob: GCS blob object
        local_base_path: Base local directory path
        prefix: GCS prefix to strip from blob name

    Returns:
        Local file path where blob was downloaded
    """
    # Calculate relative path by removing prefix
    relative_path = blob.name
    if prefix:
        relative_path = blob.name[len(prefix):]

    # Skip if relative path is empty (happens for directory markers)
    if not relative_path:
        return None

    local_path = local_base_path / relative_path

    # Create parent directories if needed
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Download the blob
    blob.download_to_filename(str(local_path))
    return str(local_path)


def download_gcs_directory(gcs_path: str, local_path: str = 'data', max_workers: int = 10):
    """Download all files from a GCS path to local directory.

    Args:
        gcs_path: GCS path like gs://bucket/path/to/data/*
        local_path: Local directory to download to (default: 'data')
        max_workers: Number of parallel download threads
    """
    print(f"Downloading from GCS: {gcs_path}")

    # Parse GCS path
    bucket_name, prefix = parse_gcs_path(gcs_path)
    print(f"Bucket: {bucket_name}, Prefix: {prefix or '(root)'}")

    # Initialize GCS client
    client = storage.Client()

    try:
        bucket = client.bucket(bucket_name)
    except NotFound:
        print(f"Error: Bucket '{bucket_name}' not found")
        sys.exit(1)

    # List all blobs with the prefix
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        print(f"Warning: No objects found at {gcs_path}")
        return

    print(f"Found {len(blobs)} objects to download")

    # Create local directory
    local_base_path = Path(local_path)
    local_base_path.mkdir(parents=True, exist_ok=True)

    # Download blobs in parallel
    downloaded_files = []
    failed_downloads = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_blob = {
            executor.submit(download_blob, blob, local_base_path, prefix): blob
            for blob in blobs
        }

        # Process completed downloads with progress bar
        with tqdm(total=len(blobs), desc="Downloading files") as pbar:
            for future in as_completed(future_to_blob):
                blob = future_to_blob[future]
                try:
                    local_file = future.result()
                    if local_file:  # Skip None results (directory markers)
                        downloaded_files.append(local_file)
                except Exception as e:
                    print(f"\nError downloading {blob.name}: {e}")
                    failed_downloads.append(blob.name)
                finally:
                    pbar.update(1)

    # Print summary
    print(f"\nDownload complete!")
    print(f"Successfully downloaded: {len(downloaded_files)} files")
    if failed_downloads:
        print(f"Failed downloads: {len(failed_downloads)} files")
        for failed in failed_downloads[:5]:  # Show first 5 failures
            print(f"  - {failed}")
        if len(failed_downloads) > 5:
            print(f"  ... and {len(failed_downloads) - 5} more")

    # List downloaded directory structure
    print(f"\nDownloaded to {local_path}/:")
    os.system(f"ls -la {local_path}/ | head -20")


def main():
    """Main entry point for standalone script usage."""
    # Get GCS path from environment variable
    gcs_path = os.environ.get('GCS_DATA_PATH')

    if not gcs_path:
        print("GCS_DATA_PATH environment variable not set, skipping download")
        return

    # Download the data
    download_gcs_directory(gcs_path)


if __name__ == "__main__":
    main()