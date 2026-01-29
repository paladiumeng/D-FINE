#!/usr/bin/env bash
set -euo pipefail

cd /app

# Default to train.py if TRAINING_CMD not set
CMD="${TRAINING_CMD:-uv run python train.py}"

# Download dataset from GCS if GCS_DATA_PATH is set
if [[ -n "${GCS_DATA_PATH:-}" ]]; then
  echo "Downloading dataset from GCS: ${GCS_DATA_PATH}"
  # Use uv run to execute Python script with all dependencies
  uv run python /app/vertex_ai/download_gcs_data.py
fi

# Generate a random 8-character ID for this run
RUN_ID=$(openssl rand -hex 4)
echo "Generated unique run ID: ${RUN_ID}"

# Process arguments and add random ID to output directory
ARGS=()
OUTPUT_DIR_SET=false

# Parse arguments to find and modify --output-dir
for arg in "$@"; do
  if [[ "$arg" == "--output-dir"* ]]; then
    if [[ "$arg" == "--output-dir="* ]]; then
      # Format: --output-dir=/path/to/output
      BASE_DIR="${arg#--output-dir=}"
      ARGS+=("--output-dir=${BASE_DIR}/${RUN_ID}")
      OUTPUT_DIR_SET=true
    else
      # Format: --output-dir /path/to/output (next arg is the path)
      ARGS+=("$arg")
      NEXT_IS_OUTPUT=true
    fi
  elif [[ "${NEXT_IS_OUTPUT:-false}" == "true" ]]; then
    # This is the output directory path
    ARGS+=("${arg}/${RUN_ID}")
    NEXT_IS_OUTPUT=false
    OUTPUT_DIR_SET=true
  else
    ARGS+=("$arg")
  fi
done

# If no output-dir was specified in args, add one with random ID
if [[ "$OUTPUT_DIR_SET" == "false" ]] && [[ ${#ARGS[@]} -gt 0 ]]; then
  ARGS+=("--output-dir" "/gcs/cv_models_dir/vehicle_detection/outputs/${RUN_ID}")
fi

# Accept arguments passed to the container
# If script arguments exist, use them; otherwise fall back to EXTRA_ARGS env var
if [[ ${#ARGS[@]} -gt 0 ]]; then
  echo "Running D-FINE training with unique output: ${ARGS[@]}"
  exec $CMD "${ARGS[@]}"
else
  EXTRA_ARGS="${TRAINING_EXTRA_ARGS:-}"
  echo "Running D-FINE training: $CMD $EXTRA_ARGS"
  exec $CMD $EXTRA_ARGS
fi