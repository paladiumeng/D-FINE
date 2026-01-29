# syntax=docker/dockerfile:1.4
# Single-stage optimized Dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_HTTP_TIMEOUT=600 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CUDA_LAUNCH_BLOCKING=1 \
    PATH="/root/.local/bin/:$PATH"

# Install everything in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    software-properties-common \
    git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3.11-distutils \
    python3-pip && \
    # Clean up some build deps we don't need at runtime
    apt-get remove -y software-properties-common && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install uv with cache mount
RUN --mount=type=cache,target=/root/.cache \
    curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock README.md .python-version ./

# Create package structure
RUN mkdir -p src/zoo src/nn src/data src/core src/solver src/misc

# Install dependencies with cache mounts (biggest time saver)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv python install && \
    uv sync --frozen --no-dev

# Copy source code and configs
COPY src ./src
COPY configs ./configs
COPY tools ./tools
COPY train.py ./train.py
COPY weights ./weights

# Copy Vertex AI scripts
COPY vertex_ai/entrypoint.sh /app/entrypoint.sh
COPY vertex_ai/download_gcs_data.py /app/vertex_ai/download_gcs_data.py
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]