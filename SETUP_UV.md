# D-FINE Setup with uv on macOS

This guide will help you set up D-FINE training environment using `uv` package manager and train on your custom vehicle dataset.

## Prerequisites

- macOS
- `uv` package manager installed (if not installed, see Installation section below)

## Installation

### 1. Install uv

If you don't have `uv` installed, install it using:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or using Homebrew:

```bash
brew install uv
```

### 2. Set up Python Environment

The project requires Python 3.11.9. uv will automatically manage the Python version:

```bash
# Install dependencies (uv will automatically create a virtual environment)
uv sync
```

This will:
- Create a virtual environment with Python 3.11.9
- Install all required dependencies from `pyproject.toml`

### 3. Verify Installation

You can verify the installation by running:

```bash
uv run python --version
```

## Dataset Preparation

### Convert YOLO Format to COCO Format

Your dataset is currently in YOLO format. Convert it to COCO format using the provided script:

```bash
uv run python tools/dataset/yolo_to_coco.py \
  --images-dir data/vehicle_dataset/images \
  --labels-dir data/vehicle_dataset/labels \
  --label-list data/vehicle_dataset/label_list.txt \
  --output-dir data/vehicle_dataset \
  --train-ratio 0.9 \
  --seed 727
```

This will:
- Convert YOLO format labels to COCO format
- Split dataset into 90% train / 10% validation
- Create the following directory structure:
  ```
  data/vehicle_dataset/
  ├── train/
  │   ├── images/
  │   └── annotations/
  │       └── instances_train.json
  └── val/
      ├── images/
      └── annotations/
          └── instances_val.json
  ```

### Dataset Classes

Your vehicle dataset has 7 classes:
1. car
2. motorcycle
3. bus
4. truck
5. trailer
6. bicycle
7. van

## Training

### Download Pretrained Weights

Download the COCO pretrained weights for D-FINE-N:

```bash
# Create weights directory
mkdir -p weights

# Download D-FINE-N pretrained on COCO
curl -L -o weights/dfine_n_coco.pth \
  https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth
```

**Note:** D-FINE-N is only available pretrained on COCO (not Objects365). If you prefer Objects365 pretraining, consider using D-FINE-S or larger models. You can also train from scratch by omitting the `-t` flag.

### Training Commands

#### Single GPU Training

```bash
uv run torchrun --master_port=7777 --nproc_per_node=1 train.py \
  -c configs/dfine/custom/dfine_hgnetv2_n_custom.yml \
  --use-amp --seed=0 \
  -t weights/dfine_n_coco.pth
```

#### Multi-GPU Training (if available)

```bash
uv run torchrun --master_port=7777 --nproc_per_node=4 train.py \
  -c configs/dfine/custom/dfine_hgnetv2_n_custom.yml \
  --use-amp --seed=0 \
  -t weights/dfine_n_coco.pth
```

Replace `--nproc_per_node=4` with the number of GPUs you have.

### Training from Scratch (without pretrained weights)

If you prefer to train from scratch:

```bash
uv run torchrun --master_port=7777 --nproc_per_node=1 train.py \
  -c configs/dfine/custom/dfine_hgnetv2_n_custom.yml \
  --use-amp --seed=0
```

## Testing/Evaluation

After training, evaluate your model:

```bash
uv run torchrun --master_port=7777 --nproc_per_node=1 train.py \
  -c configs/dfine/custom/dfine_hgnetv2_n_custom.yml \
  --test-only \
  -r output/dfine_hgnetv2_n_custom/checkpoints/checkpoint.pth
```

## Configuration Files

- **Dataset config**: `configs/dataset/custom_detection.yml`
  - Contains dataset paths and number of classes (7)
  
- **Model config**: `configs/dfine/custom/dfine_hgnetv2_n_custom.yml`
  - Contains D-FINE-N model configuration
  - References the custom dataset config

## Troubleshooting

### Issue: uv command not found

Make sure `uv` is installed and in your PATH. After installation, you may need to restart your terminal or run:

```bash
source $HOME/.cargo/env  # if installed via curl
```

### Issue: Python version mismatch

uv will automatically use Python 3.11.9 as specified in `pyproject.toml`. If you encounter issues, check:

```bash
uv run python --version
```

### Issue: CUDA/GPU not available

If you're on macOS without CUDA support, the training will use CPU (which will be slow). For GPU training, you'll need a Linux machine with CUDA support.

### Issue: Out of memory

If you encounter out-of-memory errors, reduce the batch size in `configs/dfine/include/dataloader.yml`:

```yaml
train_dataloader:
  total_batch_size: 16  # Reduce from default 32
```

## Model Variants

You can use different model sizes by changing the config file:

- **D-FINE-N** (smallest, fastest): `configs/dfine/custom/dfine_hgnetv2_n_custom.yml`
- **D-FINE-S**: `configs/dfine/custom/dfine_hgnetv2_s_custom.yml`
- **D-FINE-M**: `configs/dfine/custom/dfine_hgnetv2_m_custom.yml`
- **D-FINE-L**: `configs/dfine/custom/dfine_hgnetv2_l_custom.yml`
- **D-FINE-X** (largest): `configs/dfine/custom/dfine_hgnetv2_x_custom.yml`

## Additional Resources

- Main README: [README.md](README.md)
- D-FINE Paper: [arXiv:2410.13842](https://arxiv.org/abs/2410.13842)
- Original Repository: [D-FINE GitHub](https://github.com/Peterande/D-FINE)
