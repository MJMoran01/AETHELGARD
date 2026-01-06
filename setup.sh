#!/bin/bash
# =============================================================================
# Project AETHELGARD - Environment Setup Script
# =============================================================================
# This script creates the conda environment and installs all dependencies.
#
# Usage:
#   bash setup.sh         # CPU-only PyTorch (default)
#   bash setup.sh cuda118 # CUDA 11.8 PyTorch
#   bash setup.sh cuda121 # CUDA 12.1 PyTorch
# =============================================================================

set -e

# Default to CPU
PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"
PYTORCH_VARIANT="CPU"

if [ "$1" == "cuda118" ]; then
    PYTORCH_INDEX="https://download.pytorch.org/whl/cu118"
    PYTORCH_VARIANT="CUDA 11.8"
elif [ "$1" == "cuda121" ]; then
    PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
    PYTORCH_VARIANT="CUDA 12.1"
fi

echo "=============================================="
echo "  Project AETHELGARD Environment Setup"
echo "  PyTorch variant: $PYTORCH_VARIANT"
echo "=============================================="

# Create conda environment
echo "[1/3] Creating conda environment AETHEL..."
conda env create -f environment.yml --force

# Activate environment
echo "[2/3] Activating environment and installing PyTorch..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate AETHEL

# Install PyTorch
pip install torch torchvision torchaudio --index-url $PYTORCH_INDEX

# Verify installation
echo "[3/3] Verifying installation..."
python -c "
import torch
import scipy
import matplotlib
import tifffile
import numpy as np

print('\\n=== Installation Verified ===')
print(f'Python:     {__import__(\"sys\").version.split()[0]}')
print(f'PyTorch:    {torch.__version__}')
print(f'NumPy:      {np.__version__}')
print(f'SciPy:      {scipy.__version__}')
print(f'Matplotlib: {matplotlib.__version__}')
print(f'Tifffile:   {tifffile.__version__}')
print(f'CUDA:       {torch.cuda.is_available()}')
print('\\nAll dependencies installed successfully!')
"

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo ""
echo "  To activate the environment:"
echo "    conda activate AETHEL"
echo ""
echo "  Python interpreter path:"
echo "    $(which python)"
echo "=============================================="
