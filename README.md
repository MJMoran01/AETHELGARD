# Project AETHELGARD

**Physics-Gated Attention Network for Dual-Energy X-Ray Security Imaging**

A Glass Box approach to X-ray threat detection using physics-constrained deep learning.

> "Physics filters the Signal; AI interprets the Texture."

---

## Overview

AETHELGARD implements a dual-stream neural network architecture that fuses traditional signal processing with deep learning for X-ray security imaging. Unlike black-box CNNs, our approach explicitly encodes the physics of dual-energy X-ray attenuation into differentiable layers.

### Key Features

- **PhysicsHead**: Differentiable Alvarez-Macovski decomposition layer
- **Material Decomposition**: Separates photoelectric (Z-dependent) and Compton (density-dependent) components
- **Z_eff Estimation**: Derives effective atomic number from dual-energy measurements
- **Material Edge Detection**: Thickness-invariant ratio gradient for compositional boundaries
- **Pre-log Gaussian Blur**: Physics-informed denoising for Poisson shot noise

---

## Architecture

```
Input: Dual-Energy X-Ray Images
        │
        ▼
┌───────────────────┐
│ RawToLogAttenuation│  ← Gaussian blur + Beer-Lambert linearization
│   (Preprocessing)  │
└────────┬──────────┘
         │
         ▼
    (B, 2, H, W)
    [L_low, L_high]
         │
         ▼
┌───────────────────┐
│    PhysicsHead    │  ← Polynomial decomposition
│                   │  ← Z_eff = (A₁/A₂)^(1/3)
│                   │  ← Ratio gradient ∇R
└────────┬──────────┘
         │
         ▼
    (B, 4, H, W)
    [A₁, A₂, Z_eff, ∇R]
         │
         ▼
┌───────────────────┐
│  VisionHead (CNN) │  ← [Future: ResNet backbone]
│  + PhysicsAttention│
└────────┬──────────┘
         │
         ▼
   Classification/Detection
```

---

## Physics Background

### Beer-Lambert Law
```
I = I₀ · exp(-μt)
L = -ln(I/I₀) = μt
```

### Alvarez-Macovski Decomposition (1976)
In the diagnostic energy range (30-200 keV), attenuation decomposes into:
```
μ(E) = a₁·f₁(E) + a₂·f₂(E)
```
Where:
- `a₁` = Photoelectric coefficient ∝ ρZ³
- `a₂` = Compton coefficient ∝ ρ
- `f₁(E)` = 1/E³ (photoelectric energy dependence)
- `f₂(E)` = Klein-Nishina cross-section

### Effective Atomic Number
```
Z_eff = (A₁/A₂)^(1/3)
```
The 1/3 exponent comes from inverting the Z³ scaling of the photoelectric effect.

---

## Installation

```bash
# Clone the repository
git clone https://github.gatech.edu/[your-username]/aethelgard.git
cd aethelgard

# Install dependencies
pip install torch numpy matplotlib scipy tifffile
```

---

## Usage

### Basic Usage

```python
from aethelgard import RawToLogAttenuation, PhysicsHead, DualEnergyDataset
import torch

# Load dual-energy images
dataset = DualEnergyDataset(
    data_dir="HUMS-X-ray-Dataset/HighLow",
    preprocess=True
)

# Get a sample
sample = dataset[0]
image = sample['image'].unsqueeze(0)  # (1, 2, H, W)

# Apply physics decomposition
physics_head = PhysicsHead(order=2, learnable=False)
physics_maps = physics_head(image)  # (1, 4, H, W)

# Extract features
A1 = physics_maps[:, 0]      # Photoelectric coefficient
A2 = physics_maps[:, 1]      # Compton coefficient
Z_eff = physics_maps[:, 2]   # Effective atomic number
grad_R = physics_maps[:, 3]  # Material edge detector
```

### Visualization

```bash
python scripts/visualize_physics.py
```

This generates diagnostic visualizations in `outputs/`:
- Raw intensity images
- Log-attenuation maps
- Physics feature maps (A₁, A₂, Z_eff, ∇R)

---

## Project Structure

```
aethelgard/
├── __init__.py          # Package exports
├── preprocessing.py     # RawToLogAttenuation (Beer-Lambert + Gaussian blur)
├── physics_head.py      # PhysicsHead (Alvarez-Macovski decomposition)
└── dataset.py           # DualEnergyDataset (HUMS data loader)

scripts/
└── visualize_physics.py # Validation and visualization

outputs/                 # Generated visualizations (gitignored)

HUMS-X-ray-Dataset/      # Dual-energy X-ray images (gitignored)
├── HighLow/             # Paired _hi.tif / _lo.tif files
├── Annotation/          # Pascal VOC format annotations
├── NotThreats/          # Negative samples
└── ThreatsRGB/          # Positive samples (RGB pseudocolor)
```

---

## Dataset

This project uses the **HUMS Dual-Energy X-Ray Dataset**, which contains:
- 82 paired high/low energy TIF images (16-bit grayscale)
- Pascal VOC format annotations for knife detection
- Both threat and non-threat samples

**Note:** The raw TIF images are not included in the repository due to size. See `HUMS-X-ray-Dataset/README.md` for dataset details.

---

## References

1. Alvarez, R.E. and Macovski, A. (1976). "Energy-selective reconstructions in X-ray computerized tomography." *Physics in Medicine & Biology*, 21(5), 733.

2. [US Patent 4029963A](https://patents.google.com/patent/US4029963A/en) - X-ray spectral decomposition imaging system

3. Analytical AI Technology Overview: https://www.analyticalai.com/technology

---

## Author

**Michael Moran**  
AI Engineer (Candidate)  
Georgia Institute of Technology

Supervised by: Dr. Thomas Anthony, CTO, Analytical AI

---

## License

This project is for academic and demonstration purposes. Contact the author for licensing inquiries.
