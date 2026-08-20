"""
Dual-Energy X-Ray Dataset for HUMS Scanner Images

This module provides PyTorch Dataset classes for loading and preprocessing
dual-energy X-ray image pairs from the HUMS dataset.

DATA FORMAT:
------------
The HUMS dataset contains paired TIF images:
    - *_hi.tif: High energy acquisition
    - *_lo.tif: Low energy acquisition

Both are 16-bit grayscale TIF files representing raw photon counts.

IMPORTANT PHYSICS NOTE:
-----------------------
The raw TIF values are INTENSITY (photon counts), NOT log-attenuation.
The preprocessing pipeline must convert:

    Raw Intensity I → Log-Attenuation L = -ln(I/I₀)

before feeding to the PhysicsHead.

Author: Michael Moran
"""

import os
import re
import warnings
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Try to import tifffile for better 16-bit TIF support
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("Warning: tifffile not installed. Using PIL for TIF loading.")
    print("For best 16-bit support, install: pip install tifffile")


def load_tif_image(path: str) -> np.ndarray:
    """
    Load a TIF image as a numpy array.
    
    Uses tifffile if available (better 16-bit support), otherwise PIL.
    
    Args:
        path: Path to TIF file
        
    Returns:
        Image as numpy array (H, W), dtype depends on source
    """
    if HAS_TIFFFILE:
        img = tifffile.imread(path)
    else:
        img = np.array(Image.open(path))

    if img.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale TIF image at {path!r}, but got an array "
            f"with shape {img.shape} (ndim={img.ndim}). Multipage or RGB TIFs "
            f"are not supported by this dataset loader."
        )

    return img


def find_dual_energy_pairs(directory: str) -> List[Tuple[str, str]]:
    """
    Find all matching high/low energy image pairs in a directory.
    
    Expects naming convention: *_hi.tif and *_lo.tif
    
    Args:
        directory: Path to directory containing TIF images
        
    Returns:
        List of tuples (path_low, path_high)
    """
    directory = Path(directory)
    
    # Find all high energy files
    hi_files = list(directory.glob("*_hi.tif"))
    
    pairs = []
    matched_lo_names = set()
    for hi_path in hi_files:
        # Construct matching low energy filename
        lo_path = hi_path.parent / hi_path.name.replace("_hi.tif", "_lo.tif")
        
        if lo_path.exists():
            pairs.append((str(lo_path), str(hi_path)))
            matched_lo_names.add(lo_path.name)
        else:
            print(f"Warning: No matching low energy file for {hi_path}")
    
    # Find low energy files with no matching high energy mate
    lo_files = list(directory.glob("*_lo.tif"))
    orphan_lo_files = [lo_path for lo_path in lo_files if lo_path.name not in matched_lo_names]
    for orphan_path in orphan_lo_files:
        warnings.warn(f"Orphan low energy file with no matching high energy file: {orphan_path}")
    
    # Sort for reproducibility
    pairs.sort(key=lambda x: x[0])
    
    return pairs


class DualEnergyDataset(Dataset):
    """
    PyTorch Dataset for dual-energy X-ray image pairs.
    
    Loads paired low/high energy TIF images and optionally applies
    preprocessing (raw intensity → log-attenuation).
    
    Output format: (B, 2, H, W) tensor where:
        - Channel 0: Low energy (L_low or I_low depending on preprocess)
        - Channel 1: High energy (L_high or I_high)
    """
    
    def __init__(
        self,
        data_dir: str,
        preprocess: bool = True,
        i0_method: str = "per_image_max",
        epsilon: float = 1e-6,
        transform: Optional[Callable] = None,
        normalize_range: bool = False,
        target_size: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing *_hi.tif and *_lo.tif files
            preprocess: If True, convert raw intensity to log-attenuation.
                       If False, return raw 16-bit intensity values.
            i0_method: Method for estimating I₀ (see RawToLogAttenuation)
            epsilon: Numerical stability constant
            transform: Optional transform to apply to output tensor
            normalize_range: If True, normalize output to [0, 1] range
            target_size: Optional (H, W) to resize images
        """
        self.data_dir = Path(data_dir)
        self.preprocess = preprocess
        self.i0_method = i0_method
        self.epsilon = epsilon
        self.transform = transform
        self.normalize_range = normalize_range
        self.target_size = target_size
        
        # Find all image pairs
        self.pairs = find_dual_energy_pairs(data_dir)
        
        if len(self.pairs) == 0:
            raise ValueError(f"No dual-energy pairs found in {data_dir}")
        
        print(f"Found {len(self.pairs)} dual-energy image pairs")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def _load_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single image pair as numpy arrays."""
        lo_path, hi_path = self.pairs[idx]
        
        img_lo = load_tif_image(lo_path).astype(np.float32)
        img_hi = load_tif_image(hi_path).astype(np.float32)
        
        if img_lo.shape != img_hi.shape:
            raise ValueError(
                f"Mismatched shapes for dual-energy pair: {lo_path!r} has shape "
                f"{img_lo.shape} but {hi_path!r} has shape {img_hi.shape}. "
                f"Both images in a pair must have identical dimensions."
            )
        
        return img_lo, img_hi
    
    def _to_log_attenuation(
        self,
        img_lo: np.ndarray,
        img_hi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert raw intensity images to log-attenuation.
        
        L = -ln(I / I₀)
        
        Where I₀ is estimated from the image itself.
        """
        # Estimate I₀
        if self.i0_method == "per_image_max":
            i0_lo = img_lo.max()
            i0_hi = img_hi.max()
        elif self.i0_method == "global":
            # Use theoretical 16-bit max
            i0_lo = i0_hi = 65535.0
        else:
            raise ValueError(
                f"Unknown i0_method: {self.i0_method!r}. Expected one of: "
                f"'per_image_max', 'global'."
            )
        
        # Compute transmission
        T_lo = img_lo / (i0_lo + self.epsilon)
        T_hi = img_hi / (i0_hi + self.epsilon)
        
        # Clamp to valid range
        T_lo = np.clip(T_lo, self.epsilon, 1.0)
        T_hi = np.clip(T_hi, self.epsilon, 1.0)
        
        # Compute log-attenuation
        L_lo = -np.log(T_lo)
        L_hi = -np.log(T_hi)
        
        return L_lo, L_hi
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dictionary with keys:
                - 'image': Tensor (2, H, W) - stacked low/high energy
                - 'path_lo': Path to low energy image
                - 'path_hi': Path to high energy image
                - 'index': Dataset index
        """
        lo_path, hi_path = self.pairs[idx]
        
        # Load images
        img_lo, img_hi = self._load_pair(idx)
        
        # Optional preprocessing
        if self.preprocess:
            img_lo, img_hi = self._to_log_attenuation(img_lo, img_hi)
        
        # Optional resize
        if self.target_size is not None:
            from PIL import Image as PILImage
            img_lo = np.array(PILImage.fromarray(img_lo).resize(
                self.target_size[::-1], PILImage.BILINEAR
            ))
            img_hi = np.array(PILImage.fromarray(img_hi).resize(
                self.target_size[::-1], PILImage.BILINEAR
            ))
        
        # Stack into (2, H, W) tensor
        image = np.stack([img_lo, img_hi], axis=0)
        image = torch.from_numpy(image).float()
        
        # Optional normalization
        if self.normalize_range:
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)
            else:
                # Constant tensor: min/max normalization is undefined (0/0).
                # Map to the midpoint of [0, 1] rather than leaving the
                # original (unnormalized, physically-unitted) value in place.
                image = torch.full_like(image, 0.5)
        
        # Optional transform
        if self.transform is not None:
            image = self.transform(image)
        
        return {
            'image': image,
            'path_lo': lo_path,
            'path_hi': hi_path,
            'index': idx
        }
    
    def get_raw_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get raw numpy arrays without any preprocessing.
        Useful for data exploration.
        """
        return self._load_pair(idx)
    
    def get_statistics(self) -> Dict:
        """
        Compute dataset statistics (for debugging/exploration).
        
        Warning: This loads all images, can be slow for large datasets.
        """
        all_lo_min, all_lo_max = [], []
        all_hi_min, all_hi_max = [], []
        shapes = []
        
        for idx in range(len(self)):
            img_lo, img_hi = self._load_pair(idx)
            
            all_lo_min.append(img_lo.min())
            all_lo_max.append(img_lo.max())
            all_hi_min.append(img_hi.min())
            all_hi_max.append(img_hi.max())
            shapes.append(img_lo.shape)
        
        return {
            'num_pairs': len(self),
            'lo_intensity_range': (min(all_lo_min), max(all_lo_max)),
            'hi_intensity_range': (min(all_hi_min), max(all_hi_max)),
            'shapes': list(set(shapes)),
            'unique_shapes': len(set(shapes))
        }


class SyntheticDualEnergyDataset(Dataset):
    """
    Synthetic dataset for unit testing the physics pipeline.
    
    Generates fake dual-energy images with known material properties
    for validating the PhysicsHead decomposition.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        image_size: Tuple[int, int] = (256, 256),
        seed: int = 42
    ):
        """
        Args:
            num_samples: Number of synthetic images to generate
            image_size: (H, W) size of generated images
            seed: Random seed for reproducibility
        """
        H, W = image_size
        if H <= 50 or W <= 50:
            raise ValueError(
                f"image_size={image_size!r} is too small: both dimensions must be "
                f"greater than 50px because synthetic objects are placed with a "
                f"hardcoded up-to-50px offset/extent."
            )
        
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Define some "materials" with (mu_low, mu_high) attenuation coefficients
        # These are arbitrary but reflect the physics:
        # - Low-Z materials: mu_low ≈ mu_high (Compton dominated)
        # - High-Z materials: mu_low >> mu_high (Photoelectric at low E)
        self.materials = {
            'air': (0.01, 0.01),      # Almost transparent
            'plastic': (0.5, 0.45),    # Low-Z organic
            'water': (0.7, 0.6),       # Reference material
            'aluminum': (1.5, 0.8),    # Medium-Z metal
            'steel': (3.0, 1.2),       # High-Z metal
            'lead': (8.0, 2.0),        # Very high-Z
        }
    
    def __len__(self) -> int:
        return self.num_samples
    
    def _generate_sample(self) -> Tuple[np.ndarray, np.ndarray, str]:
        """Generate a single synthetic dual-energy image pair."""
        H, W = self.image_size
        
        # Start with air background
        L_lo = np.ones((H, W), dtype=np.float32) * self.materials['air'][0]
        L_hi = np.ones((H, W), dtype=np.float32) * self.materials['air'][1]
        
        # Add random objects
        num_objects = self.rng.randint(2, 6)
        material_names = list(self.materials.keys())
        
        for _ in range(num_objects):
            # Random material (excluding air)
            mat_name = self.rng.choice(material_names[1:])
            mu_lo, mu_hi = self.materials[mat_name]
            
            # Random thickness (simulates line integral)
            thickness = self.rng.uniform(0.5, 3.0)
            
            # Random rectangular region
            x1 = self.rng.randint(0, W - 50)
            y1 = self.rng.randint(0, H - 50)
            x2 = min(x1 + self.rng.randint(30, 100), W)
            y2 = min(y1 + self.rng.randint(30, 100), H)
            
            # Add attenuation (objects can overlap)
            L_lo[y1:y2, x1:x2] += mu_lo * thickness
            L_hi[y1:y2, x1:x2] += mu_hi * thickness
        
        return L_lo, L_hi, "synthetic"
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a synthetic sample (already in log-attenuation space)."""
        # Derive a per-index RNG from this instance's seed to ensure
        # reproducibility while still respecting the constructor's `seed` arg.
        self.rng = np.random.RandomState(self.seed + idx)
        
        L_lo, L_hi, label = self._generate_sample()
        
        # Stack into tensor
        image = np.stack([L_lo, L_hi], axis=0)
        image = torch.from_numpy(image).float()
        
        return {
            'image': image,
            'label': label,
            'index': idx
        }


# ==============================================================================
# UNIT TEST
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Dataset Module Unit Test")
    print("=" * 60)
    
    # Test 1: Synthetic Dataset
    print("\n[Test 1] Synthetic Dataset")
    syn_dataset = SyntheticDualEnergyDataset(num_samples=10, image_size=(64, 64))
    sample = syn_dataset[0]
    print(f"  Synthetic sample shape: {sample['image'].shape}")
    print(f"  L_low range: [{sample['image'][0].min():.2f}, {sample['image'][0].max():.2f}]")
    print(f"  L_high range: [{sample['image'][1].min():.2f}, {sample['image'][1].max():.2f}]")
    print("  ✓ Synthetic dataset works")
    
    # Test 2: Real Dataset (if available)
    print("\n[Test 2] HUMS Dataset")
    hums_path = Path(__file__).parent.parent / "HUMS-X-ray-Dataset" / "HighLow"
    
    if hums_path.exists():
        try:
            dataset = DualEnergyDataset(
                str(hums_path),
                preprocess=True,
                i0_method="per_image_max"
            )
            
            sample = dataset[0]
            print(f"  Loaded {len(dataset)} image pairs")
            print(f"  Sample shape: {sample['image'].shape}")
            print(f"  L_low range: [{sample['image'][0].min():.4f}, {sample['image'][0].max():.4f}]")
            print(f"  L_high range: [{sample['image'][1].min():.4f}, {sample['image'][1].max():.4f}]")
            
            # Check for NaN/Inf
            if torch.isnan(sample['image']).any():
                print("  ✗ WARNING: NaN in output!")
            elif torch.isinf(sample['image']).any():
                print("  ✗ WARNING: Inf in output!")
            else:
                print("  ✓ Real dataset loads successfully")
                
        except Exception as e:
            print(f"  Error loading HUMS dataset: {e}")
            raise
    else:
        print(f"  HUMS dataset not found at {hums_path}")
        print("  Skipping real data test")
    
    print("\n✓ Dataset tests completed!")
