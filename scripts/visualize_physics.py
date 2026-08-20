#!/usr/bin/env python3
"""
Physics Pipeline Visualization Script

This script validates the entire Aethelgard preprocessing + PhysicsHead pipeline
by loading real HUMS dual-energy X-ray images and visualizing the physics maps.

WHAT TO LOOK FOR:
-----------------
1. L_low and L_high should be ≥ 0 everywhere (no negative attenuation)
2. Z_eff should show contrast: metals (bright) vs organics (dark)
3. Ratio gradient (∇R) should highlight material BOUNDARIES, not geometric edges
4. No NaN or Inf values anywhere

USAGE:
------
    python scripts/visualize_physics.py

OUTPUT:
-------
    Saves figures to outputs/ directory

Author: Michael Moran
Supervisor: Dr. Thomas Anthony, CTO, Analytical AI
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

# Import our modules
from aethelgard.dataset import DualEnergyDataset, load_tif_image
from aethelgard.physics_head import PhysicsHead
from aethelgard.preprocessing import RawToLogAttenuation


def setup_output_dir() -> Path:
    """Create output directory if it doesn't exist."""
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def visualize_raw_images(
    dataset: DualEnergyDataset,
    idx: int = 0,
    output_dir: Path = None
):
    """
    Visualize raw intensity images before preprocessing.
    
    This helps us understand the input data format.
    """
    print(f"\n[1] Visualizing raw images (sample {idx})...")
    
    # Get raw images (no preprocessing)
    img_lo, img_hi = dataset.get_raw_pair(idx)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Raw Intensity Images (Sample {idx})', fontsize=14, fontweight='bold')
    
    # Row 1: Low Energy
    ax = axes[0, 0]
    im = ax.imshow(img_lo, cmap='gray')
    ax.set_title(f'Low Energy (Raw)\nShape: {img_lo.shape}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[0, 1]
    ax.hist(img_lo.ravel(), bins=100, color='blue', alpha=0.7)
    ax.set_title(f'Low Energy Histogram\nRange: [{img_lo.min():.0f}, {img_lo.max():.0f}]')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    
    # Row 2: High Energy
    ax = axes[1, 0]
    im = ax.imshow(img_hi, cmap='gray')
    ax.set_title(f'High Energy (Raw)\nShape: {img_hi.shape}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[1, 1]
    ax.hist(img_hi.ravel(), bins=100, color='red', alpha=0.7)
    ax.set_title(f'High Energy Histogram\nRange: [{img_hi.min():.0f}, {img_hi.max():.0f}]')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    
    # Difference and ratio
    ax = axes[0, 2]
    diff = img_lo.astype(np.float32) - img_hi.astype(np.float32)
    im = ax.imshow(diff, cmap='RdBu', vmin=-5000, vmax=5000)
    ax.set_title('Intensity Difference\n(Low - High)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[1, 2]
    ratio = img_lo.astype(np.float32) / (img_hi.astype(np.float32) + 1)
    im = ax.imshow(ratio, cmap='viridis', vmin=0.5, vmax=2.0)
    ax.set_title('Intensity Ratio\n(Low / High)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    
    if output_dir:
        save_path = output_dir / f"01_raw_images_sample{idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()
    
    return img_lo, img_hi


def visualize_log_attenuation(
    dataset: DualEnergyDataset,
    idx: int = 0,
    output_dir: Path = None
):
    """
    Visualize log-attenuation images after preprocessing.

    Runs the REAL preprocessing module (RawToLogAttenuation) rather than
    the dataset's separate NumPy conversion path, so this validates the
    actual pipeline (including Gaussian denoising) that feeds PhysicsHead
    in production.
    """
    print(f"\n[2] Visualizing log-attenuation (sample {idx})...")

    # Get raw (unprocessed) images and run them through the real
    # preprocessing module, matching how the production pipeline (and
    # demo_debug.py) constructs log-attenuation from raw intensity.
    img_lo, img_hi = dataset.get_raw_pair(idx)
    raw_tensor = torch.from_numpy(
        np.stack([img_lo, img_hi], axis=0)
    ).unsqueeze(0).float()  # (1, 2, H, W)

    preprocessor = RawToLogAttenuation(
        epsilon=dataset.epsilon,
        gaussian_sigma=1.0,
        i0_method=dataset.i0_method,
    )
    with torch.no_grad():
        L_batch = preprocessor(raw_tensor)  # (1, 2, H, W)
    L_tensor = L_batch.squeeze(0)  # (2, H, W)

    L_low = L_tensor[0].numpy()
    L_high = L_tensor[1].numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Log-Attenuation Maps (Sample {idx})', fontsize=14, fontweight='bold')
    
    # Row 1: Log-attenuation maps
    ax = axes[0, 0]
    im = ax.imshow(L_low, cmap='magma')
    ax.set_title(f'L_low (Log-Attenuation)\nRange: [{L_low.min():.3f}, {L_low.max():.3f}]')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[0, 1]
    im = ax.imshow(L_high, cmap='magma')
    ax.set_title(f'L_high (Log-Attenuation)\nRange: [{L_high.min():.3f}, {L_high.max():.3f}]')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Ratio map (thickness-invariant!)
    ax = axes[0, 2]
    R = L_low / (L_high + 1e-6)
    im = ax.imshow(R, cmap='coolwarm', vmin=0.5, vmax=2.0)
    ax.set_title(f'Ratio R = L_low/L_high\n(Thickness Invariant!)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 2: Histograms and analysis
    ax = axes[1, 0]
    ax.hist(L_low.ravel(), bins=100, color='orange', alpha=0.7, label='L_low')
    ax.hist(L_high.ravel(), bins=100, color='purple', alpha=0.7, label='L_high')
    ax.set_title('Log-Attenuation Histograms')
    ax.set_xlabel('Log-Attenuation')
    ax.set_ylabel('Count')
    ax.legend()
    ax.set_yscale('log')
    
    ax = axes[1, 1]
    ax.hist(R.ravel(), bins=100, color='green', alpha=0.7)
    ax.set_title(f'Ratio Histogram\nMean: {R.mean():.3f}, Std: {R.std():.3f}')
    ax.set_xlabel('R = L_low / L_high')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    
    # Physics sanity check
    ax = axes[1, 2]
    ax.text(0.1, 0.9, 'Physics Sanity Check:', fontsize=12, fontweight='bold',
            transform=ax.transAxes)
    
    checks = []
    # Check 1: Non-negative attenuation
    if L_low.min() >= 0 and L_high.min() >= 0:
        checks.append('✓ Non-negative attenuation')
    else:
        checks.append('✗ NEGATIVE attenuation detected!')
    
    # Check 2: No NaN
    if not (np.isnan(L_low).any() or np.isnan(L_high).any()):
        checks.append('✓ No NaN values')
    else:
        checks.append('✗ NaN values detected!')
    
    # Check 3: No Inf
    if not (np.isinf(L_low).any() or np.isinf(L_high).any()):
        checks.append('✓ No Inf values')
    else:
        checks.append('✗ Inf values detected!')
    
    # Check 4: Reasonable range (typical log-atten is 0-10)
    max_L = max(L_low.max(), L_high.max())
    if max_L < 15:
        checks.append(f'✓ Reasonable range (max={max_L:.2f})')
    else:
        checks.append(f'⚠ Large attenuation (max={max_L:.2f})')
    
    for i, check in enumerate(checks):
        color = 'green' if '✓' in check else 'red' if '✗' in check else 'orange'
        ax.text(0.1, 0.7 - i*0.15, check, fontsize=11, color=color,
                transform=ax.transAxes)
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_dir:
        save_path = output_dir / f"02_log_attenuation_sample{idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()
    
    return L_tensor


def visualize_physics_maps(
    L_tensor: torch.Tensor,
    idx: int = 0,
    output_dir: Path = None
):
    """
    Visualize PhysicsHead output maps.
    
    This is the key validation: do the physics maps make sense?
    """
    print(f"\n[3] Visualizing PhysicsHead output (sample {idx})...")
    
    # Initialize PhysicsHead
    physics_head = PhysicsHead(order=2, learnable=False)
    
    # Run forward pass
    with torch.no_grad():
        physics_maps = physics_head(L_tensor.unsqueeze(0))  # Add batch dim
    
    # Extract individual maps
    A1 = physics_maps[0, 0].numpy()      # Photoelectric
    A2 = physics_maps[0, 1].numpy()      # Compton
    Z_eff = physics_maps[0, 2].numpy()   # Effective atomic number
    grad_R = physics_maps[0, 3].numpy()  # Ratio gradient
    
    # Also get L_low and L_high for reference
    L_low = L_tensor[0].numpy()
    L_high = L_tensor[1].numpy()
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'PhysicsHead Output Maps (Sample {idx})', fontsize=14, fontweight='bold')
    
    # Row 1: Input and basis coefficients
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(L_low, cmap='magma')
    ax.set_title('Input: L_low')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(L_high, cmap='magma')
    ax.set_title('Input: L_high')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(A1, cmap='hot')
    ax.set_title(f'A₁ (Photoelectric)\nRange: [{A1.min():.2f}, {A1.max():.2f}]')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = fig.add_subplot(gs[0, 3])
    im = ax.imshow(A2, cmap='hot')
    ax.set_title(f'A₂ (Compton)\nRange: [{A2.min():.2f}, {A2.max():.2f}]')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 2: Derived maps
    ax = fig.add_subplot(gs[1, 0])
    # Z_eff with custom colormap - lower values (organics) darker, higher (metals) brighter
    im = ax.imshow(Z_eff, cmap='viridis', vmin=np.percentile(Z_eff, 5), 
                   vmax=np.percentile(Z_eff, 95))
    ax.set_title(f'Z_eff (Atomic Number)\nRange: [{Z_eff.min():.1f}, {Z_eff.max():.1f}]')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(grad_R, cmap='hot')
    ax.set_title(f'∇R (Material Edge Detector)\nRange: [{grad_R.min():.3f}, {grad_R.max():.3f}]')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Overlay: Z_eff with edges
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(Z_eff, cmap='viridis', vmin=np.percentile(Z_eff, 5), 
              vmax=np.percentile(Z_eff, 95), alpha=0.7)
    # Overlay edges where gradient is high
    edge_threshold = np.percentile(grad_R, 90)
    edge_mask = grad_R > edge_threshold
    ax.imshow(np.ma.masked_where(~edge_mask, grad_R), cmap='Reds', alpha=0.8)
    ax.set_title('Z_eff + Material Edges\n(Red = High ∇R)')
    ax.axis('off')
    
    # Ratio map for reference
    ax = fig.add_subplot(gs[1, 3])
    R = L_low / (L_high + 1e-6)
    im = ax.imshow(R, cmap='coolwarm', vmin=0.5, vmax=2.0)
    ax.set_title('Ratio R = L_low/L_high\n(Input to ∇R)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 3: Analysis
    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(A2.ravel()[::100], A1.ravel()[::100], alpha=0.3, s=1)
    ax.set_xlabel('A₂ (Compton)')
    ax.set_ylabel('A₁ (Photoelectric)')
    ax.set_title('A-Space Decomposition\n(Each point = one pixel)')
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[2, 1])
    ax.hist(Z_eff.ravel(), bins=100, color='green', alpha=0.7)
    ax.set_xlabel('Z_eff')
    ax.set_ylabel('Count')
    ax.set_title('Z_eff Distribution')
    ax.set_yscale('log')
    
    ax = fig.add_subplot(gs[2, 2])
    ax.hist(grad_R.ravel(), bins=100, color='red', alpha=0.7)
    ax.set_xlabel('∇R')
    ax.set_ylabel('Count')
    ax.set_title('Gradient Distribution\n(Most should be near 0)')
    ax.set_yscale('log')
    
    # Coefficient summary
    ax = fig.add_subplot(gs[2, 3])
    summary = physics_head.get_coefficient_summary()
    text = "Polynomial Coefficients:\n\n"
    text += "A₁ (Photoelectric):\n"
    for name, val in zip(summary['term_names'], summary['coeffs_a1']):
        text += f"  {name}: {val:.3f}\n"
    text += f"\nZ scale: {summary['z_scale']:.1f}"
    ax.text(0.1, 0.95, text, fontsize=9, family='monospace',
            transform=ax.transAxes, verticalalignment='top')
    ax.axis('off')
    ax.set_title('PhysicsHead Parameters')
    
    plt.tight_layout()
    
    if output_dir:
        save_path = output_dir / f"03_physics_maps_sample{idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()
    
    return physics_maps


def visualize_comparison(
    dataset: DualEnergyDataset,
    physics_head: PhysicsHead,
    indices: list = [0, 1, 2],
    output_dir: Path = None
):
    """
    Compare physics maps across multiple samples.
    """
    print(f"\n[4] Comparing physics maps across samples {indices}...")
    
    n = len(indices)
    fig, axes = plt.subplots(n, 5, figsize=(20, 4*n))
    fig.suptitle('Physics Maps Comparison Across Samples', fontsize=14, fontweight='bold')
    
    for row, idx in enumerate(indices):
        sample = dataset[idx]
        L_tensor = sample['image']
        
        with torch.no_grad():
            physics_maps = physics_head(L_tensor.unsqueeze(0))
        
        # Extract maps
        L_low = L_tensor[0].numpy()
        A1 = physics_maps[0, 0].numpy()
        A2 = physics_maps[0, 1].numpy()
        Z_eff = physics_maps[0, 2].numpy()
        grad_R = physics_maps[0, 3].numpy()
        
        # Plot
        ax = axes[row, 0] if n > 1 else axes[0]
        ax.imshow(L_low, cmap='gray')
        ax.set_title(f'Sample {idx}\nL_low' if row == 0 else f'Sample {idx}')
        ax.axis('off')
        
        ax = axes[row, 1] if n > 1 else axes[1]
        ax.imshow(A1, cmap='hot')
        ax.set_title('A₁ (Photoelectric)' if row == 0 else '')
        ax.axis('off')
        
        ax = axes[row, 2] if n > 1 else axes[2]
        ax.imshow(A2, cmap='hot')
        ax.set_title('A₂ (Compton)' if row == 0 else '')
        ax.axis('off')
        
        ax = axes[row, 3] if n > 1 else axes[3]
        ax.imshow(Z_eff, cmap='viridis', vmin=np.percentile(Z_eff, 5),
                  vmax=np.percentile(Z_eff, 95))
        ax.set_title('Z_eff' if row == 0 else '')
        ax.axis('off')
        
        ax = axes[row, 4] if n > 1 else axes[4]
        ax.imshow(grad_R, cmap='hot')
        ax.set_title('∇R (Edges)' if row == 0 else '')
        ax.axis('off')
    
    plt.tight_layout()
    
    if output_dir:
        save_path = output_dir / "04_physics_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def run_validation():
    """
    Run the complete validation pipeline.
    """
    print("=" * 70)
    print("AETHELGARD PHYSICS PIPELINE VALIDATION")
    print("=" * 70)
    
    # Setup
    output_dir = setup_output_dir()
    print(f"\nOutput directory: {output_dir}")
    
    # Find HUMS dataset
    data_dir = Path(__file__).parent.parent / "HUMS-X-ray-Dataset" / "HighLow"
    
    if not data_dir.exists():
        print(f"\n✗ ERROR: HUMS dataset not found at {data_dir}")
        print("  Please ensure the dataset is in the correct location.")
        raise FileNotFoundError(f"HUMS dataset not found at {data_dir}")
    
    # Load dataset
    print(f"\nLoading dataset from: {data_dir}")
    try:
        dataset = DualEnergyDataset(
            str(data_dir),
            preprocess=True,
            i0_method="per_image_max"
        )
    except Exception as e:
        print(f"✗ ERROR loading dataset: {e}")
        raise
    
    # Print dataset stats
    print(f"\nDataset Statistics:")
    print(f"  Number of pairs: {len(dataset)}")
    
    # Run visualizations for first sample
    sample_idx = 0
    
    # 1. Raw images
    visualize_raw_images(dataset, idx=sample_idx, output_dir=output_dir)
    
    # 2. Log-attenuation
    L_tensor = visualize_log_attenuation(dataset, idx=sample_idx, output_dir=output_dir)
    
    # 3. Physics maps
    visualize_physics_maps(L_tensor, idx=sample_idx, output_dir=output_dir)
    
    # 4. Compare multiple samples
    physics_head = PhysicsHead(order=2, learnable=False)
    n_samples = min(3, len(dataset))
    visualize_comparison(
        dataset, physics_head, 
        indices=list(range(n_samples)),
        output_dir=output_dir
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"\nGenerated visualizations saved to: {output_dir}")
    print("\nReview the images to verify:")
    print("  1. Log-attenuation maps have no NaN/Inf values")
    print("  2. Z_eff shows contrast between different materials")
    print("  3. ∇R highlights material boundaries, not geometric edges")
    print("  4. A-space plot shows physically meaningful clustering")
    print("\nIf everything looks good, the PhysicsHead is working correctly!")


if __name__ == "__main__":
    run_validation()
