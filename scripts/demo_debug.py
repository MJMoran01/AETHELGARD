#!/usr/bin/env python3
"""
Demo Debug Script: Visual Pipeline Verification

This script takes a single raw dual-energy X-ray image pair and generates
a comprehensive visual report showing what happens at each pipeline stage.

OUTPUT FILES (saved to outputs/debug/):
---------------------------------------
1. 01_raw_input.png         - Raw intensity images (Hi/Lo energy)
2. 02_preprocessing.png     - Log-attenuation + Ghost Map (denoiser diff)
3. 03_bhc_comparison.png    - BHC vs No-BHC Z_eff comparison
4. 04_aspace_scatter.png    - A-space scatter with feasibility bounds
5. 05_zeff_histogram.png    - Z_eff distribution with material markers
6. 06_pseudocolor.png       - Final security colormap output
7. 07_sanity_check.png      - Physics validation summary

CRITICAL NOTE:
--------------
The "No-BHC" calculation in this script is FOR DEMO VISUALIZATION ONLY.
It exists solely to show WHY the polynomial correction matters.
The main pipeline should ALWAYS use BHC. This naive calculation must
NEVER be fed to the VisionHead.

Author: Michael Moran
Supervisor: Dr. Thomas Anthony, CTO, Analytical AI
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# Import our modules
from aethelgard.preprocessing import RawToLogAttenuation
from aethelgard.physics_head import PhysicsHead, ASpaceNoiseFilter


# ==============================================================================
# CONFIGURATION
# ==============================================================================
EPSILON = 1e-6
Z_SCALE = 10.0  # Match PhysicsHead default
PLOT_SAMPLING_SEED = 42  # Fixed seed for scatter-plot subsampling, so committed
                          # debug artifacts are reproducible across re-runs.

# Material Z values for histogram markers
Z_MARKERS = {
    'Hydrogen (H)': 1,
    'Carbon (C)': 6,
    'Nitrogen (N)': 7,
    'Oxygen (O)': 8,
    'Aluminum (Al)': 13,
    'Iron (Fe)': 26,
    'Copper (Cu)': 29,
}

# Security colormap boundaries (Z_eff values)
# These are approximate and depend on calibration
ORGANIC_MAX = 10      # Orange/Brown: Z < 10
INORGANIC_MAX = 18    # Green: 10 < Z < 18
                      # Blue: Z > 18 (Metals)



# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def setup_output_dir() -> Path:
    """Create output directory for debug visualizations."""
    output_dir = Path(__file__).parent.parent / "outputs" / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_image_pair(lo_path: str, hi_path: str) -> tuple:
    """
    Load a dual-energy image pair from TIF files.
    
    Args:
        lo_path: Path to low-energy TIF
        hi_path: Path to high-energy TIF
        
    Returns:
        (img_lo, img_hi): NumPy arrays (H, W), dtype=uint16
    """
    img_lo = np.array(Image.open(lo_path))
    img_hi = np.array(Image.open(hi_path))
    
    print(f"  Loaded: {Path(lo_path).name}")
    print(f"    Shape: {img_lo.shape}, dtype: {img_lo.dtype}")
    print(f"    Range: [{img_lo.min()}, {img_lo.max()}]")
    print(f"  Loaded: {Path(hi_path).name}")
    print(f"    Shape: {img_hi.shape}, dtype: {img_hi.dtype}")
    print(f"    Range: [{img_hi.min()}, {img_hi.max()}]")
    
    return img_lo, img_hi


def create_security_colormap():
    """
    Create TSA-style security colormap.
    
    Returns a colormap that maps Z_eff to:
        - Orange/Brown: Organics (Z < 10)
        - Green: Inorganics (10 < Z < 18)
        - Blue: Metals (Z > 18)
    """
    # Define colors at key points
    colors = [
        (0.0, '#8B4513'),    # SaddleBrown at Z=0
        (0.3, '#FFA500'),    # Orange at ~Z=9
        (0.4, '#228B22'),    # ForestGreen at ~Z=12
        (0.55, '#32CD32'),   # LimeGreen at ~Z=16
        (0.6, '#1E90FF'),    # DodgerBlue at ~Z=18
        (1.0, '#00008B'),    # DarkBlue at Z=30+
    ]
    
    positions = [c[0] for c in colors]
    hex_colors = [c[1] for c in colors]
    rgb_colors = [mcolors.hex2color(h) for h in hex_colors]
    
    cmap = LinearSegmentedColormap.from_list('security', list(zip(positions, rgb_colors)))
    return cmap


def compute_naive_zeff(L_low: np.ndarray, L_high: np.ndarray) -> np.ndarray:
    """
    Compute Z_eff WITHOUT BHC polynomial correction.
    
    THIS IS FOR DEMO VISUALIZATION ONLY!
    =====================================
    This naive calculation assumes log-attenuation is already linear,
    which is FALSE due to beam hardening. The result will show:
    - Incorrect Z values
    - Thickness-dependent errors (the "cupping" artifact)
    
    The purpose is to visually demonstrate WHY BHC matters.
    DO NOT use this output for anything other than comparison plots.
    
    THE PHYSICS (Naive Approximation):
    ----------------------------------
    Without BHC, we directly compute:
        A1_naive = L_low - L_high  (isolate photoelectric contribution)
        A2_naive = L_high          (Compton dominated at high energy)
        Z_naive = (A1_naive / A2_naive)^(1/3)
    
    This is wrong because L is not linearly related to material properties
    due to the polychromatic X-ray spectrum.
    """
    # Naive basis decomposition (no polynomial correction)
    A1_naive = np.maximum(L_low - L_high, EPSILON)  # Clamp to prevent negative
    A2_naive = np.maximum(L_high, EPSILON)
    
    # Naive Z_eff calculation
    ratio = A1_naive / (A2_naive + EPSILON)
    Z_naive = Z_SCALE * np.power(ratio + EPSILON, 1.0 / 3.0)
    
    return Z_naive



# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_01_raw_input(img_lo: np.ndarray, img_hi: np.ndarray, output_dir: Path):
    """Panel 1: Raw intensity images."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('01. Raw Input Images (Intensity Domain)', fontsize=14, fontweight='bold')
    
    # Low energy
    ax = axes[0]
    im = ax.imshow(img_lo, cmap='gray')
    ax.set_title(f'Low Energy\nRange: [{img_lo.min()}, {img_lo.max()}]')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, label='Intensity')
    
    # High energy
    ax = axes[1]
    im = ax.imshow(img_hi, cmap='gray')
    ax.set_title(f'High Energy\nRange: [{img_hi.min()}, {img_hi.max()}]')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, label='Intensity')
    
    plt.tight_layout()
    save_path = output_dir / '01_raw_input.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_02_preprocessing(img_lo: np.ndarray, img_hi: np.ndarray, 
                          L_low: np.ndarray, L_high: np.ndarray,
                          preprocess_debug: dict, output_dir: Path):
    """Panel 2: Log-attenuation + Ghost Map (denoiser difference)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('02. Preprocessing: Intensity → Log-Attenuation', fontsize=14, fontweight='bold')
    
    # Row 1: Log-attenuation maps
    ax = axes[0, 0]
    im = ax.imshow(L_low, cmap='magma')
    ax.set_title(f'L_low (Log-Attenuation)\n[{L_low.min():.3f}, {L_low.max():.3f}]')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[0, 1]
    im = ax.imshow(L_high, cmap='magma')
    ax.set_title(f'L_high (Log-Attenuation)\n[{L_high.min():.3f}, {L_high.max():.3f}]')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Ratio map (thickness-invariant)
    ax = axes[0, 2]
    R = L_low / (L_high + EPSILON)
    im = ax.imshow(R, cmap='coolwarm', vmin=0.5, vmax=2.0)
    ax.set_title('Ratio R = L_low/L_high\n(Thickness-Invariant)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 2: Ghost Map (denoiser difference) - LOW ENERGY CHANNEL
    # Extract the difference between raw and blurred (before log transform)
    raw_input = preprocess_debug['raw_input'][0, 0].numpy()  # Low energy channel
    after_blur = preprocess_debug['after_blur'][0, 0].numpy()
    
    ghost_map = np.abs(raw_input - after_blur)
    
    ax = axes[1, 0]
    im = ax.imshow(ghost_map, cmap='magma')
    ax.set_title('Ghost Map (|Raw - Blurred|)\nShould look like NOISE, not edges')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, label='Intensity Diff')
    
    # Ghost map histogram
    ax = axes[1, 1]
    ax.hist(ghost_map.ravel(), bins=100, color='orange', alpha=0.7)
    ax.set_xlabel('|Raw - Blurred|')
    ax.set_ylabel('Count')
    ax.set_title('Ghost Map Distribution\nShould be right-skewed (mostly small)')
    ax.set_yscale('log')
    
    # Text summary
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = (
        "GHOST MAP INTERPRETATION\n"
        "========================\n\n"
        f"Max difference: {ghost_map.max():.1f}\n"
        f"Mean difference: {ghost_map.mean():.1f}\n"
        f"Std difference: {ghost_map.std():.1f}\n\n"
        "WHAT TO LOOK FOR:\n"
        "• Should show random noise pattern\n"
        "• Should NOT show object edges\n"
        "• If edges visible: blur is too strong\n"
    )
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            family='monospace', verticalalignment='top')
    
    plt.tight_layout()
    save_path = output_dir / '02_preprocessing.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")



def plot_03_bhc_comparison(L_low: np.ndarray, L_high: np.ndarray,
                           Z_eff_bhc: np.ndarray, output_dir: Path):
    """
    Panel 3: BHC vs No-BHC Z_eff comparison.
    
    CRITICAL: The No-BHC calculation is FOR VISUALIZATION ONLY!
    It demonstrates why polynomial BHC matters.
    """
    # Compute naive Z_eff (no BHC)
    Z_naive = compute_naive_zeff(L_low, L_high)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('03. BHC Comparison: Why Polynomial Correction Matters', 
                 fontsize=14, fontweight='bold')
    
    # Row 1: Z_eff images
    vmin = min(np.percentile(Z_naive, 5), np.percentile(Z_eff_bhc, 5))
    vmax = max(np.percentile(Z_naive, 95), np.percentile(Z_eff_bhc, 95))
    
    ax = axes[0, 0]
    im = ax.imshow(Z_naive, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title('Z_eff WITHOUT BHC\n(DEMO ONLY - Never use!)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[0, 1]
    im = ax.imshow(Z_eff_bhc, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title('Z_eff WITH BHC\n(Correct - Use this!)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Difference map
    ax = axes[0, 2]
    diff = Z_eff_bhc - Z_naive
    im = ax.imshow(diff, cmap='RdBu', vmin=-5, vmax=5)
    ax.set_title('Difference (BHC - Naive)\nRed=BHC higher, Blue=lower')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 2: Histograms and explanation
    ax = axes[1, 0]
    ax.hist(Z_naive.ravel(), bins=100, alpha=0.5, label='No BHC', color='red')
    ax.hist(Z_eff_bhc.ravel(), bins=100, alpha=0.5, label='With BHC', color='green')
    ax.set_xlabel('Z_eff')
    ax.set_ylabel('Count')
    ax.set_title('Z_eff Distributions')
    ax.legend()
    ax.set_yscale('log')
    
    # Scatter: Naive vs BHC
    ax = axes[1, 1]
    rng = np.random.default_rng(PLOT_SAMPLING_SEED)
    sample_idx = rng.choice(Z_naive.size, min(10000, Z_naive.size), replace=False)
    ax.scatter(Z_naive.ravel()[sample_idx], Z_eff_bhc.ravel()[sample_idx], 
               alpha=0.1, s=1)
    ax.plot([0, 40], [0, 40], 'r--', label='y=x')
    ax.set_xlabel('Z_eff (No BHC)')
    ax.set_ylabel('Z_eff (With BHC)')
    ax.set_title('Correlation Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Explanation text
    ax = axes[1, 2]
    ax.axis('off')
    explanation = (
        "WHY BHC MATTERS\n"
        "================\n\n"
        "The X-ray beam is POLYCHROMATIC\n"
        "(many energies, not just one).\n\n"
        "As the beam traverses material:\n"
        "• Low energies are absorbed first\n"
        "• Beam becomes 'harder' (higher avg E)\n"
        "• This causes NONLINEAR attenuation\n\n"
        "The NAIVE calculation assumes:\n"
        "  L = μ × t  (linear)\n\n"
        "The TRUTH is:\n"
        "  L = f(μ, t, spectrum)  (nonlinear)\n\n"
        "BHC polynomial CORRECTS this\n"
        "nonlinearity, giving accurate Z.\n\n"
        "WITHOUT BHC:\n"
        "• Thick objects appear higher-Z\n"
        "• 'Cupping' artifacts in CT\n"
        "• Material misclassification\n"
    )
    ax.text(0.05, 0.95, explanation, transform=ax.transAxes, fontsize=9,
            family='monospace', verticalalignment='top')
    
    plt.tight_layout()
    save_path = output_dir / '03_bhc_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")



def plot_04_aspace_scatter(A1: np.ndarray, A2: np.ndarray, 
                           filter_diagnostics: dict, output_dir: Path):
    """Panel 4: A-space scatter with feasibility bounds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('04. A-Space Analysis: Feasibility Wedge', fontsize=14, fontweight='bold')
    
    # Subsample for plotting
    n_points = A1.size
    max_plot = 50000
    if n_points > max_plot:
        rng = np.random.default_rng(PLOT_SAMPLING_SEED)
        idx = rng.choice(n_points, max_plot, replace=False)
        A1_plot = A1.ravel()[idx]
        A2_plot = A2.ravel()[idx]
    else:
        A1_plot = A1.ravel()
        A2_plot = A2.ravel()
    
    # Left: Cartesian A-space
    ax = axes[0]
    ax.scatter(A2_plot, A1_plot, alpha=0.1, s=1, c='blue')
    ax.set_xlabel('A₂ (Compton)')
    ax.set_ylabel('A₁ (Photoelectric)')
    ax.set_title('A-Space (Cartesian)\nEach point = one pixel')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, np.percentile(A2_plot, 99))
    ax.set_ylim(0, np.percentile(A1_plot, 99))
    
    # Draw wedge bounds (θ_min and θ_max lines)
    theta_min = 0.05  # From ASpaceNoiseFilter defaults
    theta_max = 1.50
    r_max = np.percentile(np.sqrt(A1_plot**2 + A2_plot**2), 99)
    
    # θ_min line: A1/A2 = tan(θ_min)
    a2_line = np.linspace(0, r_max, 100)
    a1_min = a2_line * np.tan(theta_min)
    a1_max = a2_line * np.tan(theta_max)
    
    ax.plot(a2_line * np.cos(theta_min), a2_line * np.sin(theta_min), 
            'r--', linewidth=2, label=f'θ_min={theta_min:.2f} rad')
    ax.plot(a2_line * np.cos(theta_max), a2_line * np.sin(theta_max), 
            'g--', linewidth=2, label=f'θ_max={theta_max:.2f} rad')
    ax.legend(loc='upper left')
    
    # Right: Polar histogram
    ax = axes[1]
    theta = np.arctan2(A1_plot, A2_plot + EPSILON)
    ax.hist(theta, bins=100, color='purple', alpha=0.7)
    ax.axvline(theta_min, color='red', linestyle='--', linewidth=2, label='θ_min')
    ax.axvline(theta_max, color='green', linestyle='--', linewidth=2, label='θ_max')
    ax.set_xlabel('θ = atan2(A₁, A₂) [radians]')
    ax.set_ylabel('Count')
    ax.set_title('Polar Angle Distribution\n(θ encodes Z_eff)')
    ax.legend()
    
    # Add filter stats
    if filter_diagnostics:
        stats_text = (
            f"Filter Stats:\n"
            f"  Wedge-clamped: {filter_diagnostics.get('n_wedge_clamped', 'N/A')}\n"
            f"  k-NN outliers: {filter_diagnostics.get('n_outliers', 'N/A')}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = output_dir / '04_aspace_scatter.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")



def plot_05_zeff_histogram(Z_eff: np.ndarray, L_low: np.ndarray, output_dir: Path):
    """Panel 5: Z_eff histogram with material markers."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('05. Z_eff Distribution with Material References', fontsize=14, fontweight='bold')
    
    # Create mask to exclude air/background (very low attenuation)
    air_mask = L_low < 0.1  # Low attenuation = air
    Z_material = Z_eff[~air_mask]
    
    ax.hist(Z_material.ravel(), bins=150, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Z_eff (Effective Atomic Number)')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    
    # Add material reference lines
    for name, z in Z_MARKERS.items():
        ax.axvline(z, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.text(z, ax.get_ylim()[1] * 0.8, name, rotation=90, va='top', ha='right', fontsize=8)
    
    # Add colored regions
    ax.axvspan(0, ORGANIC_MAX, alpha=0.1, color='orange', label='Organics (Z<10)')
    ax.axvspan(ORGANIC_MAX, INORGANIC_MAX, alpha=0.1, color='green', label='Inorganics (10<Z<18)')
    ax.axvspan(INORGANIC_MAX, 35, alpha=0.1, color='blue', label='Metals (Z>18)')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 35)
    
    plt.tight_layout()
    save_path = output_dir / '05_zeff_histogram.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_06_pseudocolor(Z_eff: np.ndarray, output_dir: Path):
    """Panel 6: Security pseudo-color output."""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('06. Pseudo-Color Security Output (The "Money Shot")', fontsize=14, fontweight='bold')
    
    cmap = create_security_colormap()
    im = ax.imshow(Z_eff, cmap=cmap, vmin=0, vmax=30)
    ax.axis('off')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label('Z_eff')
    # The colorbar's own axes are in DATA coordinates (0-30, matching
    # vmin/vmax above), not normalized [0, 1] fractions -- so boundary
    # lines/text must use the raw Z-value scale to line up with the
    # colorbar's own tick labels.
    cbar.ax.axhline(y=ORGANIC_MAX, color='black', linewidth=2)
    cbar.ax.axhline(y=INORGANIC_MAX, color='black', linewidth=2)
    cbar.ax.text(1.5, ORGANIC_MAX / 2, 'Organics', fontsize=8, va='center')
    cbar.ax.text(1.5, (ORGANIC_MAX + INORGANIC_MAX) / 2, 'Inorganics', fontsize=8, va='center')
    cbar.ax.text(1.5, (INORGANIC_MAX + 30) / 2, 'Metals', fontsize=8, va='center')
    
    plt.tight_layout()
    save_path = output_dir / '06_pseudocolor.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_07_sanity_check(L_low, L_high, Z_eff, A1, A2, grad_R, output_dir: Path):
    """Panel 7: Physics sanity check summary."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('07. Physics Sanity Check', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    checks = []
    # Check 1: Non-negative log-attenuation
    if L_low.min() >= 0 and L_high.min() >= 0:
        checks.append(('✓', 'Non-negative log-attenuation', 'green'))
    else:
        checks.append(('✗', f'NEGATIVE log-attenuation! min={min(L_low.min(), L_high.min()):.4f}', 'red'))
    
    # Check 2: No NaN
    has_nan = np.isnan(L_low).any() or np.isnan(L_high).any() or np.isnan(Z_eff).any()
    if not has_nan:
        checks.append(('✓', 'No NaN values', 'green'))
    else:
        checks.append(('✗', 'NaN values detected!', 'red'))
    
    # Check 3: No Inf
    has_inf = np.isinf(L_low).any() or np.isinf(L_high).any() or np.isinf(Z_eff).any()
    if not has_inf:
        checks.append(('✓', 'No Inf values', 'green'))
    else:
        checks.append(('✗', 'Inf values detected!', 'red'))
    
    # Check 4: Reasonable Z_eff range
    z_min, z_max = Z_eff.min(), Z_eff.max()
    if 0 < z_min < 5 and 20 < z_max < 50:
        checks.append(('✓', f'Z_eff range reasonable: [{z_min:.1f}, {z_max:.1f}]', 'green'))
    else:
        checks.append(('⚠', f'Z_eff range unusual: [{z_min:.1f}, {z_max:.1f}]', 'orange'))
    
    # Check 5: Non-negative basis coefficients
    if (A1 >= 0).all() and (A2 >= 0).all():
        checks.append(('✓', 'A₁, A₂ non-negative (physical)', 'green'))
    else:
        checks.append(('✗', 'Negative A₁ or A₂ detected!', 'red'))

    # Check 6: grad_R (ratio-gradient channel) finite, no NaN/Inf
    has_grad_r_nan_inf = np.isnan(grad_R).any() or np.isinf(grad_R).any()
    if not has_grad_r_nan_inf:
        checks.append(('✓', f'grad_R finite (no NaN/Inf), range=[{grad_R.min():.3f}, {grad_R.max():.3f}]', 'green'))
    else:
        checks.append(('✗', 'NaN or Inf detected in grad_R!', 'red'))
    
    # Display checks
    y_pos = 0.9
    for symbol, text, color in checks:
        ax.text(0.1, y_pos, f"{symbol} {text}", transform=ax.transAxes, fontsize=12,
                color=color, family='monospace')
        y_pos -= 0.1
    
    # Summary stats
    stats_text = (
        f"\nSUMMARY STATISTICS\n"
        f"==================\n"
        f"L_low:  range=[{L_low.min():.3f}, {L_low.max():.3f}], mean={L_low.mean():.3f}\n"
        f"L_high: range=[{L_high.min():.3f}, {L_high.max():.3f}], mean={L_high.mean():.3f}\n"
        f"Z_eff:  range=[{Z_eff.min():.1f}, {Z_eff.max():.1f}], mean={Z_eff.mean():.1f}\n"
        f"A₁:     range=[{A1.min():.3f}, {A1.max():.3f}]\n"
        f"A₂:     range=[{A2.min():.3f}, {A2.max():.3f}]\n"
        f"grad_R: range=[{grad_R.min():.3f}, {grad_R.max():.3f}], mean={grad_R.mean():.3f}\n"
    )
    ax.text(0.1, 0.35, stats_text, transform=ax.transAxes, fontsize=10, family='monospace')
    
    plt.tight_layout()
    save_path = output_dir / '07_sanity_check.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def run_debug_pipeline(lo_path: str, hi_path: str):
    """Run the complete debug visualization pipeline."""
    print("=" * 70)
    print("AETHELGARD DEBUG VISUALIZATION PIPELINE")
    print("=" * 70)
    
    output_dir = setup_output_dir()
    print(f"\nOutput directory: {output_dir}")
    
    # Step 1: Load raw images
    print("\n[1/7] Loading raw images...")
    img_lo, img_hi = load_image_pair(lo_path, hi_path)
    
    # Step 2: Preprocessing
    print("\n[2/7] Running preprocessing (log-attenuation)...")
    preprocessor = RawToLogAttenuation(gaussian_sigma=1.0, i0_method='per_image_max')
    raw_tensor = torch.from_numpy(np.stack([img_lo, img_hi], axis=0)).unsqueeze(0).float()
    L_tensor, preprocess_debug = preprocessor(raw_tensor, return_debug=True)
    L_low = L_tensor[0, 0].numpy()
    L_high = L_tensor[0, 1].numpy()
    
    # Step 3: PhysicsHead
    print("\n[3/7] Running PhysicsHead (BHC decomposition)...")
    physics_head = PhysicsHead(order=2, learnable=False)
    with torch.no_grad():
        physics_maps, physics_debug = physics_head(L_tensor, return_debug=True)
    A1 = physics_debug['A1'][0, 0].numpy()
    A2 = physics_debug['A2'][0, 0].numpy()
    Z_eff = physics_debug['Z_eff'][0, 0].numpy()
    grad_R = physics_debug['grad_R'][0, 0].numpy()
    
    # Step 4: A-space filtering (optional, for diagnostics)
    print("\n[4/7] Running A-space noise filter...")
    noise_filter = ASpaceNoiseFilter()
    A1_t = torch.from_numpy(A1).unsqueeze(0).unsqueeze(0).float()
    A2_t = torch.from_numpy(A2).unsqueeze(0).unsqueeze(0).float()
    _, _, filter_diag = noise_filter.filter(A1_t, A2_t, return_diagnostics=True)
    
    # Generate visualizations
    print("\n[5/7] Generating visualizations...")
    plot_01_raw_input(img_lo, img_hi, output_dir)
    plot_02_preprocessing(img_lo, img_hi, L_low, L_high, preprocess_debug, output_dir)
    plot_03_bhc_comparison(L_low, L_high, Z_eff, output_dir)
    plot_04_aspace_scatter(A1, A2, filter_diag[0] if filter_diag else {}, output_dir)
    plot_05_zeff_histogram(Z_eff, L_low, output_dir)
    plot_06_pseudocolor(Z_eff, output_dir)
    plot_07_sanity_check(L_low, L_high, Z_eff, A1, A2, grad_R, output_dir)
    
    print("\n" + "=" * 70)
    print("DEBUG VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Debug visualization for Aethelgard pipeline')
    parser.add_argument('--lo', type=str, help='Path to low-energy TIF')
    parser.add_argument('--hi', type=str, help='Path to high-energy TIF')
    args = parser.parse_args()
    
    if args.lo and args.hi:
        run_debug_pipeline(args.lo, args.hi)
    else:
        # Default: use first image pair from NotThreats
        data_dir = Path(__file__).parent.parent / "HUMS-X-ray-Dataset" / "NotThreats"
        lo_files = sorted(data_dir.glob("*_lo.tif"))
        if lo_files:
            lo_path = str(lo_files[0])
            hi_path = lo_path.replace("_lo.tif", "_hi.tif")
            run_debug_pipeline(lo_path, hi_path)
        else:
            print("No image files found. Please specify --lo and --hi paths.")
            raise FileNotFoundError(
                f"No *_lo.tif files found in {data_dir}; "
                "specify --lo/--hi explicitly."
            )
