"""
Preprocessing Module for Dual-Energy X-Ray Images

This module handles the critical conversion from raw detector intensity values
to log-attenuation space, which is required before physics-based decomposition.

THE PHYSICS:
------------
Raw X-ray detectors measure photon counts (intensity I). The Beer-Lambert law
relates intensity to attenuation:

    I = I_0 * exp(-μt)

Where:
    I   = Measured intensity (what the detector sees)
    I_0 = Incident intensity (unattenuated beam through air)
    μ   = Linear attenuation coefficient
    t   = Material thickness

To linearize this for the Alvarez-Macovski decomposition, we compute
the log-attenuation L:

    L = -ln(I / I_0) = μt

POISSON SHOT NOISE & PRE-LOG GAUSSIAN BLUR:
-------------------------------------------
X-ray photon detection is fundamentally a quantum counting process. Each pixel
counts discrete photon arrivals, which follow a Poisson distribution:

    P(k photons) = (λ^k * e^(-λ)) / k!

Where λ is the expected number of photons. Key property: Var(k) = λ.

This means:
    - High-intensity pixels (λ=10000): relative noise = √10000/10000 = 1%
    - Low-intensity pixels (λ=100): relative noise = √100/100 = 10%

When we take the logarithm L = -ln(I/I₀), the derivative is -1/I, so noise
is AMPLIFIED in low-intensity (high-attenuation) regions. A small absolute
fluctuation in I causes a large change in L.

The solution: Apply Gaussian blur BEFORE the log transform to average out
shot noise while preserving material structure. This is a physics-informed
denoising step, not a learned filter.

THE TRAP (Watch for this!):
---------------------------
- If I = 0 (photon starvation in dense regions), ln(0) = -∞ → NaN explosion
- If I > I_0 (noise/scattered photons), ln(I/I_0) < 0 → negative attenuation (unphysical)

THE FIX:
--------
We clamp the transmission ratio T = I/I_0 to the range [epsilon, 1.0]:
    - epsilon ≈ 1e-6 prevents log(0)
    - Upper bound of 1.0 prevents negative attenuation

Author: Michael Moran
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

# Import scipy for Gaussian filtering
try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Gaussian blur will use PyTorch fallback.")


def create_gaussian_kernel(sigma: float, kernel_size: int = None) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel for convolution-based blurring.
    
    This is a pure PyTorch implementation for GPU compatibility.
    
    Args:
        sigma: Standard deviation of the Gaussian
        kernel_size: Size of the kernel (must be odd). If None, auto-computed.
        
    Returns:
        Gaussian kernel tensor of shape (1, 1, kernel_size, kernel_size)
    """
    if kernel_size is None:
        # Rule of thumb: kernel should span ~3 sigma on each side
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd size
    
    # Create 1D Gaussian
    x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    
    # Create 2D Gaussian via outer product
    gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
    
    # Reshape for conv2d: (out_channels, in_channels, H, W)
    return gauss_2d.unsqueeze(0).unsqueeze(0)


class RawToLogAttenuation(nn.Module):
    """
    Differentiable preprocessing layer that converts raw 16-bit intensity
    images to log-attenuation space.
    
    This is NOT a learned layer - it implements deterministic physics.
    However, we make it an nn.Module so it can be part of a differentiable
    pipeline and run on GPU.
    
    Input:  Raw intensity tensor (B, 2, H, W) - [Low Energy, High Energy]
    Output: Log-attenuation tensor (B, 2, H, W) - [L_low, L_high]
    
    PREPROCESSING PIPELINE:
    -----------------------
    1. Gaussian blur (denoise Poisson shot noise BEFORE log transform)
    2. Estimate I_0 (incident beam intensity)
    3. Compute transmission T = I / I_0
    4. Clamp T to [epsilon, 1.0] for numerical stability
    5. Compute log-attenuation L = -ln(T)
    
    I_0 Estimation Strategy:
    ------------------------
    Without calibration images, we estimate I_0 as the maximum pixel value
    in each image. This assumes:
    1. There exists at least one "air" pixel with minimal attenuation
    2. The brightest pixel represents the unattenuated beam
    
    This is a "robust approximation" - not perfect, but stable.
    """
    
    def __init__(
        self,
        epsilon: float = 1e-6,
        gaussian_sigma: float = 1.0,
        i0_method: str = "per_image_max",
        i0_percentile: float = 99.5,
        max_bit_depth: int = 16
    ):
        """
        Initialize the preprocessing layer.
        
        Args:
            epsilon: Minimum transmission ratio to prevent log(0). 
                     Physics justification: Even in dense regions, some
                     photons scatter through. epsilon=1e-6 corresponds to
                     ~14 half-value layers of attenuation.
            gaussian_sigma: Standard deviation for pre-log Gaussian blur.
                           Set to 0 to disable blur. Default 1.0 pixels.
                           Physics justification: Averages out Poisson shot
                           noise before the log transform amplifies it.
            i0_method: How to estimate I_0. Options:
                - "per_image_max": Use max pixel value in each image
                - "per_image_percentile": Use percentile (more robust to hot pixels)
                - "global": Use theoretical max (2^bit_depth - 1)
            i0_percentile: Percentile to use if i0_method="per_image_percentile"
            max_bit_depth: Bit depth of input images (16 for our TIFs)
        """
        super().__init__()
        
        # Store as buffers (not parameters - these don't get gradients)
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.register_buffer("global_i0", torch.tensor(2**max_bit_depth - 1, dtype=torch.float32))
        
        self.gaussian_sigma = gaussian_sigma
        self.i0_method = i0_method
        self.i0_percentile = i0_percentile
        
        # Pre-compute Gaussian kernel if sigma > 0
        if self.gaussian_sigma > 0:
            kernel = create_gaussian_kernel(self.gaussian_sigma)
            self.register_buffer("gaussian_kernel", kernel)
        else:
            self.gaussian_kernel = None
    
    def _apply_gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian blur to denoise Poisson shot noise.
        
        WHY BLUR BEFORE LOG:
        --------------------
        The log transform L = -ln(I/I₀) has derivative dL/dI = -1/I.
        This means noise is amplified by a factor of 1/I in low-intensity regions.
        
        Example with I₀ = 10000:
            - Air region (I = 9900): L = 0.01, noise amplification = 1/9900 ≈ 0.0001
            - Dense region (I = 100): L = 4.6, noise amplification = 1/100 = 0.01
            - Very dense (I = 10): L = 6.9, noise amplification = 1/10 = 0.1
        
        By blurring BEFORE the log, we average out the raw photon count fluctuations
        while they're still in linear space, before the nonlinear amplification.
        
        Args:
            x: Raw intensity tensor (B, C, H, W)
            
        Returns:
            Blurred intensity tensor (B, C, H, W)
        """
        if self.gaussian_sigma <= 0 or self.gaussian_kernel is None:
            return x
        
        B, C, H, W = x.shape
        
        # Get kernel and ensure it's on the same device as input
        kernel = self.gaussian_kernel.to(x.device)
        
        # Calculate padding to maintain spatial dimensions
        kernel_size = kernel.shape[-1]
        padding = kernel_size // 2
        
        # Apply blur to each channel separately (kernel is single-channel)
        # We use groups=C to apply the same kernel to each channel independently
        blurred_channels = []
        for c in range(C):
            channel = x[:, c:c+1, :, :]  # (B, 1, H, W)
            blurred = F.conv2d(channel, kernel, padding=padding)
            blurred_channels.append(blurred)
        
        return torch.cat(blurred_channels, dim=1)
    
    def estimate_i0(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate the incident intensity I_0 for normalization.
        
        Args:
            x: Raw intensity tensor (B, C, H, W)
            
        Returns:
            I_0 tensor of shape (B, C, 1, 1) for broadcasting
        """
        if self.i0_method == "global":
            # Use theoretical maximum (fastest, but assumes well-calibrated detector)
            return self.global_i0.expand(x.shape[0], x.shape[1], 1, 1)
            
        elif self.i0_method == "per_image_max":
            # Use maximum pixel value per channel per image
            # Shape: (B, C) -> (B, C, 1, 1)
            i0 = x.amax(dim=(-2, -1), keepdim=True)
            return i0
            
        elif self.i0_method == "per_image_percentile":
            # Use percentile (robust to hot pixels / cosmic ray hits)
            # More expensive but handles outliers
            B, C, H, W = x.shape
            x_flat = x.view(B, C, -1)  # (B, C, H*W)
            
            # Compute percentile along spatial dimension
            k = int((self.i0_percentile / 100.0) * (H * W))
            k = max(1, min(k, H * W - 1))  # Clamp to valid range
            
            # torch.kthvalue returns (values, indices)
            # We want the k-th largest, so use (total - k)-th smallest
            i0, _ = x_flat.kthvalue(H * W - k + 1, dim=-1, keepdim=True)
            return i0.unsqueeze(-1)  # (B, C, 1, 1)
            
        else:
            raise ValueError(f"Unknown i0_method: {self.i0_method}")
    
    def forward(
        self, 
        x: torch.Tensor,
        return_debug: bool = False
    ) -> torch.Tensor:
        """
        Convert raw intensity to log-attenuation.
        
        Physics Pipeline:
        1. Apply Gaussian blur (denoise Poisson shot noise)
        2. Estimate I_0 (incident beam intensity)
        3. Compute transmission T = I / I_0
        4. Clamp T to [epsilon, 1.0] for numerical stability
        5. Compute log-attenuation L = -ln(T)
        
        Args:
            x: Raw intensity tensor (B, 2, H, W) dtype=float32
               Channel 0: Low energy image
               Channel 1: High energy image
            return_debug: If True, also return debug_dict with intermediate states
               
        Returns:
            Log-attenuation tensor (B, 2, H, W)
            L = -ln(I/I_0), where L ≥ 0 for physical materials
            
            If return_debug=True, also returns debug_dict containing:
                - 'raw_input': Original input before any processing
                - 'after_blur': After Gaussian blur, before log transform
                - 'i0_estimated': Estimated incident intensity
                - 'transmission': T = I/I_0 before clamping
                - 'transmission_clamped': T after clamping to [epsilon, 1]
        """
        # Ensure float32 for numerical stability
        x = x.float()
        
        # Store raw input for debug
        raw_input = x.clone() if return_debug else None
        
        # Step 1: Gaussian blur BEFORE log transform
        # This averages out Poisson shot noise while still in linear space,
        # before the log transform amplifies noise in low-intensity regions.
        x = self._apply_gaussian_blur(x)
        
        # Store after blur for debug (this is the "denoised" state)
        after_blur = x.clone() if return_debug else None
        
        # Step 2: Estimate I_0
        i0 = self.estimate_i0(x)
        
        # Step 3: Compute transmission ratio T = I / I_0
        # Add epsilon to I_0 to prevent division by zero if image is all black
        transmission = x / (i0 + self.epsilon)
        
        # Store transmission before clamping for debug
        transmission_pre_clamp = transmission.clone() if return_debug else None
        
        # Step 4: Clamp transmission to physically valid range
        # T > 1 means more photons detected than incident (unphysical - noise/scatter)
        # T < epsilon means we're in photon starvation territory
        transmission = torch.clamp(transmission, min=self.epsilon, max=1.0)
        
        # Step 5: Compute log-attenuation
        # L = -ln(T) = -ln(I/I_0) = ln(I_0) - ln(I)
        # For T ∈ [epsilon, 1], L ∈ [0, -ln(epsilon)] ≈ [0, 13.8]
        log_attenuation = -torch.log(transmission)
        
        if return_debug:
            debug_dict = {
                'raw_input': raw_input,
                'after_blur': after_blur,
                'i0_estimated': i0,
                'transmission': transmission_pre_clamp,
                'transmission_clamped': transmission,
            }
            return log_attenuation, debug_dict
        
        return log_attenuation
    
    def inverse(self, log_atten: torch.Tensor, i0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convert log-attenuation back to intensity (for visualization).
        
        I = I_0 * exp(-L)
        
        Note: This does NOT undo the Gaussian blur (that information is lost).
        
        Args:
            log_atten: Log-attenuation tensor (B, 2, H, W)
            i0: Optional I_0 tensor. If None, uses global_i0.
            
        Returns:
            Intensity tensor (B, 2, H, W)
        """
        if i0 is None:
            i0 = self.global_i0
        return i0 * torch.exp(-log_atten)


def numpy_to_log_attenuation(
    img_low: np.ndarray,
    img_high: np.ndarray,
    epsilon: float = 1e-6,
    gaussian_sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    NumPy-based preprocessing for data exploration (non-differentiable).
    
    This function is useful for initial data analysis before building
    the full PyTorch pipeline.
    
    Args:
        img_low: Low energy image (H, W), dtype=uint16 or float
        img_high: High energy image (H, W), dtype=uint16 or float
        epsilon: Minimum transmission ratio
        gaussian_sigma: Sigma for pre-log Gaussian blur (0 to disable)
        
    Returns:
        L_low: Log-attenuation of low energy image
        L_high: Log-attenuation of high energy image
        stats: Dictionary of calibration statistics
    """
    # Convert to float64 for precision
    img_low = img_low.astype(np.float64)
    img_high = img_high.astype(np.float64)
    
    # Apply Gaussian blur BEFORE log transform
    if gaussian_sigma > 0:
        if HAS_SCIPY:
            img_low = gaussian_filter(img_low, sigma=gaussian_sigma)
            img_high = gaussian_filter(img_high, sigma=gaussian_sigma)
        else:
            print("Warning: scipy not available, skipping Gaussian blur")
    
    # Estimate I_0 as max value (brightest = air)
    i0_low = img_low.max()
    i0_high = img_high.max()
    
    # Compute transmission
    T_low = img_low / (i0_low + epsilon)
    T_high = img_high / (i0_high + epsilon)
    
    # Clamp to valid range
    T_low = np.clip(T_low, epsilon, 1.0)
    T_high = np.clip(T_high, epsilon, 1.0)
    
    # Compute log-attenuation
    L_low = -np.log(T_low)
    L_high = -np.log(T_high)
    
    # Collect statistics for debugging
    stats = {
        "i0_low": i0_low,
        "i0_high": i0_high,
        "L_low_range": (L_low.min(), L_low.max()),
        "L_high_range": (L_high.min(), L_high.max()),
        "air_fraction": np.mean(L_low < 0.01),  # Fraction of near-zero attenuation
        "gaussian_sigma": gaussian_sigma,
    }
    
    return L_low, L_high, stats


# ==============================================================================
# UNIT TEST / SANITY CHECK
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RawToLogAttenuation Unit Test")
    print("=" * 60)
    
    # Create synthetic 16-bit test data
    # Simulate: air region (high intensity) and dense object (low intensity)
    H, W = 64, 64
    
    # Air = 60000 counts, Dense object = 100 counts
    test_low = torch.ones(1, 1, H, W) * 60000
    test_low[:, :, 20:40, 20:40] = 100  # Dense region
    
    test_high = torch.ones(1, 1, H, W) * 62000  # Slightly different I_0
    test_high[:, :, 20:40, 20:40] = 500  # Less attenuation at high energy
    
    # Stack into dual-energy input
    test_input = torch.cat([test_low, test_high], dim=1)  # (1, 2, 64, 64)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Input range: [{test_input.min():.0f}, {test_input.max():.0f}]")
    
    # Test WITHOUT Gaussian blur
    print("\n--- Test 1: Without Gaussian blur (sigma=0) ---")
    preprocess_no_blur = RawToLogAttenuation(epsilon=1e-6, gaussian_sigma=0.0, i0_method="per_image_max")
    log_atten_no_blur = preprocess_no_blur(test_input)
    
    print(f"Output shape: {log_atten_no_blur.shape}")
    print(f"L_low range: [{log_atten_no_blur[0, 0].min():.4f}, {log_atten_no_blur[0, 0].max():.4f}]")
    
    # Test WITH Gaussian blur (default sigma=1.0)
    print("\n--- Test 2: With Gaussian blur (sigma=1.0) ---")
    preprocess_with_blur = RawToLogAttenuation(epsilon=1e-6, gaussian_sigma=1.0, i0_method="per_image_max")
    log_atten_with_blur = preprocess_with_blur(test_input)
    
    print(f"Output shape: {log_atten_with_blur.shape}")
    print(f"L_low range: [{log_atten_with_blur[0, 0].min():.4f}, {log_atten_with_blur[0, 0].max():.4f}]")
    
    # Check physics: air should have L ≈ 0, dense region should have L > 0
    air_L = log_atten_with_blur[0, 0, 0, 0].item()
    dense_L_low = log_atten_with_blur[0, 0, 30, 30].item()
    dense_L_high = log_atten_with_blur[0, 1, 30, 30].item()
    
    print(f"\nPhysics Check (with blur):")
    print(f"  Air attenuation (L_low): {air_L:.4f} (should be ≈ 0)")
    print(f"  Dense region L_low: {dense_L_low:.4f}")
    print(f"  Dense region L_high: {dense_L_high:.4f}")
    print(f"  Ratio L_low/L_high: {dense_L_low/dense_L_high:.4f}")
    print(f"  (Ratio > 1 expected: low energy attenuates more)")
    
    # Test 3: Verify blur reduces edge sharpness (expected tradeoff)
    print("\n--- Test 3: Edge smoothing effect ---")
    edge_no_blur = torch.abs(log_atten_no_blur[0, 0, 19, 30] - log_atten_no_blur[0, 0, 20, 30]).item()
    edge_with_blur = torch.abs(log_atten_with_blur[0, 0, 19, 30] - log_atten_with_blur[0, 0, 20, 30]).item()
    print(f"  Edge gradient without blur: {edge_no_blur:.4f}")
    print(f"  Edge gradient with blur: {edge_with_blur:.4f}")
    print(f"  Blur smooths edges by: {100*(1 - edge_with_blur/edge_no_blur):.1f}%")
    
    # Verify no NaN or Inf
    assert not torch.isnan(log_atten_with_blur).any(), "NaN detected in output!"
    assert not torch.isinf(log_atten_with_blur).any(), "Inf detected in output!"
    assert (log_atten_with_blur >= 0).all(), "Negative attenuation detected (unphysical)!"
    
    print("\n✓ All tests passed!")
