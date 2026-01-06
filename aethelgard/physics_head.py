"""
PhysicsHead: Differentiable Alvarez-Macovski Decomposition Layer

This module implements the core physics transformation that converts
dual-energy log-attenuation maps into material property maps.

THE PHYSICS (Alvarez-Macovski Theorem, 1976):
---------------------------------------------
In the diagnostic X-ray energy range (30-200 keV), the linear attenuation
coefficient μ(E) can be decomposed into two basis functions:

    μ(r, E) ≈ a₁(r)·f₁(E) + a₂(r)·f₂(E)

Where:
    a₁(r) = Photoelectric coefficient (scales as ρ·Z³ to ρ·Z⁴)
    f₁(E) = Photoelectric energy dependence ≈ 1/E³
    
    a₂(r) = Compton coefficient (scales as electron density ρₑ ≈ ρ)
    f₂(E) = Klein-Nishina cross-section

THE DECOMPOSITION PROBLEM:
--------------------------
Given two measurements (L_low, L_high) of the same ray path at different
effective energies, we solve for the line integrals:

    A₁ = ∫ a₁(r) ds  (Photoelectric line integral)
    A₂ = ∫ a₂(r) ds  (Compton line integral)

This is a system of 2 nonlinear equations with 2 unknowns.
We approximate the inverse mapping with a polynomial:

    A₁ = Σᵢⱼ c¹ᵢⱼ · L_low^i · L_high^j
    A₂ = Σᵢⱼ c²ᵢⱼ · L_low^i · L_high^j

THE DERIVED FEATURES:
---------------------
From A₁ and A₂, we derive physically meaningful features:

1. Z_eff (Effective Atomic Number):
   Since a₁ ∝ ρZ³ and a₂ ∝ ρ, we have:
   
       A₁/A₂ ∝ (ρZ³·t)/(ρ·t) = Z³
       
   Therefore: Z_eff ≈ (A₁/A₂)^(1/3)
   
   This is the "1/3 exponent" — it comes from inverting the Z³ scaling
   of the photoelectric effect.

2. ∇R (Ratio Gradient - Material Edge Detector):
   The ratio R = L_low/L_high is thickness-invariant:
   
       R = (μ_low·t)/(μ_high·t) = μ_low/μ_high
       
   The gradient ∇R is zero inside homogeneous materials and non-zero
   at material boundaries. This is a "compositional edge detector"
   that ignores geometric edges (thickness changes).

Author: Michael Moran
Supervisor: Dr. Thomas Anthony, CTO, Analytical AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree


# ==============================================================================
# A-SPACE NOISE FILTER
# ==============================================================================
class ASpaceNoiseFilter:
    """
    Two-Pass Noise Filter Operating in A-Space (Polar Coordinates).
    
    THE PHYSICS INTUITION:
    ----------------------
    In A-space, each pixel is a 2D point (A₁, A₂) which can be viewed
    in polar coordinates:
    
        θ = atan2(A₁, A₂)  → Encodes Z_eff (material type)
        r = √(A₁² + A₂²)   → Encodes thickness (total attenuation)
    
    The feasible region in A-space is a WEDGE bounded by:
        - θ_min: Angle corresponding to Z = 1 (hydrogen)
        - θ_max: Angle corresponding to Z = 92 (uranium)
    
    Points outside this wedge have physically impossible atomic numbers.
    
    NOISE DETECTION STRATEGY:
    -------------------------
    Pass 1 (Wedge Clamp):
        Hard physics constraint. Clamp θ to [θ_min, θ_max].
        Catches systematic errors with impossible Z_eff.
    
    Pass 2 (k-NN Outlier Detection):
        Statistical constraint. For each pixel, find its nearest neighbor
        IN A-SPACE (not image space). If the distance is too large, the
        pixel is an isolated outlier.
        
        Key insight: Real objects (even small ones) produce CLUSTERS of
        similar (θ, r) values. True noise produces ISOLATED points with
        no A-space neighbors.
    
    Author: Michael Moran
    Supervisor: Dr. Thomas Anthony, CTO, Analytical AI
    """
    
    def __init__(
        self,
        theta_min: float = 0.05,
        theta_max: float = 1.50,
        k_neighbors: int = 1,
        mad_multiplier: float = 3.0,
        subsample_threshold: int = 100000,
        epsilon: float = 1e-6
    ):
        """
        Initialize the A-Space Noise Filter.
        
        Args:
            theta_min: Minimum valid angle in A-space (radians).
                       Corresponds to very low Z materials.
                       Default 0.05 rad ≈ 3° (near Z=1)
            theta_max: Maximum valid angle in A-space (radians).
                       Corresponds to very high Z materials.
                       Default 1.50 rad ≈ 86° (near Z=92)
            k_neighbors: Number of nearest neighbors for outlier detection.
                        k=1 is most conservative (only truly isolated pixels).
            mad_multiplier: Threshold multiplier for Median Absolute Deviation.
                           Points with kNN distance > median + mad_multiplier*MAD
                           are flagged as outliers.
            subsample_threshold: If image has more pixels than this, subsample
                                for k-NN computation (performance optimization).
            epsilon: Small constant for numerical stability.
        """
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.k_neighbors = k_neighbors
        self.mad_multiplier = mad_multiplier
        self.subsample_threshold = subsample_threshold
        self.epsilon = epsilon
    
    def _cartesian_to_polar(
        self,
        A1: np.ndarray,
        A2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert (A₁, A₂) Cartesian coordinates to polar (θ, r).
        
        THE PHYSICS:
        ------------
        θ = atan2(A₁, A₂) encodes the ratio A₁/A₂, which determines Z_eff.
        r = √(A₁² + A₂²) encodes total line integral (thickness × density).
        
        We use atan2(A₁, A₂) NOT atan2(A₂, A₁) because:
        - A₁ is photoelectric (∝ Z³), increases faster with Z
        - A₂ is Compton (∝ ρ), relatively constant
        - Higher Z → higher A₁/A₂ ratio → larger θ
        
        Args:
            A1: Photoelectric coefficients (flattened array)
            A2: Compton coefficients (flattened array)
            
        Returns:
            theta: Angle in radians, in [0, π/2] for non-negative A1, A2
            r: Magnitude (distance from origin)
        """
        # atan2(y, x) - we put A1 as y so higher Z → larger angle
        theta = np.arctan2(A1 + self.epsilon, A2 + self.epsilon)
        r = np.sqrt(A1**2 + A2**2)
        return theta, r
    
    def _polar_to_cartesian(
        self,
        theta: np.ndarray,
        r: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert polar (θ, r) back to Cartesian (A₁, A₂).
        
        Args:
            theta: Angle in radians
            r: Magnitude
            
        Returns:
            A1: Photoelectric coefficients
            A2: Compton coefficients
        """
        # Inverse of atan2(A1, A2): A1 = r*sin(θ), A2 = r*cos(θ)
        A1 = r * np.sin(theta)
        A2 = r * np.cos(theta)
        return A1, A2
    
    def _pass1_wedge_clamp(
        self,
        theta: np.ndarray,
        r: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pass 1: Wedge Clamp - Enforce Physical Z_eff Bounds.
        
        THE PHYSICS:
        ------------
        The feasible region in A-space is a WEDGE, not the full quadrant.
        
            - θ < θ_min: Impossible (Z < 1, no such element)
            - θ > θ_max: Impossible (Z > 92, beyond uranium)
        
        These violations indicate sensor error or numerical artifacts.
        We CLAMP θ to the valid range while preserving r (thickness).
        
        Args:
            theta: Array of angles
            r: Array of magnitudes
            
        Returns:
            theta_clamped: Angles clamped to [θ_min, θ_max]
            clamp_mask: Boolean array, True where clamping occurred
        """
        # Track which pixels were clamped (for diagnostics)
        clamp_mask = (theta < self.theta_min) | (theta > self.theta_max)
        
        # Clamp to valid range
        theta_clamped = np.clip(theta, self.theta_min, self.theta_max)
        
        return theta_clamped, clamp_mask
    
    def _pass2_knn_outlier_detection(
        self,
        theta: np.ndarray,
        r: np.ndarray,
        original_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Pass 2: k-NN Outlier Detection in A-Space.
        
        THE PHYSICS:
        ------------
        Real objects produce CLUSTERS of pixels with similar (θ, r).
        - A metal key: many pixels all at high-θ, moderate-r
        - Clothing: many pixels at low-θ, varying r
        
        Noise produces ISOLATED pixels with no A-space neighbors.
        - Random sensor spike: one pixel at weird (θ, r)
        
        ALGORITHM:
        ----------
        1. Build a KD-tree of all (θ, r) points
        2. For each point, find distance to k-th nearest neighbor
        3. Compute median and MAD of these distances
        4. Flag points where distance > median + mad_multiplier * MAD
        5. Replace flagged points with nearest valid neighbor's values
        
        Args:
            theta: Array of angles (flattened)
            r: Array of magnitudes (flattened)
            original_shape: (H, W) for reshaping output
            
        Returns:
            theta_filtered: Filtered angles
            r_filtered: Filtered magnitudes
            stats: Dictionary with diagnostic statistics
        """
        n_pixels = len(theta)
        
        # Normalize θ and r to similar scales for distance computation
        # This prevents one dimension from dominating the distance metric
        theta_range = self.theta_max - self.theta_min
        r_max = np.percentile(r, 99) + self.epsilon  # Robust max
        
        theta_normalized = (theta - self.theta_min) / theta_range
        r_normalized = r / r_max
        
        # Stack into 2D points for k-NN
        points = np.column_stack([theta_normalized, r_normalized])
        
        # Build KD-tree for efficient nearest neighbor search
        # For very large images, we may need to subsample
        if n_pixels > self.subsample_threshold:
            # Subsample for tree building, but query all points
            subsample_idx = np.random.choice(
                n_pixels, self.subsample_threshold, replace=False
            )
            tree = cKDTree(points[subsample_idx])
        else:
            tree = cKDTree(points)
        
        # Query k+1 neighbors (first neighbor is the point itself)
        # Returns (distances, indices) for k+1 nearest neighbors
        distances, _ = tree.query(points, k=self.k_neighbors + 1)
        
        # Get distance to k-th nearest neighbor (excluding self)
        # If k=1, this is distance to closest other point
        knn_distances = distances[:, -1]  # Last column is k-th neighbor
        
        # Compute robust statistics
        median_dist = np.median(knn_distances)
        mad = np.median(np.abs(knn_distances - median_dist))
        
        # Threshold: points with distance > median + multiplier * MAD are outliers
        # Add epsilon to MAD to handle degenerate cases (all same distance)
        threshold = median_dist + self.mad_multiplier * (mad + self.epsilon)
        
        outlier_mask = knn_distances > threshold
        n_outliers = np.sum(outlier_mask)
        
        # For outliers, replace with nearest valid neighbor's values
        theta_filtered = theta.copy()
        r_filtered = r.copy()
        
        if n_outliers > 0:
            # Get indices of valid (non-outlier) points
            valid_mask = ~outlier_mask
            valid_points = points[valid_mask]
            valid_theta = theta[valid_mask]
            valid_r = r[valid_mask]
            
            if len(valid_points) > 0:
                # Build tree of valid points only
                valid_tree = cKDTree(valid_points)
                
                # For each outlier, find nearest valid point
                outlier_points = points[outlier_mask]
                _, nearest_valid_idx = valid_tree.query(outlier_points, k=1)
                
                # Replace outlier values with nearest valid neighbor
                theta_filtered[outlier_mask] = valid_theta[nearest_valid_idx]
                r_filtered[outlier_mask] = valid_r[nearest_valid_idx]
        
        # Compile statistics for diagnostics
        stats = {
            "n_pixels": n_pixels,
            "n_outliers": n_outliers,
            "outlier_fraction": n_outliers / n_pixels if n_pixels > 0 else 0,
            "median_knn_distance": median_dist,
            "mad_knn_distance": mad,
            "threshold": threshold,
            "max_knn_distance": np.max(knn_distances),
        }
        
        return theta_filtered, r_filtered, stats
    
    def filter(
        self,
        A1: torch.Tensor,
        A2: torch.Tensor,
        return_diagnostics: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply two-pass A-space noise filter.
        
        Args:
            A1: Photoelectric line integral map (B, 1, H, W)
            A2: Compton line integral map (B, 1, H, W)
            return_diagnostics: If True, also return diagnostic info
            
        Returns:
            A1_filtered: Filtered photoelectric map (B, 1, H, W)
            A2_filtered: Filtered Compton map (B, 1, H, W)
            diagnostics: (optional) Dictionary with filter statistics
        """
        # Store original properties
        device = A1.device
        dtype = A1.dtype
        B, C, H, W = A1.shape
        
        # Process each batch item separately
        A1_filtered_list = []
        A2_filtered_list = []
        all_diagnostics = []
        
        for b in range(B):
            # Convert to numpy for scipy operations
            a1_np = A1[b, 0].detach().cpu().numpy().flatten()
            a2_np = A2[b, 0].detach().cpu().numpy().flatten()
            
            # Convert to polar coordinates
            theta, r = self._cartesian_to_polar(a1_np, a2_np)
            
            # Pass 1: Wedge Clamp
            theta, clamp_mask = self._pass1_wedge_clamp(theta, r)
            
            # Pass 2: k-NN Outlier Detection
            theta, r, stats = self._pass2_knn_outlier_detection(theta, r, (H, W))
            
            # Add clamp info to stats
            stats["n_wedge_clamped"] = np.sum(clamp_mask)
            stats["wedge_clamp_fraction"] = np.sum(clamp_mask) / len(clamp_mask)
            
            # Convert back to Cartesian
            a1_filtered, a2_filtered = self._polar_to_cartesian(theta, r)
            
            # Reshape to image dimensions
            a1_filtered = a1_filtered.reshape(H, W)
            a2_filtered = a2_filtered.reshape(H, W)
            
            # Convert back to torch
            A1_filtered_list.append(
                torch.from_numpy(a1_filtered).to(device=device, dtype=dtype)
            )
            A2_filtered_list.append(
                torch.from_numpy(a2_filtered).to(device=device, dtype=dtype)
            )
            all_diagnostics.append(stats)
        
        # Stack batch dimension
        A1_filtered = torch.stack(A1_filtered_list, dim=0).unsqueeze(1)
        A2_filtered = torch.stack(A2_filtered_list, dim=0).unsqueeze(1)
        
        if return_diagnostics:
            return A1_filtered, A2_filtered, all_diagnostics
        return A1_filtered, A2_filtered
    
    def __repr__(self):
        return (
            f"ASpaceNoiseFilter("
            f"θ_range=[{self.theta_min:.2f}, {self.theta_max:.2f}], "
            f"k={self.k_neighbors}, "
            f"mad_mult={self.mad_multiplier})"
        )


class PhysicsHead(nn.Module):
    """
    Differentiable Physics Layer for Alvarez-Macovski Decomposition.
    
    Maps Dual-Energy Log-Attenuation → Basis Material Maps.
    
    Input:  (B, 2, H, W) - [L_low, L_high] log-attenuation images
    Output: (B, 4, H, W) - [A₁, A₂, Z_eff, ∇R] physics feature maps
    
    Output Channels:
        0: A₁ - Photoelectric coefficient (high for metals, low for organics)
        1: A₂ - Compton coefficient (proportional to electron density)
        2: Z_eff - Effective atomic number (dimensionless, ~6 for carbon, ~26 for iron)
        3: ∇R - Ratio gradient magnitude (material edge detector)
    
    The polynomial coefficients can be:
        - Initialized heuristically (default)
        - Loaded from phantom calibration data
        - Fine-tuned via backpropagation during training
    """
    
    def __init__(
        self,
        order: int = 2,
        learnable: bool = True,
        z_scale: float = 10.0,
        z_offset: float = 0.0,
        epsilon: float = 1e-6
    ):
        """
        Initialize the PhysicsHead.
        
        Args:
            order: Polynomial order for decomposition. Order=2 gives 6 terms:
                   {1, L, H, L², LH, H²}
            learnable: If True, coefficients are nn.Parameters (can be fine-tuned).
                      If False, coefficients are buffers (frozen).
            z_scale: Scaling factor for Z_eff output (helps with interpretability)
            z_offset: Offset for Z_eff output
            epsilon: Small constant for numerical stability (prevents division by zero)
        """
        super().__init__()
        
        self.order = order
        self.epsilon = epsilon
        
        # Number of polynomial terms: (n+1)(n+2)/2 for 2D polynomial of order n
        # Order 2: 1 + 2 + 3 = 6 terms
        self.num_terms = (order + 1) * (order + 2) // 2
        
        # =====================================================================
        # LEARNABLE COEFFICIENTS
        # Shape: (num_terms,) - will be broadcast across spatial dimensions
        # =====================================================================
        
        # Coefficients for A₁ (Photoelectric) decomposition
        coeffs_a1 = torch.zeros(self.num_terms)
        
        # Coefficients for A₂ (Compton) decomposition  
        coeffs_a2 = torch.zeros(self.num_terms)
        
        # Initialize with physics-informed heuristics
        self._init_heuristic_coefficients(coeffs_a1, coeffs_a2)
        
        if learnable:
            self.coeffs_a1 = nn.Parameter(coeffs_a1)
            self.coeffs_a2 = nn.Parameter(coeffs_a2)
        else:
            self.register_buffer("coeffs_a1", coeffs_a1)
            self.register_buffer("coeffs_a2", coeffs_a2)
        
        # Z_eff scaling parameters (also learnable)
        if learnable:
            self.z_scale = nn.Parameter(torch.tensor(z_scale))
            self.z_offset = nn.Parameter(torch.tensor(z_offset))
        else:
            self.register_buffer("z_scale", torch.tensor(z_scale))
            self.register_buffer("z_offset", torch.tensor(z_offset))
    
    def _init_heuristic_coefficients(
        self,
        coeffs_a1: torch.Tensor,
        coeffs_a2: torch.Tensor
    ):
        """
        Initialize coefficients with physics-informed heuristics.
        
        THE PHYSICS INTUITION:
        ----------------------
        Without calibration data, we use the following approximations:
        
        A₁ (Photoelectric) is enhanced at LOW energy (more absorption).
        A₂ (Compton) is relatively constant but dominates at HIGH energy.
        
        A simple first-order approximation:
            A₁ ≈ c₁·L_low - c₂·L_high  (difference isolates photoelectric)
            A₂ ≈ c₃·L_high            (high energy ≈ Compton-dominated)
        
        For order=2, term indices are:
            0: constant (1)
            1: L_low
            2: L_high  
            3: L_low²
            4: L_low·L_high
            5: L_high²
        """
        # ---------------------------------------------------------------------
        # A₁ (Photoelectric) initialization
        # Photoelectric effect is stronger at low energies, so we emphasize
        # the difference between low and high energy attenuation.
        # ---------------------------------------------------------------------
        coeffs_a1[0] = 0.0      # No constant offset
        coeffs_a1[1] = 1.5      # Strong positive weight on L_low
        coeffs_a1[2] = -0.5     # Subtract some L_high (isolate photoelectric)
        # Higher order terms start small
        if self.num_terms > 3:
            coeffs_a1[3] = 0.01   # L_low² (small quadratic correction)
            coeffs_a1[4] = -0.02  # L_low·L_high
            coeffs_a1[5] = 0.01   # L_high²
        
        # ---------------------------------------------------------------------
        # A₂ (Compton) initialization
        # Compton scattering dominates at higher energies and is proportional
        # to electron density (≈ mass density for light elements).
        # ---------------------------------------------------------------------
        coeffs_a2[0] = 0.0      # No constant offset
        coeffs_a2[1] = 0.5      # Some contribution from L_low
        coeffs_a2[2] = 1.0      # Primary weight on L_high
        # Higher order terms
        if self.num_terms > 3:
            coeffs_a2[3] = 0.01
            coeffs_a2[4] = -0.01
            coeffs_a2[5] = 0.02
    
    def _build_polynomial_stack(
        self,
        L_low: torch.Tensor,
        L_high: torch.Tensor
    ) -> torch.Tensor:
        """
        Build the polynomial feature tensor from L_low and L_high.
        
        For order=2, generates: {1, L, H, L², LH, H²}
        
        Args:
            L_low: Low energy log-attenuation (B, 1, H, W)
            L_high: High energy log-attenuation (B, 1, H, W)
            
        Returns:
            Polynomial stack tensor (B, num_terms, H, W)
        """
        terms = []
        
        # Constant term (bias)
        terms.append(torch.ones_like(L_low))
        
        # Build polynomial terms systematically
        # For order n, we want all terms L^i * H^j where i + j <= n
        for degree in range(1, self.order + 1):
            for j in range(degree + 1):
                i = degree - j
                # Term = L_low^i * L_high^j
                term = (L_low ** i) * (L_high ** j)
                terms.append(term)
        
        # Stack along channel dimension
        poly_stack = torch.cat(terms, dim=1)  # (B, num_terms, H, W)
        
        return poly_stack
    
    def _compute_basis_decomposition(
        self,
        poly_stack: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute A₁ and A₂ line integral maps via polynomial dot product.
        
        This applies the learned polynomial coefficients to the feature stack
        to produce the Alvarez-Macovski basis decomposition. Mathematically:
        
            A₁[x,y] = Σᵢ coeffs_a1[i] · poly_stack[i, x, y]
            A₂[x,y] = Σᵢ coeffs_a2[i] · poly_stack[i, x, y]
        
        This is equivalent to a 1x1 convolution with learned weights,
        but implemented explicitly for clarity.
        
        Args:
            poly_stack: Polynomial features (B, num_terms, H, W)
                        Contains [1, L, H, L², LH, H²] for each pixel
            
        Returns:
            A1: Photoelectric line integral map (B, 1, H, W) - NOT the coefficients!
            A2: Compton line integral map (B, 1, H, W) - NOT the coefficients!
            
        Note:
            The "coefficients" (coeffs_a1, coeffs_a2) are the polynomial WEIGHTS.
            The outputs (A₁, A₂) are the computed line integrals for each pixel.
        """
        # Reshape coefficients for broadcasting: (num_terms,) -> (1, num_terms, 1, 1)
        c1 = self.coeffs_a1.view(1, -1, 1, 1)
        c2 = self.coeffs_a2.view(1, -1, 1, 1)
        
        # Weighted sum over polynomial terms
        # (B, num_terms, H, W) * (1, num_terms, 1, 1) -> sum over dim=1 -> (B, 1, H, W)
        A1 = (poly_stack * c1).sum(dim=1, keepdim=True)
        A2 = (poly_stack * c2).sum(dim=1, keepdim=True)
        
        # ---------------------------------------------------------------------
        # PHYSICAL CONSTRAINT: Non-negativity
        # Mass attenuation coefficients cannot be negative. This is a hard
        # physics constraint. We use ReLU to enforce the feasible region.
        # 
        # WARNING: This creates a gradient discontinuity at zero. For very
        # noisy data, consider using softplus: log(1 + exp(x)) for smoother
        # gradients, but ReLU is more physically correct.
        # ---------------------------------------------------------------------
        A1 = F.relu(A1)
        A2 = F.relu(A2)
        
        return A1, A2
    
    def _compute_z_effective(
        self,
        A1: torch.Tensor,
        A2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute effective atomic number from basis coefficients.
        
        THE PHYSICS:
        ------------
        Photoelectric coefficient: a₁ ∝ ρ·Z^n (n ≈ 3 to 4)
        Compton coefficient: a₂ ∝ ρ (electron density ≈ mass density)
        
        Therefore:
            A₁/A₂ ∝ (ρ·Z³·t)/(ρ·t) = Z³
            
        And:
            Z_eff = (A₁/A₂)^(1/3)
        
        THE 1/3 EXPONENT EXPLAINED:
        ---------------------------
        If A₁ ∝ Z³, then to recover Z we must take the cube root.
        This is NOT a tunable parameter — it's derived from the physics.
        
        In reality, the exponent varies between 3 and 4 depending on
        energy and material. We use 3 as a reasonable approximation.
        
        Args:
            A1: Photoelectric coefficient (B, 1, H, W)
            A2: Compton coefficient (B, 1, H, W)
            
        Returns:
            Z_eff: Effective atomic number (B, 1, H, W), scaled for interpretability
        """
        # Compute ratio with epsilon to prevent division by zero
        # In air regions, A2 ≈ 0, so we need numerical protection
        ratio = A1 / (A2 + self.epsilon)
        
        # Cube root to invert the Z³ scaling
        # torch.pow handles negative inputs by returning NaN, but we've
        # already ensured A1, A2 >= 0 via ReLU
        z_raw = torch.pow(ratio + self.epsilon, 1.0 / 3.0)
        
        # Scale and offset for interpretability
        # Default scaling aims to put carbon ~6, iron ~26
        Z_eff = self.z_scale * z_raw + self.z_offset
        
        return Z_eff
    
    def _compute_ratio_gradient(
        self,
        L_low: torch.Tensor,
        L_high: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the gradient of the dual-energy ratio map.
        
        THE PHYSICS:
        ------------
        The ratio R = L_low / L_high is thickness-invariant:
        
            R = (μ_low · t) / (μ_high · t) = μ_low / μ_high
        
        The thickness t cancels out! This means R depends ONLY on
        material composition, not geometry.
        
        The spatial gradient ∇R:
            - Is zero inside homogeneous materials (constant R)
            - Is non-zero at material BOUNDARIES
        
        This creates a "material edge detector" that ignores geometric
        edges (like a book's spine) and highlights only compositional
        changes (like metal touching plastic).
        
        Args:
            L_low: Low energy log-attenuation (B, 1, H, W)
            L_high: High energy log-attenuation (B, 1, H, W)
            
        Returns:
            grad_R: Ratio gradient magnitude (B, 1, H, W)
        """
        # Compute ratio map
        R = L_low / (L_high + self.epsilon)
        
        # Compute spatial gradients using finite differences
        # Sobel-like kernels would be smoother, but simple differences are faster
        
        # Gradient in X direction (horizontal)
        # Pad on the right to maintain shape
        grad_x = torch.abs(R[:, :, :, 1:] - R[:, :, :, :-1])
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        
        # Gradient in Y direction (vertical)
        # Pad on the bottom to maintain shape
        grad_y = torch.abs(R[:, :, 1:, :] - R[:, :, :-1, :])
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        
        # Gradient magnitude (L1 norm for speed, L2 would be sqrt(gx² + gy²))
        grad_R = grad_x + grad_y
        
        return grad_R
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False,
        return_debug: bool = False
    ) -> torch.Tensor:
        """
        Forward pass: Dual-energy log-attenuation → Physics feature maps.
        
        Args:
            x: Input tensor (B, 2, H, W)
               Channel 0: L_low (low energy log-attenuation)
               Channel 1: L_high (high energy log-attenuation)
            return_intermediate: If True, also return polynomial stack (for debugging)
            return_debug: If True, return detailed debug dict with all intermediate states
            
        Returns:
            physics_maps: Output tensor (B, 4, H, W)
               Channel 0: A₁ (Photoelectric coefficient)
               Channel 1: A₂ (Compton coefficient)
               Channel 2: Z_eff (Effective atomic number)
               Channel 3: ∇R (Ratio gradient - material edge detector)
               
            If return_debug=True, also returns debug_dict containing:
                - 'L_low': Input low energy log-attenuation
                - 'L_high': Input high energy log-attenuation  
                - 'poly_stack': Polynomial feature stack
                - 'A1': Photoelectric coefficient (after ReLU)
                - 'A2': Compton coefficient (after ReLU)
                - 'Z_eff': Effective atomic number
                - 'grad_R': Ratio gradient
                - 'ratio_map': The R = L_low/L_high map (thickness-invariant)
        """
        # Split input channels
        L_low = x[:, 0:1, :, :]   # (B, 1, H, W)
        L_high = x[:, 1:2, :, :]  # (B, 1, H, W)
        
        # Step 1: Build polynomial feature stack
        poly_stack = self._build_polynomial_stack(L_low, L_high)
        
        # Step 2: Compute basis decomposition (A₁, A₂)
        A1, A2 = self._compute_basis_decomposition(poly_stack)
        
        # Step 3: Compute derived features
        Z_eff = self._compute_z_effective(A1, A2)
        grad_R = self._compute_ratio_gradient(L_low, L_high)
        
        # Concatenate all physics features
        physics_maps = torch.cat([A1, A2, Z_eff, grad_R], dim=1)
        
        if return_debug:
            # Compute the ratio map for debug visualization
            ratio_map = L_low / (L_high + self.epsilon)
            debug_dict = {
                'L_low': L_low,
                'L_high': L_high,
                'poly_stack': poly_stack,
                'A1': A1,
                'A2': A2,
                'Z_eff': Z_eff,
                'grad_R': grad_R,
                'ratio_map': ratio_map,
            }
            return physics_maps, debug_dict
        
        if return_intermediate:
            return physics_maps, poly_stack
        return physics_maps
    
    def get_coefficient_summary(self) -> dict:
        """
        Return a summary of the current polynomial coefficients.
        Useful for debugging and understanding what the model has learned.
        """
        term_names = ["const"]
        for degree in range(1, self.order + 1):
            for j in range(degree + 1):
                i = degree - j
                if i == 0:
                    term_names.append(f"H^{j}")
                elif j == 0:
                    term_names.append(f"L^{i}")
                else:
                    term_names.append(f"L^{i}*H^{j}")
        
        return {
            "term_names": term_names,
            "coeffs_a1": self.coeffs_a1.detach().cpu().numpy(),
            "coeffs_a2": self.coeffs_a2.detach().cpu().numpy(),
            "z_scale": self.z_scale.item() if isinstance(self.z_scale, torch.Tensor) else self.z_scale,
            "z_offset": self.z_offset.item() if isinstance(self.z_offset, torch.Tensor) else self.z_offset,
        }


# ==============================================================================
# UNIT TESTS / PHYSICS VALIDATION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PhysicsHead Unit Test")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Test 1: Shape Verification
    # -------------------------------------------------------------------------
    print("\n[Test 1] Shape Verification")
    
    B, H, W = 2, 64, 64
    test_input = torch.rand(B, 2, H, W) * 5  # Random log-attenuation in [0, 5]
    
    physics_head = PhysicsHead(order=2, learnable=True)
    output = physics_head(test_input)
    
    print(f"  Input shape:  {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (B, 4, H, W), f"Shape mismatch! Expected {(B, 4, H, W)}"
    print("  ✓ Shape test passed")
    
    # -------------------------------------------------------------------------
    # Test 2: Physics Sanity Check (Salt vs Sugar)
    # -------------------------------------------------------------------------
    print("\n[Test 2] Salt vs Sugar Physics Check")
    print("  Creating synthetic materials with known properties...")
    
    # Simulate two materials:
    # - Sugar (low Z): L_low ≈ L_high (ratio ≈ 1)
    # - Salt (high Z): L_low > L_high (ratio > 1, photoelectric enhanced at low E)
    
    H, W = 32, 32
    
    # Sugar region: low-Z, organic
    # At low-Z, Compton dominates at both energies, so L_low ≈ L_high
    L_low_sugar = torch.ones(1, 1, H, W) * 2.0
    L_high_sugar = torch.ones(1, 1, H, W) * 1.8  # Slightly less (small photoelectric)
    
    # Salt region: higher-Z, inorganic
    # Photoelectric enhanced at low energy, so L_low >> L_high
    L_low_salt = torch.ones(1, 1, H, W) * 3.5
    L_high_salt = torch.ones(1, 1, H, W) * 2.0  # Much less (strong photoelectric at low E)
    
    # Create test images
    sugar_input = torch.cat([L_low_sugar, L_high_sugar], dim=1)
    salt_input = torch.cat([L_low_salt, L_high_salt], dim=1)
    
    physics_head = PhysicsHead(order=2, learnable=False)
    
    sugar_out = physics_head(sugar_input)
    salt_out = physics_head(salt_input)
    
    # Extract Z_eff (channel 2)
    z_sugar = sugar_out[0, 2, 16, 16].item()
    z_salt = salt_out[0, 2, 16, 16].item()
    
    print(f"  Sugar Z_eff: {z_sugar:.2f}")
    print(f"  Salt Z_eff:  {z_salt:.2f}")
    print(f"  Ratio (salt/sugar): {z_salt/z_sugar:.2f}")
    
    # Salt should have higher Z_eff than sugar
    if z_salt > z_sugar:
        print("  ✓ Salt has higher Z_eff than sugar (physically correct!)")
    else:
        print("  ✗ WARNING: Salt should have higher Z_eff than sugar!")
    
    # -------------------------------------------------------------------------
    # Test 3: Gradient Check for Material Boundary
    # -------------------------------------------------------------------------
    print("\n[Test 3] Material Boundary Detection (Ratio Gradient)")
    
    # Create an image with two materials side by side
    H, W = 64, 64
    L_low = torch.ones(1, 1, H, W) * 2.0
    L_high = torch.ones(1, 1, H, W) * 1.8
    
    # Right half is different material (higher ratio)
    # Boundary is at x = W//2 = 32
    L_low[:, :, :, W//2:] = 3.5
    L_high[:, :, :, W//2:] = 2.0
    
    boundary_input = torch.cat([L_low, L_high], dim=1)
    boundary_out = physics_head(boundary_input)
    
    grad_R = boundary_out[0, 3]  # Channel 3 is ratio gradient
    
    # Check gradient at boundary vs interior
    # NOTE: The forward difference grad[x] = |R[x+1] - R[x]| means:
    #   - The boundary at x=32 produces gradient at x=31 (one pixel LEFT of boundary)
    #   - grad[31] = |R[32] - R[31]| = |1.75 - 1.111| = 0.639 (the actual edge)
    #   - grad[32] = |R[33] - R[32]| = |1.75 - 1.75| = 0 (already past the edge)
    interior_grad = grad_R[H//2, W//4].item()      # Left side interior (x=16)
    boundary_grad = grad_R[H//2, W//2 - 1].item()  # One pixel before boundary (x=31)
    
    print(f"  Interior gradient (x=16): {interior_grad:.4f}")
    print(f"  Boundary gradient (x=31): {boundary_grad:.4f}")
    
    if boundary_grad > 0.1 and boundary_grad > interior_grad * 5:
        print("  ✓ Boundary has significantly higher gradient (edge detection works!)")
    else:
        print(f"  ✗ WARNING: Boundary gradient should be much higher than interior")
        print(f"      Expected boundary_grad > 0.1, got {boundary_grad:.4f}")
    
    # -------------------------------------------------------------------------
    # Test 4: Non-negativity Constraint
    # -------------------------------------------------------------------------
    print("\n[Test 4] Non-negativity Constraint (Physical Realizability)")
    
    # Feed random data (some might try to produce negative coefficients)
    random_input = torch.randn(4, 2, 64, 64) * 3  # Can have negative values
    random_output = physics_head(F.relu(random_input))  # Clamp input to be valid
    
    A1 = random_output[:, 0]
    A2 = random_output[:, 1]
    
    if (A1 >= 0).all() and (A2 >= 0).all():
        print("  ✓ All A₁, A₂ values are non-negative (physically valid)")
    else:
        print("  ✗ WARNING: Negative attenuation coefficients detected!")
    
    # -------------------------------------------------------------------------
    # Test 5: Numerical Stability
    # -------------------------------------------------------------------------
    print("\n[Test 5] Numerical Stability")
    
    output = physics_head(test_input)
    
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()
    
    print(f"  NaN present: {has_nan}")
    print(f"  Inf present: {has_inf}")
    
    if not has_nan and not has_inf:
        print("  ✓ No NaN or Inf in output")
    else:
        print("  ✗ WARNING: Numerical instability detected!")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Coefficient Summary:")
    summary = physics_head.get_coefficient_summary()
    print(f"  Terms: {summary['term_names']}")
    print(f"  A₁ coeffs: {summary['coeffs_a1']}")
    print(f"  A₂ coeffs: {summary['coeffs_a2']}")
    print(f"  Z scale: {summary['z_scale']}, offset: {summary['z_offset']}")
    print("=" * 70)
    print("\n✓ All PhysicsHead tests completed!")
    
    # ==========================================================================
    # A-SPACE NOISE FILTER TESTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ASpaceNoiseFilter Unit Tests")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Test 6: Wedge Clamp - Physically Impossible Z_eff
    # -------------------------------------------------------------------------
    print("\n[Test 6] Wedge Clamp - Impossible Z_eff Detection")
    print("  THE PHYSICS: Points with θ outside [θ_min, θ_max] have impossible Z.")
    print("  Creating synthetic data with known outliers...")
    
    noise_filter = ASpaceNoiseFilter(
        theta_min=0.1,   # ~6° minimum angle
        theta_max=1.4,   # ~80° maximum angle
        k_neighbors=1,
        mad_multiplier=3.0
    )
    
    # Create A1, A2 with mostly valid points and some outliers
    H, W = 32, 32
    
    # Valid region: θ in [0.3, 1.2] - well within bounds
    # We'll create points along a line (same Z_eff, varying thickness)
    valid_theta = 0.6  # ~34°, middle of valid range
    valid_r = torch.linspace(0.5, 3.0, H * W).reshape(H, W)
    
    A1_valid = valid_r * np.sin(valid_theta)
    A2_valid = valid_r * np.cos(valid_theta)
    
    # Inject outliers: 10 pixels with impossible θ (too low, Z < minimum)
    outlier_indices = [(5, 5), (10, 10), (15, 15), (20, 20), (25, 25),
                       (5, 25), (10, 20), (15, 25), (20, 5), (25, 10)]
    
    A1_with_outliers = A1_valid.clone()
    A2_with_outliers = A2_valid.clone()
    
    # Set outliers to θ = 0.02 (way below θ_min = 0.1)
    # θ = atan2(A1, A2) = 0.02 → A1/A2 = tan(0.02) ≈ 0.02
    outlier_theta = 0.02
    outlier_r = 2.0
    for (i, j) in outlier_indices:
        A1_with_outliers[i, j] = outlier_r * np.sin(outlier_theta)
        A2_with_outliers[i, j] = outlier_r * np.cos(outlier_theta)
    
    # Format as batch tensors
    A1_batch = A1_with_outliers.unsqueeze(0).unsqueeze(0).float()
    A2_batch = A2_with_outliers.unsqueeze(0).unsqueeze(0).float()
    
    # Apply filter
    A1_filtered, A2_filtered, diagnostics = noise_filter.filter(
        A1_batch, A2_batch, return_diagnostics=True
    )
    
    print(f"  Injected outliers: {len(outlier_indices)}")
    print(f"  Wedge-clamped pixels: {diagnostics[0]['n_wedge_clamped']}")
    print(f"  k-NN outliers detected: {diagnostics[0]['n_outliers']}")
    print(f"  Outlier fraction: {diagnostics[0]['outlier_fraction']*100:.2f}%")
    
    # Verify the filter caught the outliers
    # After wedge clamp, all θ should be >= θ_min
    a1_np = A1_filtered[0, 0].numpy().flatten()
    a2_np = A2_filtered[0, 0].numpy().flatten()
    theta_filtered = np.arctan2(a1_np, a2_np)
    
    min_theta_after = theta_filtered.min()
    max_theta_after = theta_filtered.max()
    
    print(f"  θ range after filter: [{min_theta_after:.3f}, {max_theta_after:.3f}]")
    print(f"  Valid θ range: [{noise_filter.theta_min:.3f}, {noise_filter.theta_max:.3f}]")
    
    if min_theta_after >= noise_filter.theta_min - 0.01:  # Small tolerance
        print("  ✓ All impossible θ values were clamped (wedge clamp works!)")
    else:
        print(f"  ✗ WARNING: Some θ values still below θ_min!")
    
    # -------------------------------------------------------------------------
    # Test 7: k-NN Outlier Detection - Isolated Noise Spike
    # -------------------------------------------------------------------------
    print("\n[Test 7] k-NN Outlier Detection - Isolated Noise Spike")
    print("  THE PHYSICS: A pixel with no A-space neighbors is likely noise.")
    print("  Creating uniform material with one isolated spike...")
    
    H, W = 64, 64
    
    # Create uniform material: all pixels have similar (θ, r)
    # This represents a homogeneous region (e.g., a sheet of plastic)
    base_theta = 0.5
    base_r = 2.0
    noise_std = 0.01  # Small variation
    
    theta_uniform = base_theta + np.random.randn(H, W) * noise_std
    r_uniform = base_r + np.random.randn(H, W) * noise_std * 0.5
    
    A1_uniform = torch.from_numpy(r_uniform * np.sin(theta_uniform)).float()
    A2_uniform = torch.from_numpy(r_uniform * np.cos(theta_uniform)).float()
    
    # Inject ONE isolated outlier at center
    # This pixel has completely different (θ, r) - no neighbors in A-space
    spike_theta = 1.2  # Very different angle
    spike_r = 5.0      # Very different magnitude
    center_i, center_j = H // 2, W // 2
    
    A1_with_spike = A1_uniform.clone()
    A2_with_spike = A2_uniform.clone()
    A1_with_spike[center_i, center_j] = spike_r * np.sin(spike_theta)
    A2_with_spike[center_i, center_j] = spike_r * np.cos(spike_theta)
    
    # Format as batch
    A1_spike_batch = A1_with_spike.unsqueeze(0).unsqueeze(0)
    A2_spike_batch = A2_with_spike.unsqueeze(0).unsqueeze(0)
    
    # Apply filter
    A1_spike_filtered, A2_spike_filtered, spike_diagnostics = noise_filter.filter(
        A1_spike_batch, A2_spike_batch, return_diagnostics=True
    )
    
    print(f"  Median k-NN distance: {spike_diagnostics[0]['median_knn_distance']:.6f}")
    print(f"  MAD: {spike_diagnostics[0]['mad_knn_distance']:.6f}")
    print(f"  Threshold: {spike_diagnostics[0]['threshold']:.6f}")
    print(f"  Max k-NN distance: {spike_diagnostics[0]['max_knn_distance']:.6f}")
    print(f"  Outliers detected: {spike_diagnostics[0]['n_outliers']}")
    
    # Check if the spike was corrected
    original_spike_a1 = A1_with_spike[center_i, center_j].item()
    filtered_spike_a1 = A1_spike_filtered[0, 0, center_i, center_j].item()
    
    spike_changed = abs(original_spike_a1 - filtered_spike_a1) > 0.1
    
    if spike_changed:
        print(f"  ✓ Isolated spike was detected and corrected!")
        print(f"    Original A₁ at spike: {original_spike_a1:.3f}")
        print(f"    Filtered A₁ at spike: {filtered_spike_a1:.3f}")
    else:
        print(f"  ✗ WARNING: Spike was not detected as outlier")
    
    # -------------------------------------------------------------------------
    # Test 8: Legitimate Small Object Should NOT Be Filtered
    # -------------------------------------------------------------------------
    print("\n[Test 8] Small Cluster Preservation (Metal Rivet Test)")
    print("  THE PHYSICS: A 3x3 cluster of high-Z pixels is a real object,")
    print("               not noise. The filter should NOT remove it.")
    
    H, W = 64, 64
    
    # Create background of low-Z material (plastic/clothing)
    bg_theta = 0.3  # Low Z
    bg_r = 1.5
    
    A1_bg = torch.ones(H, W) * (bg_r * np.sin(bg_theta))
    A2_bg = torch.ones(H, W) * (bg_r * np.cos(bg_theta))
    
    # Add 3x3 cluster of high-Z material (metal rivet)
    rivet_theta = 1.1  # High Z
    rivet_r = 2.5
    rivet_center_i, rivet_center_j = 30, 30
    
    A1_with_rivet = A1_bg.clone()
    A2_with_rivet = A2_bg.clone()
    
    # 3x3 cluster
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            A1_with_rivet[rivet_center_i + di, rivet_center_j + dj] = rivet_r * np.sin(rivet_theta)
            A2_with_rivet[rivet_center_i + di, rivet_center_j + dj] = rivet_r * np.cos(rivet_theta)
    
    # Format as batch
    A1_rivet_batch = A1_with_rivet.unsqueeze(0).unsqueeze(0).float()
    A2_rivet_batch = A2_with_rivet.unsqueeze(0).unsqueeze(0).float()
    
    # Apply filter
    A1_rivet_filtered, A2_rivet_filtered, rivet_diagnostics = noise_filter.filter(
        A1_rivet_batch, A2_rivet_batch, return_diagnostics=True
    )
    
    # Check if rivet center was preserved
    original_rivet_a1 = A1_with_rivet[rivet_center_i, rivet_center_j].item()
    filtered_rivet_a1 = A1_rivet_filtered[0, 0, rivet_center_i, rivet_center_j].item()
    
    # Rivet should be mostly preserved (cluster has A-space neighbors)
    rivet_preserved = abs(original_rivet_a1 - filtered_rivet_a1) < 0.3
    
    print(f"  Rivet cluster size: 3x3 = 9 pixels")
    print(f"  Original rivet A₁: {original_rivet_a1:.3f}")
    print(f"  Filtered rivet A₁: {filtered_rivet_a1:.3f}")
    print(f"  Outliers detected: {rivet_diagnostics[0]['n_outliers']}")
    
    if rivet_preserved:
        print("  ✓ Metal rivet cluster was preserved (not falsely flagged as noise)")
    else:
        print("  ✗ WARNING: Rivet was incorrectly filtered out!")
    
    # -------------------------------------------------------------------------
    # Test 9: Filter Shape Preservation
    # -------------------------------------------------------------------------
    print("\n[Test 9] Output Shape Preservation")
    
    B, H, W = 3, 128, 128
    A1_test = torch.rand(B, 1, H, W) * 3
    A2_test = torch.rand(B, 1, H, W) * 3
    
    A1_out, A2_out = noise_filter.filter(A1_test, A2_test)
    
    print(f"  Input shape: A1={A1_test.shape}, A2={A2_test.shape}")
    print(f"  Output shape: A1={A1_out.shape}, A2={A2_out.shape}")
    
    if A1_out.shape == A1_test.shape and A2_out.shape == A2_test.shape:
        print("  ✓ Output shapes match input shapes")
    else:
        print("  ✗ WARNING: Shape mismatch!")
    
    # -------------------------------------------------------------------------
    # Test 10: Numerical Stability After Filtering
    # -------------------------------------------------------------------------
    print("\n[Test 10] Numerical Stability After Filtering")
    
    has_nan = torch.isnan(A1_out).any() or torch.isnan(A2_out).any()
    has_inf = torch.isinf(A1_out).any() or torch.isinf(A2_out).any()
    has_negative = (A1_out < 0).any() or (A2_out < 0).any()
    
    print(f"  NaN present: {has_nan}")
    print(f"  Inf present: {has_inf}")
    print(f"  Negative values: {has_negative}")
    
    if not has_nan and not has_inf:
        print("  ✓ No NaN or Inf in filtered output")
    else:
        print("  ✗ WARNING: Numerical instability detected!")
    
    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ASpaceNoiseFilter Configuration:")
    print(f"  {noise_filter}")
    print("=" * 70)
    print("\n✓ All ASpaceNoiseFilter tests completed!")
