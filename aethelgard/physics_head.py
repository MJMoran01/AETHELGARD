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

import math

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
        epsilon: float = 1e-6,
        seed: Optional[int] = 0
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
            seed: Seed for the RNG used to subsample the k-NN reference
                  cloud when an image exceeds subsample_threshold pixels.
                  Defaults to 0 so a filter run is reproducible by default;
                  pass None to draw from OS entropy instead. The Generator
                  is created once per filter instance, so it is the
                  SEQUENCE of filter() calls from construction that is
                  reproducible, not any single call in isolation.
        """
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.k_neighbors = k_neighbors
        self.mad_multiplier = mad_multiplier
        self.subsample_threshold = subsample_threshold
        self.epsilon = epsilon
        self.seed = seed
        # Determinism is a correctness property here, not a convenience: an
        # unseeded subsample changes the median/MAD outlier threshold, so the
        # same input image produced different filtered A-maps on different
        # runs. Own the Generator rather than touching global numpy state.
        self._rng = np.random.default_rng(seed)
    
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
        1. Build a KD-tree over a reference cloud of (θ, r) points (all of
           them, or a bounded random subsample for very large images)
        2. For each point, find distance to its k-th nearest neighbor in
           that reference cloud, discounting a self-match if (and only if)
           the point is itself part of the reference cloud
        3. Compute median and MAD of these distances
        4. Flag points where distance > median + mad_multiplier * MAD
        5. Replace flagged points with the values of their nearest
           non-outlier point in that same reference cloud
        
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

        # Normalize theta and r to similar scales for distance computation
        # This prevents one dimension from dominating the distance metric
        theta_range = self.theta_max - self.theta_min
        r_max = np.percentile(r, 99) + self.epsilon  # Robust max

        theta_normalized = (theta - self.theta_min) / theta_range
        r_normalized = r / r_max

        # Stack into 2D points for k-NN
        points = np.column_stack([theta_normalized, r_normalized])

        theta_filtered = theta.copy()
        r_filtered = r.copy()

        # Build KD-tree for efficient nearest neighbor search.
        # For very large images we subsample the TREE, not the query set: every
        # pixel is still tested, but against a bounded reference cloud.
        if n_pixels > self.subsample_threshold:
            tree_idx = self._rng.choice(
                n_pixels, self.subsample_threshold, replace=False
            )
        else:
            tree_idx = np.arange(n_pixels)
        n_tree = len(tree_idx)

        # Degenerate case, made explicit per the Glass Box policy: with fewer
        # reference points than neighbours requested there is no meaningful
        # "distance to my k-th neighbour" and cKDTree returns inf, which would
        # poison the median/MAD threshold. An A-space cloud that small carries
        # no outlier information, so pass the data through untouched.
        if n_pixels < 2 or n_tree < self.k_neighbors + 1:
            stats = {
                "n_pixels": n_pixels,
                "n_outliers": 0,
                "outlier_fraction": 0.0,
                "median_knn_distance": 0.0,
                "mad_knn_distance": 0.0,
                "threshold": 0.0,
                "max_knn_distance": 0.0,
                "n_tree_points": n_tree,
                "subsampled": bool(n_tree < n_pixels),
                "n_donors": 0,
                "donor_fallback": False,
                "degenerate": True,
            }
            return theta_filtered, r_filtered, stats

        tree = cKDTree(points[tree_idx])

        # Query k+1 neighbours, then discount the self-match CONDITIONALLY.
        # A point that is in the reference cloud matches itself at distance 0,
        # so its k-th true neighbour is in column k. A point that is NOT in the
        # cloud has no self-match, so its k-th true neighbour is in column k-1.
        # Taking column k for everyone (the previous behaviour) skipped a real
        # neighbour for every out-of-cloud pixel, inflating its distance and
        # contaminating the median/MAD threshold on exactly the large images
        # that trigger subsampling.
        in_tree = np.zeros(n_pixels, dtype=bool)
        in_tree[tree_idx] = True

        distances, _ = tree.query(points, k=self.k_neighbors + 1)
        knn_distances = np.where(
            in_tree,
            distances[:, self.k_neighbors],
            distances[:, self.k_neighbors - 1],
        )

        # Compute robust statistics
        median_dist = np.median(knn_distances)
        mad = np.median(np.abs(knn_distances - median_dist))

        # Threshold: points with distance > median + multiplier * MAD are outliers
        # Add epsilon to MAD to handle degenerate cases (all same distance)
        threshold = median_dist + self.mad_multiplier * (mad + self.epsilon)

        outlier_mask = knn_distances > threshold
        n_outliers = int(np.sum(outlier_mask))
        n_donors = 0
        donor_fallback = False

        if n_outliers > 0:
            # Replacement donors come from the SAME (possibly subsampled)
            # reference cloud used for detection. Rebuilding a tree over every
            # valid pixel would undo the memory/time bound precisely on the
            # images that have outliers. When no subsampling occurred tree_idx
            # is every pixel, so this is identical to the previous behaviour.
            donor_idx = tree_idx[~outlier_mask[tree_idx]]

            # Pathological but reachable: EVERY reference-cloud point can be
            # flagged while non-outliers survive outside the cloud (an
            # out-of-cloud duplicate of a cloud point scores distance 0,
            # which a cloud point can never do once its self-match is
            # discounted). Silently skipping replacement there would leave
            # known-bad pixels in the output while the diagnostics still
            # reported outliers, so fall back to the full valid set - and
            # record that the bound was exceeded rather than hiding it.
            if len(donor_idx) == 0:
                donor_idx = np.flatnonzero(~outlier_mask)
                donor_fallback = bool(len(donor_idx) > 0)
            n_donors = int(len(donor_idx))

            if len(donor_idx) > 0:
                donor_tree = cKDTree(points[donor_idx])

                # For each outlier, find its nearest surviving donor
                _, nearest_donor = donor_tree.query(points[outlier_mask], k=1)

                # Replace outlier values with that donor's values
                theta_filtered[outlier_mask] = theta[donor_idx][nearest_donor]
                r_filtered[outlier_mask] = r[donor_idx][nearest_donor]

        # Compile statistics for diagnostics
        stats = {
            "n_pixels": n_pixels,
            "n_outliers": n_outliers,
            "outlier_fraction": n_outliers / n_pixels if n_pixels > 0 else 0,
            "median_knn_distance": median_dist,
            "mad_knn_distance": mad,
            "threshold": threshold,
            "max_knn_distance": np.max(knn_distances),
            "n_tree_points": n_tree,
            "subsampled": bool(n_tree < n_pixels),
            "n_donors": n_donors,
            "donor_fallback": donor_fallback,
            "degenerate": False,
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


# The module stores its scalars as float32 tensors, so "finite in Python"
# is not the property that matters: 1e39 is a perfectly finite Python float
# that becomes inf on conversion, and 1e-50 becomes exactly zero. Validate
# against the usable float32 range instead.
#
# _FLOAT32_SMALLEST_NORMAL is np.finfo(np.float32).tiny, the smallest
# positive NORMAL float32 - not the smallest representable one. Values below
# it (1e-40, say) are still representable as subnormals rather than becoming
# zero, so rejecting them is a deliberate choice, not a claim that they
# underflow: subnormals lose mantissa bits as they shrink, and several
# backends (and most GPU fast-math paths) flush them to zero anyway, so a
# guard constant living down there would silently mean something different
# per device. This module needs its epsilon and floor to mean the same thing
# everywhere, so the normal range is the supported range.
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_SMALLEST_NORMAL = float(np.finfo(np.float32).tiny)


def _check_float32_scalar(name: str, value: float, positive: bool = True) -> float:
    """
    Reject a configuration scalar that cannot survive the trip to float32.

    NaN is checked with math.isfinite rather than a bare comparison because
    NaN fails every ordering test: `if value <= 0` waves NaN straight
    through, and a NaN threshold or scale silently poisons every downstream
    map instead of raising here.
    """
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
    if abs(value) > _FLOAT32_MAX:
        raise ValueError(
            f"{name}={value} overflows float32 (limit {_FLOAT32_MAX:g})"
        )
    if value != 0.0 and abs(value) < _FLOAT32_SMALLEST_NORMAL:
        raise ValueError(
            f"{name}={value} is subnormal in float32 and is rejected as "
            "device-dependent (smallest supported magnitude "
            f"{_FLOAT32_SMALLEST_NORMAL:g}); it is representable, but see "
            "the note on the module constants"
        )
    return value


def _inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    """
    Inverse of softplus, for storing a strictly-positive parameter
    in unconstrained form.

    softplus(x) = log(1 + exp(x)); its algebraic inverse log(exp(y) - 1)
    overflows for moderate y, so we use the stable equivalent
    y + log(-expm1(-y)), valid for y > 0.
    """
    return y + torch.log(-torch.expm1(-y))


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
        z_cbrt_grad_max: float = 100.0,
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
            z_cbrt_grad_max: Largest permitted slope of the cube root used
                for Z_eff. d(r^(1/3))/dr diverges as r -> 0, so below a
                floor the cube root is replaced by a straight line whose
                slope is exactly this value; the floor is DERIVED from it
                as z_cbrt_grad_max^(-3/2) rather than being a second free
                constant (see _compute_z_effective). The default of 100
                puts that floor at 1e-3, i.e. the linear segment covers
                exactly the region below z_raw = 0.1, which is Z_eff = 1 at
                the module's DEFAULT z_scale: below hydrogen, the lowest
                element that exists, so at that scale the guard can only
                reshape a region that is already physically empty. That
                anchor is only as meaningful as the default scale it refers
                to, which is itself uncalibrated (issue #3) - hence this is
                an argued default, not a derived physical constant, and it
                is a constructor argument so recalibration can move it.
                The low-signal mask that goes with it introduces no new
                constant at all: it reuses `epsilon` (below).
            epsilon: Small constant for numerical stability (prevents division by zero)
        """
        super().__init__()
        
        self.order = order
        self.epsilon = epsilon
        _check_float32_scalar("epsilon", epsilon)
        _check_float32_scalar("z_cbrt_grad_max", z_cbrt_grad_max)
        self.z_cbrt_grad_max = z_cbrt_grad_max
        # Derived, not chosen: the linear segment through the origin and
        # (floor, floor^(1/3)) has slope floor^(-2/3); setting that equal to
        # z_cbrt_grad_max gives floor = z_cbrt_grad_max^(-3/2). Validated in
        # turn, because a huge (but finite) z_cbrt_grad_max underflows the
        # floor to zero, and 0 ** (-2/3) raises rather than clamping.
        self.z_ratio_floor = _check_float32_scalar(
            "z_ratio_floor (derived from z_cbrt_grad_max)",
            z_cbrt_grad_max ** -1.5,
        )
        
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
        
        # =====================================================================
        # Z_eff SCALING PARAMETERS
        # z_scale converts a dimensionless basis ratio into a reported
        # "effective atomic number". Z counts protons, so the map has to be
        # non-negative - but an unconstrained learnable scale lets an
        # optimizer walk it through zero and report negative atomic numbers,
        # violating the module's own physical interpretation. We therefore
        # store a RAW parameter and expose z_scale = softplus(raw), which is
        # strictly positive for every real value the optimizer can reach.
        #
        # This constrains the learnable parameter's RANGE only. Whether 10.0
        # is the physically correct value is a tier-3 calibration question
        # tracked in issue #3; the default is unchanged here (preserved to
        # float32 round-trip precision through inverse-softplus/softplus).
        # =====================================================================
        # Positive because Z_eff is non-negative; float32-checked because the
        # value is about to become a float32 tensor.
        _check_float32_scalar("z_scale", z_scale)
        _check_float32_scalar("z_offset", z_offset, positive=False)
        # Individually representable is not enough: Z_eff = z_scale * z_raw +
        # z_offset, so at the reference point z_raw = 1 (A1 == A2, an entirely
        # ordinary pixel) the SUM has to be representable too - two separately
        # legal float32 maxima overflow to inf together.
        #
        # This is a conservative SUFFICIENT bound, deliberately not a tight
        # one: |a| + |b| <= FLOAT32_MAX guarantees a*z_raw + b is
        # representable at z_raw = 1 for every sign combination, while some
        # rejected pairs (a huge scale against an equally huge NEGATIVE
        # offset) would in fact have cancelled to something finite. Rejecting
        # those is the intended trade: a configuration that only stays finite
        # by catastrophic cancellation is not one this module should accept
        # silently.
        #
        # It bounds the CONFIGURATION, which is what a constructor can bound.
        # It does not bound the data: a large enough A1/A2 overflows any
        # finite scale, and that is an input problem this module has no more
        # defense against than it does against an A1 of 1e30.
        if z_scale + abs(z_offset) > _FLOAT32_MAX:
            raise ValueError(
                f"z_scale ({z_scale}) + |z_offset| ({abs(z_offset)}) exceeds the "
                f"float32 limit {_FLOAT32_MAX:g}. This is a conservative bound: "
                "it guarantees Z_eff = z_scale * z_raw + z_offset stays "
                "representable at z_raw = 1 for either sign of the offset, and "
                "rejects pairs that would only remain finite by cancellation"
            )
        z_scale_raw = _inverse_softplus(torch.tensor(float(z_scale)))
        z_offset_tensor = torch.tensor(float(z_offset))
        
        if learnable:
            self.z_scale_raw = nn.Parameter(z_scale_raw)
            self.z_offset = nn.Parameter(z_offset_tensor)
        else:
            self.register_buffer("z_scale_raw", z_scale_raw)
            self.register_buffer("z_offset", z_offset_tensor)
    
    @property
    def z_scale(self) -> torch.Tensor:
        """
        Z_eff scale, positive by construction: softplus(z_scale_raw).
        
        Read-only on purpose - the constraint lives in the parameterization,
        so there is no way for training (or a caller) to set a negative
        scale. Construct the module with a different `z_scale=` to change it.
        """
        return F.softplus(self.z_scale_raw)
    
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
            Z_eff: Effective atomic number (B, 1, H, W), scaled for
                   interpretability, and exactly zero (offset included)
                   wherever A2 is numerically zero - no denominator, no Z.
        """
        # -----------------------------------------------------------------
        # THE TRAP: Z_eff is a RATIO feature, and a ratio means nothing where
        # there is no signal. In air, A2 (Compton, proportional to electron
        # density) is ~0, so A1/A2 is 0/0 noise. The previous code added an
        # epsilon to the denominator AND a second epsilon inside the cube root,
        # which bounds the value but not the noise: A1=0 with A2>0 reported
        # Z_eff = z_scale * eps^(1/3) = 0.1 rather than 0, and the cube root's
        # derivative near zero, (1/3)*eps^(-2/3) ~ 3.3e3 before scaling, turned
        # basis noise into enormous gradients. A1>0 with A2 -> 0 ran the other
        # way and reported an unboundedly large atomic number.
        #
        # THE FIX (AGENTS.md Glass Box - an explicit mask/clamp, not prose):
        #   (a) a low-signal MASK: where A2 is at or below the module's own
        #       stability epsilon the denominator is numerically zero, so
        #       the WHOLE output (offset included) is forced to 0 with no
        #       gradient.
        #
        #       Be precise about what this threshold is and is not. It is a
        #       numerical-degeneracy test, NOT a physical air/material
        #       discrimination - a pixel is masked because its denominator
        #       is unusable, not because it was identified as air. Any
        #       absolute threshold on A2 is still relative to whatever scale
        #       the (uncalibrated, learnable) coefficients put A2 on, and
        #       this one is no exception. It reuses exactly the epsilon the
        #       previous `A1 / (A2 + epsilon)` denominator already applied
        #       at this same point, so the scale dependence is INHERITED and
        #       now explicit, not newly introduced. Replacing it with a
        #       physically anchored threshold requires calibrating the
        #       coefficients first - a tier-3 question tracked in issue #3,
        #       and out of scope for the code-safety fix in issue #7.
        #   (b) a gradient CLAMP on the cube root: below z_ratio_floor the cube
        #       root is replaced by the straight line through the origin and
        #       (floor, floor^(1/3)) - identical value at the join, but a
        #       slope of exactly z_cbrt_grad_max instead of a divergent one.
        # -----------------------------------------------------------------
        low_signal = A2 <= self.epsilon

        # Divide by a safe denominator. The mask discards these entries anyway;
        # this keeps their (meaningless, huge) values out of the backward pass.
        A2_safe = torch.where(low_signal, torch.ones_like(A2), A2)
        ratio = A1 / A2_safe

        # Cube root to invert the Z^3 scaling. A1, A2 >= 0 via ReLU and
        # A2_safe > 0, so the ratio is non-negative and pow() is well defined.
        floor = self.z_ratio_floor
        # clamp() contributes zero gradient below the floor and torch.where
        # routes the gradient through the linear branch there, so neither
        # branch can emit the infinite derivative of d/dr r^(1/3) at r = 0.
        z_cube_root = torch.pow(ratio.clamp(min=floor), 1.0 / 3.0)
        z_linear = ratio * (floor ** (1.0 / 3.0 - 1.0))
        z_raw = torch.where(ratio < floor, z_linear, z_cube_root)

        # Low-signal pixels carry no Z information: value AND gradient zeroed.
        z_raw = torch.where(low_signal, torch.zeros_like(z_raw), z_raw)

        # Scale and offset for interpretability
        # Default scaling aims to put carbon ~6, iron ~26
        Z_eff = self.z_scale * z_raw + self.z_offset

        # The mask has to be applied AFTER the offset, not just to z_raw:
        # zeroing z_raw alone still leaves every masked pixel reporting
        # z_offset and still routes a gradient of 1 into z_offset from each
        # of them, so a large air region could dominate the offset's
        # training signal. "No denominator" has to mean no value AND no
        # gradient.
        Z_eff = torch.where(low_signal, torch.zeros_like(Z_eff), Z_eff)

        # Physical non-negativity: Z counts protons. z_scale is positive by
        # construction (softplus), but a learnable z_offset can still be driven
        # negative, so the reported map is clamped at zero.
        Z_eff = Z_eff.clamp(min=0.0)

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
        # ---------------------------------------------------------------------
        # TENSOR CONTRACT - enforced, not assumed. Slicing x[:, 0:1] / x[:, 1:2]
        # meant a (B, 3, H, W) input silently lost its third channel, and a
        # wrong-rank tensor failed somewhere deep in the polynomial stack. The
        # spatial minimum is a physical requirement, not a style choice: the
        # ratio gradient is a finite difference along each axis, so a 1-pixel
        # axis yields an empty difference map that F.pad(mode='replicate')
        # cannot pad.
        # ---------------------------------------------------------------------
        if x.dim() != 4:
            raise ValueError(
                "PhysicsHead expects a 4-D tensor (B, 2, H, W), got shape "
                f"{tuple(x.shape)}"
            )
        if x.shape[1] != 2:
            raise ValueError(
                "PhysicsHead expects exactly 2 input channels [L_low, L_high], "
                f"got {x.shape[1]} in shape {tuple(x.shape)}"
            )
        if x.shape[2] < 2 or x.shape[3] < 2:
            raise ValueError(
                "PhysicsHead requires H >= 2 and W >= 2 (the ratio gradient is a "
                f"finite difference along both axes), got shape {tuple(x.shape)}"
            )
        
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
        
        # .numpy() SHARES storage with a CPU tensor, so the arrays returned
        # here used to be live views of the model's parameters: writing to
        # summary["coeffs_a1"] mutated the model while bypassing autograd.
        # .copy() makes this a snapshot, which is all a summary should be.
        return {
            "term_names": term_names,
            "coeffs_a1": self.coeffs_a1.detach().cpu().numpy().copy(),
            "coeffs_a2": self.coeffs_a2.detach().cpu().numpy().copy(),
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
    
    # The semantic checks below are hard assertions on RNG-drawn data, so
    # the seeds are part of the test contract: without them a passing run
    # is not evidence that the next run passes.
    np.random.seed(0)
    torch.manual_seed(0)
    
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
    assert z_salt > z_sugar, (
        f"Salt Z_eff ({z_salt:.4f}) must exceed sugar Z_eff ({z_sugar:.4f}): "
        "the photoelectric term scales as Z^3, so the higher-Z material has "
        "to report the higher effective atomic number."
    )
    print("  ✓ Salt has higher Z_eff than sugar (physically correct!)")
    
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
    
    assert boundary_grad > 0.1 and boundary_grad > interior_grad * 5, (
        f"Compositional edge not detected: boundary gradient {boundary_grad:.4f} "
        f"vs interior {interior_grad:.4f}. R is thickness-invariant, so a "
        "material change must show up in grad(R) and a homogeneous interior "
        "must not."
    )
    print("  ✓ Boundary has significantly higher gradient (edge detection works!)")
    
    # -------------------------------------------------------------------------
    # Test 4: Non-negativity Constraint
    # -------------------------------------------------------------------------
    print("\n[Test 4] Non-negativity Constraint (Physical Realizability)")
    
    # Feed random data (some might try to produce negative coefficients)
    random_input = torch.randn(4, 2, 64, 64) * 3  # Can have negative values
    random_output = physics_head(F.relu(random_input))  # Clamp input to be valid
    
    A1 = random_output[:, 0]
    A2 = random_output[:, 1]
    
    assert (A1 >= 0).all() and (A2 >= 0).all(), (
        "Negative attenuation coefficients: mass attenuation cannot be "
        "negative, so the ReLU feasibility constraint has been violated."
    )
    print("  ✓ All A₁, A₂ values are non-negative (physically valid)")
    
    # -------------------------------------------------------------------------
    # Test 5: Numerical Stability
    # -------------------------------------------------------------------------
    print("\n[Test 5] Numerical Stability")
    
    output = physics_head(test_input)
    
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()
    
    print(f"  NaN present: {has_nan}")
    print(f"  Inf present: {has_inf}")
    
    assert not has_nan and not has_inf, (
        f"Numerical instability in PhysicsHead output (NaN={has_nan}, Inf={has_inf})"
    )
    print("  ✓ No NaN or Inf in output")
    
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
    
    assert min_theta_after >= noise_filter.theta_min - 0.01, (  # Small tolerance
        f"θ = {min_theta_after:.4f} survived the wedge clamp but is below "
        f"θ_min = {noise_filter.theta_min:.4f}: that is a physically "
        "impossible effective atomic number leaving the filter."
    )
    print("  ✓ All impossible θ values were clamped (wedge clamp works!)")
    
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
    
    assert spike_changed, (
        f"Isolated spike not corrected: A₁ went {original_spike_a1:.3f} -> "
        f"{filtered_spike_a1:.3f}. A pixel with no A-space neighbours is the "
        "filter's definition of noise; if it survives, pass 2 does nothing."
    )
    print(f"  ✓ Isolated spike was detected and corrected!")
    print(f"    Original A₁ at spike: {original_spike_a1:.3f}")
    print(f"    Filtered A₁ at spike: {filtered_spike_a1:.3f}")
    
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
    
    assert rivet_preserved, (
        f"Metal rivet was filtered out: A₁ went {original_rivet_a1:.3f} -> "
        f"{filtered_rivet_a1:.3f}. A 3x3 cluster of high-Z pixels is a real "
        "object with A-space neighbours, not an isolated noise spike."
    )
    print("  ✓ Metal rivet cluster was preserved (not falsely flagged as noise)")
    
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
    
    assert A1_out.shape == A1_test.shape and A2_out.shape == A2_test.shape, (
        f"Filter changed shape: {A1_test.shape} -> {A1_out.shape}"
    )
    print("  ✓ Output shapes match input shapes")
    
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
    
    assert not has_nan and not has_inf, (
        f"Numerical instability after filtering (NaN={has_nan}, Inf={has_inf})"
    )
    print("  ✓ No NaN or Inf in filtered output")
    
    # -------------------------------------------------------------------------
    # Test 11: Subsampled k-NN Path (determinism + out-of-cloud self-match)
    # -------------------------------------------------------------------------
    print("\n[Test 11] Subsampled k-NN Path")
    print("  Every test above sits below the 100k subsample threshold, so the")
    print("  subsampled branch (bounded reference cloud, conditional self-match,")
    print("  donor reuse) would otherwise never execute. Force it with a small")
    print("  threshold on a 64x64 image.")
    
    H, W = 64, 64  # 4096 pixels, well above the 500 forced below
    sub_rng = np.random.default_rng(1234)
    theta_sub = 0.5 + sub_rng.standard_normal((H, W)) * 0.01
    r_sub = 2.0 + sub_rng.standard_normal((H, W)) * 0.005
    
    # One isolated spike, deliberately placed so it is very unlikely to be
    # drawn into the 500-point reference cloud: it exercises the
    # out-of-cloud query path where the self-match column must NOT be skipped.
    theta_sub[10, 10] = 1.25
    r_sub[10, 10] = 4.5
    
    A1_sub = torch.from_numpy(r_sub * np.sin(theta_sub)).float().unsqueeze(0).unsqueeze(0)
    A2_sub = torch.from_numpy(r_sub * np.cos(theta_sub)).float().unsqueeze(0).unsqueeze(0)
    
    sub_filter_a = ASpaceNoiseFilter(
        theta_min=0.1, theta_max=1.4, subsample_threshold=500, seed=0
    )
    sub_filter_b = ASpaceNoiseFilter(
        theta_min=0.1, theta_max=1.4, subsample_threshold=500, seed=0
    )
    
    A1_sub_a, A2_sub_a, sub_diag_a = sub_filter_a.filter(
        A1_sub, A2_sub, return_diagnostics=True
    )
    A1_sub_b, A2_sub_b, _ = sub_filter_b.filter(
        A1_sub, A2_sub, return_diagnostics=True
    )
    
    print(f"  Reference cloud: {sub_diag_a[0]['n_tree_points']} of "
          f"{sub_diag_a[0]['n_pixels']} pixels")
    print(f"  Subsampled branch taken: {sub_diag_a[0]['subsampled']}")
    print(f"  Outliers detected: {sub_diag_a[0]['n_outliers']}")
    
    assert sub_diag_a[0]["subsampled"], "Test 11 did not exercise the subsampled branch"
    assert sub_diag_a[0]["n_tree_points"] == 500, (
        f"Reference cloud should be capped at 500, got {sub_diag_a[0]['n_tree_points']}"
    )
    assert sub_diag_a[0]["n_outliers"] >= 1, (
        "The injected isolated spike was not flagged on the subsampled path"
    )
    spike_before = A1_sub[0, 0, 10, 10].item()
    spike_after = A1_sub_a[0, 0, 10, 10].item()
    assert abs(spike_before - spike_after) > 0.1, (
        f"Spike survived the subsampled path: A₁ {spike_before:.3f} -> "
        f"{spike_after:.3f}. Detecting an outlier and then not replacing it "
        "is the donor-selection failure mode, not a pass."
    )
    assert torch.equal(A1_sub_a, A1_sub_b) and torch.equal(A2_sub_a, A2_sub_b), (
        "Two identically-seeded filters produced different output: the "
        "subsampling RNG is not deterministic"
    )
    assert not torch.isnan(A1_sub_a).any() and not torch.isinf(A1_sub_a).any(), (
        "NaN/Inf in the subsampled filter output"
    )
    print("  ✓ Subsampled path is deterministic and bounded")
    
    # -------------------------------------------------------------------------
    # Test 12: Conditional Self-Match (white-box regression for the k-NN bug)
    # -------------------------------------------------------------------------
    print("\n[Test 12] Conditional Self-Match on a Known Reference Cloud")
    print("  Test 11 exercises the subsampled branch but cannot PIN the bug:")
    print("  a spike that far from the cloud is flagged either way. This test")
    print("  fixes the reference cloud and asserts the k-NN statistics")
    print("  themselves, which differ between the fixed and buggy behaviour.")
    
    class _FixedCloudRNG:
        """
        Stand-in for np.random.Generator that returns a KNOWN reference
        cloud, so the expected median/MAD below can be derived by hand
        instead of depending on which points a real draw happened to pick.
        """
        
        def __init__(self, indices):
            self._indices = np.asarray(indices)
        
        def choice(self, a, size, replace=False):
            assert a == 8 and size == 4 and replace is False
            return self._indices
    
    # 8 pixels in one 2x4 image, all at the same radius so every A-space
    # distance is a pure θ distance. Row 1 is an EXACT duplicate of row 0.
    dup_theta_row = np.array([0.4, 0.6, 0.8, 1.0])
    dup_theta = np.stack([dup_theta_row, dup_theta_row])   # (2, 4)
    dup_r = np.full((2, 4), 2.0)
    
    A1_dup = torch.from_numpy(dup_r * np.sin(dup_theta)).float().unsqueeze(0).unsqueeze(0)
    A2_dup = torch.from_numpy(dup_r * np.cos(dup_theta)).float().unsqueeze(0).unsqueeze(0)
    
    dup_filter = ASpaceNoiseFilter(
        theta_min=0.1, theta_max=1.4, k_neighbors=1, subsample_threshold=4
    )
    # Reference cloud = row 0 (indices 0..3). Row 1 (indices 4..7) is
    # therefore queried against a tree it is NOT part of - the exact case
    # the old code got wrong.
    dup_filter._rng = _FixedCloudRNG([0, 1, 2, 3])
    
    _, _, dup_diag = dup_filter.filter(A1_dup, A2_dup, return_diagnostics=True)
    
    # Hand-derived expectation, APPROXIMATE by construction: the A-maps are
    # float32, `_cartesian_to_polar` adds epsilon before arctan2, and r is
    # reconstructed as sqrt(A1²+A2²), so both the 0.2 spacing and the shared
    # radius are recovered only to ~1e-6. What IS exact is the part the test
    # turns on: row 1 is computed from bit-identical float32 inputs to row 0,
    # so each duplicate pair is at distance exactly 0. The 1e-4 tolerance
    # below is ~100x the round-trip error and ~700x smaller than the s/2 vs s
    # gap it has to resolve.
    # Normalized spacing between adjacent θ:
    spacing = 0.2 / (1.4 - 0.1)
    # In-cloud points (0..3): column 0 is their own 0-distance self-match, so
    #   their 1st true neighbour is column 1 = spacing.
    # Out-of-cloud points (4..7): no self-match in the tree, and each is an
    #   exact duplicate of a cloud point, so column 0 = 0.
    # -> distances ≈ [s, s, s, s, 0, 0, 0, 0]; median ≈ s/2, MAD ≈ s/2.
    # Under the PRE-FIX behaviour every point took column 1, giving
    #   ≈[s]*8: median ≈ s, MAD ≈ 0 (not exactly 0, but ~1e-6). Both
    #   assertions below fail in that case by ~s/2 = 0.077, four orders of
    #   magnitude outside the tolerance - which is what makes this a
    #   regression test rather than a smoke test.
    print(f"  Reference cloud: {dup_diag[0]['n_tree_points']} of "
          f"{dup_diag[0]['n_pixels']} pixels")
    print(f"  Median k-NN distance: {dup_diag[0]['median_knn_distance']:.6f} "
          f"(expected {spacing / 2:.6f}, pre-fix {spacing:.6f})")
    print(f"  MAD: {dup_diag[0]['mad_knn_distance']:.6f} "
          f"(expected {spacing / 2:.6f}, pre-fix 0.000000)")
    
    assert dup_diag[0]["subsampled"] and dup_diag[0]["n_tree_points"] == 4
    assert abs(dup_diag[0]["median_knn_distance"] - spacing / 2) < 1e-4, (
        f"Median k-NN distance {dup_diag[0]['median_knn_distance']:.6f} != "
        f"{spacing / 2:.6f}: out-of-cloud points are not having their "
        "(non-existent) self-match handled correctly."
    )
    assert abs(dup_diag[0]["mad_knn_distance"] - spacing / 2) < 1e-4, (
        f"MAD {dup_diag[0]['mad_knn_distance']:.6f} != {spacing / 2:.6f}: "
        "a MAD of 0 here means every point was treated as in-cloud."
    )
    assert dup_diag[0]["n_outliers"] == 0, (
        "An exact duplicate of a reference point is the least isolated "
        f"pixel possible and must never be flagged; got "
        f"{dup_diag[0]['n_outliers']} outliers."
    )
    print("  ✓ Self-match is discounted only for points in the reference cloud")
    
    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ASpaceNoiseFilter Configuration:")
    print(f"  {noise_filter}")
    print("=" * 70)
    print("\n✓ All ASpaceNoiseFilter tests completed!")
