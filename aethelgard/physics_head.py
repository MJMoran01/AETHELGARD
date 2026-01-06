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
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Forward pass: Dual-energy log-attenuation → Physics feature maps.
        
        Args:
            x: Input tensor (B, 2, H, W)
               Channel 0: L_low (low energy log-attenuation)
               Channel 1: L_high (high energy log-attenuation)
            return_intermediate: If True, also return polynomial stack (for debugging)
            
        Returns:
            physics_maps: Output tensor (B, 4, H, W)
               Channel 0: A₁ (Photoelectric coefficient)
               Channel 1: A₂ (Compton coefficient)
               Channel 2: Z_eff (Effective atomic number)
               Channel 3: ∇R (Ratio gradient - material edge detector)
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
