# Project AETHELGARD: Physics-Gated Attention Network for Dual-Energy X-Ray Security Imaging

**To:** Michael Moran, Senior AI Engineer (Candidate)
**From:** Dr. Thomas Anthony, CTO, Analytical AI
**Date:** December 29, 2025
**Subject:** Technical Architecture and Mathematical Derivations for Physics-Guided Deep Learning in X-Ray Security

---

## 1. Executive Summary and Architectural Vision

### 1.1 Project Mandate and Philosophy
Project AETHELGARD represents a strategic inflection point for Analytical AI. We are departing from the industry-standard methodology of treating neural networks as probabilistic "black boxes" that ingest raw pixels and output stochastic classifications. In the high-trust domains where we operate—aviation security, non-destructive testing (NDT), and threat detection—this purely data-driven approach has reached an asymptotic limit in performance. The "black box" failure modes, specifically the inability to distinguish between causal physical threats and textural correlates, create unacceptable operational friction and catastrophic false negative risks.

Our philosophy, rooted in the "Explainability of Deep Learning" and "High-Trust Image Based AI," demands a fundamental re-engineering of the computer vision stack. We are shifting to a **Physics-Gated** paradigm. Instead of asking a Convolutional Neural Network (CNN) to implicitly learn the laws of radiation physics from massive, messy datasets, we will explicitly encode these laws into the network architecture as differentiable, deterministic layers.

We will utilize the established principles of Dual-Energy X-ray Absorptiometry (DEXA)—specifically the Beer-Lambert attenuation model and the Alvarez-Macovski basis material decomposition—to preprocess, condition, and gate the information flow before it ever reaches the semantic interpretation layers.

You have been selected for this project specifically due to the hybrid nature of your background. Your academic foundation in Applied Mathematics and Scientific Computation from the University of Alabama at Birmingham, combined with your practical professional experience in signal processing at Lycoming Engines, makes you uniquely qualified to bridge the gap between deterministic physics and probabilistic learning. You are not merely building a classifier; you are engineering a differentiable physics engine that lives inside a neural network.

### 1.2 The Semantic Gap in Transmission Imaging
Standard computer vision models, such as the ResNet or YOLO architectures, are designed for reflection-based imagery (photography). In photography, pixel intensity correlates with surface reflectance and illumination. Objects occlude one another; a person standing behind a car is not visible.

In X-ray transmission imaging, however, the "image" is a map of line integrals. Pixel intensity represents the cumulative attenuation of every object along the ray path. This creates the **"superposition problem"**: a threat object (e.g., a firearm) located behind a benign dense object (e.g., a laptop battery) presents a radically different signal than a firearm behind a leather jacket. Standard CNNs struggle to disentangle these superimposed signals because they rely on texture and shape features that are corrupted by the superposition.

Current state-of-the-art models in security imaging often fail because they lack "material awareness." They cannot inherently distinguish between a thick block of plastic (low atomic number, high thickness) and a thin sheet of steel (high atomic number, low thickness) if the total attenuation is identical. This **"Intensity-Thickness Ambiguity"** is the primary source of false alarms in single-energy systems.

### 1.3 The AETHELGARD Concept
AETHELGARD (**A**ttentive **E**nergy-**T**ransmission **H**euristic employing **L**inear **G**radients and **A**tomic **R**econstruction **D**ecomposition) addresses this by introducing a dual-stream architecture that fuses signal processing with deep learning.

* **Stream A (The PhysicsHead):** A hard-coded (or semi-learnable) differentiable signal processing module. This stream performs an analytical inversion of the dual-energy X-ray attenuation data to recover the intrinsic material properties of the scene: Effective Atomic Number ($Z_{eff}$) and Electron Density ($\rho_e$). This layer does not "learn" in the traditional sense; it solves a system of non-linear integral equations based on the Alvarez-Macovski theorem.
* **Stream B (The VisionHead):** A standard deep convolutional neural network (e.g., ResNet-50 or EfficientNet-B4 backbone) responsible for hierarchical feature extraction, shape recognition, and semantic classification.
* **The Gating Mechanism:** The critical innovation is that the *PhysicsHead* gates the *VisionHead*. We utilize the derived material maps to generate an attention mechanism that suppresses visual features in regions that are physically irrelevant (e.g., low-density organic clutter like clothing) and amplifies features in regions of high probability for threats (e.g., high-Z metallic anomalies or specific explosive densities).

This document serves as the comprehensive guide for the AI Coding Agent (Cline) to implement this architecture. It details the rigorous mathematical derivations you requested to verify your "Salt & Sugar" and "Gradient Hypothesis" models, along with the specific PyTorch class structures, initialization strategies, and failure mode analyses required for a production-grade system.

---

## 2. Theoretical Framework: The Physics of Dual-Energy X-Ray

To build the PhysicsHead, we must first rigorously define the input signal. We cannot treat pixel values as arbitrary 0-255 integers; they are physical measurements of photon fluence that adhere to the laws of quantum mechanics and statistical probability.

### 2.1 The Polychromatic Beer-Lambert Law
In a monochromatic idealization, the transmission of X-rays $I$ through an object of thickness $t$ and linear attenuation coefficient $\mu$ is given by the Beer-Lambert law: $I = I_0 e^{-\mu t}$. However, real-world X-ray sources used in security scanners (Bremsstrahlung sources) are polychromatic. They emit a continuous spectrum of photon energies $S(E)$. Consequently, the measured intensity $I$ is the integral of the transmission over the entire energy spectrum:

$$
I = \int_{0}^{E_{max}} S(E) D(E) e^{-\int_{Ray} \mu(x,y,z,E) ds} dE
$$

Where:
* $S(E)$ is the source photon fluence spectrum (photons per $mm^2$ per keV).
* $D(E)$ is the detector spectral sensitivity (absorption efficiency).
* $\mu(x,y,z,E)$ is the linear attenuation coefficient at spatial point $(x,y,z)$ and energy $E$.

In Dual-Energy X-ray (DECT/DEXA), we acquire two distinct measurements for the same ray path, $I_L$ (Low Energy) and $I_H$ (High Energy). The log-attenuation (or "projection value") $L$ is defined as the negative natural logarithm of the transmission ratio:

$$
L = -\ln\left(\frac{I}{I_0}\right) = -\ln\left( \frac{\int S(E) e^{-\int \mu(E) ds} dE}{\int S(E) dE} \right)
$$

This equation is non-linear with respect to the object thickness due to the polychromatic nature of $S(E)$. As the beam passes through the object, lower energy photons are preferentially absorbed (**Beam Hardening**), shifting the effective energy of the spectrum higher. This non-linearity is a critical challenge that our PhysicsHead must model and correct.

### 2.2 Basis Material Decomposition (The Alvarez-Macovski Theorem)
The foundation of our PhysicsHead is the Alvarez-Macovski decomposition theorem. In 1976, Alvarez and Macovski proved that in the diagnostic energy range (30-200 keV), the linear attenuation coefficient $\mu(E)$ of any biological or security-relevant material can be approximated with high accuracy as a linear combination of two basis functions corresponding to the two dominant physical interactions: **Photoelectric Absorption** and **Compton Scattering**.

$$
\mu(r, E) \approx a_1(r) f_1(E) + a_2(r) f_2(E)
$$

Where:
* $a_1(r)$ (**Photoelectric Coefficient**): Represents the spatial density of the photoelectric effect. This coefficient is strongly dependent on the atomic number ($Z$) and physical density ($\rho$), scaling approximately as $\rho Z^4$ or $\rho Z^3$ depending on the energy range.
* $f_1(E)$ (**Photoelectric Basis Function**): The energy dependence of the photoelectric effect, often approximated as $1/E^3$.
* $a_2(r)$ (**Compton Coefficient**): Represents the spatial density of Compton scattering (incoherent scattering). This coefficient is dependent on the electron density ($\rho_e$), which is roughly proportional to the mass density $\rho$.
* $f_2(E)$ (**Compton Basis Function**): The energy dependence of Compton scattering, modeled by the Klein-Nishina cross-section $f_{KN}(E)$.

**The Decomposition Problem:** We aim to recover the line integrals of the material coefficients, $A_1 = \int a_1(r) ds$ and $A_2 = \int a_2(r) ds$. Substituting the basis decomposition into the transmission equation, we get:

$$
L_k(A_1, A_2) = -\ln \left( \frac{\int S_k(E) e^{-(A_1 f_1(E) + A_2 f_2(E))} dE}{\int S_k(E) dE} \right) \quad \text{for } k \in \{L, H\}
$$

This creates a system of two non-linear integral equations with two unknowns ($A_1, A_2$). The PhysicsHead layer must invert this system: map $(L_L, L_H) \rightarrow (A_1, A_2)$.

### 2.3 The "Salt & Sugar" Problem: Solving Intensity Ambiguity
You referenced "Salt & Sugar" in your resume/query. This analogy perfectly captures the fundamental limitation of single-energy X-ray imaging.

* **The Ambiguity:** Sugar (Sucrose, $C_{12}H_{22}O_{11}$) has a low effective atomic number ($Z_{eff} \approx 6.8$) and low density. Salt (Sodium Chloride, $NaCl$) has a higher effective atomic number ($Z_{eff} \approx 14$) and higher density. However, a **thick block of sugar** can produce the **exact same** total photon attenuation as a **thin sheet of salt**. In a single-energy greyscale image, these two objects are indistinguishable intensity values.

* **The Solution (Vector Logic):** In the 2D basis space $(A_1, A_2)$, often called the $\alpha$-$\beta$ plane or the Photoelectric-Compton plane, materials separate by vector angle.
    * **Vector Magnitude** ($\|A\| = \sqrt{A_1^2 + A_2^2}$): Correlates with the mass thickness of the object.
    * **Vector Angle** ($\theta = \arctan(A_1/A_2)$): Correlates with the effective atomic number $Z_{eff}$ (what the material *is*).

Since $A_1 \propto Z^3$ and $A_2 \propto \rho$, the ratio $A_1/A_2$ effectively isolates $Z$:

$$
\frac{A_1}{A_2} \propto \frac{\rho Z^3}{\rho} \propto Z^3
$$

Sugar (Organic, low Z) will have a vector lying close to the Compton axis ($A_2$), resulting in a small angle. Salt (Inorganic, higher Z) will have a significant Photoelectric component, pulling the vector away from the axis and resulting in a steeper angle. This vector logic is robust against thickness variations.

### 2.4 The Gradient Hypothesis: Thickness Invariance
You proposed a "Gradient Hypothesis" for edge detection. Let us verify this mathematically. We define the Dual-Energy Ratio $R$ as:

$$
R = \frac{L_L}{L_H}
$$

In the mono-energetic limit, $L_L = \mu_L t$ and $L_H = \mu_H t$. Substituting these into the ratio:

$$
R_{mono} = \frac{\mu_L t}{\mu_H t} = \frac{\mu_L}{\mu_H}
$$

The thickness term $t$ cancels out. $R_{mono}$ depends **only** on the material properties, not on how thick the object is. Now, consider the spatial gradient of this ratio map, $\nabla R$:

$$
\nabla R = \nabla \left( \frac{\mu_L}{\mu_H} \right)
$$

For a homogeneous object (constant material), the ratio is constant. Therefore:

$$
\nabla R = 0 \quad \text{(Inside a homogeneous material)}
$$

However, at the boundary between two different materials, the ratio changes abruptly. Thus:

$$
\nabla R \neq 0 \quad \text{(At material interfaces)}
$$

**Conclusion:** The gradient of the ratio map $\nabla R$ acts as a **Material Edge Detector**. It suppresses geometric edges (changes in thickness) and highlights only compositional changes.

### 2.5 Feasible Region Constraints in A-Space
The decomposition is physically constrained:
1.  **Constraint 1 (Non-Negativity):** $A_1 \ge 0$ and $A_2 \ge 0$.
2.  **Constraint 2 (The Feasible Triangle):** Real materials do not fill the entire positive quadrant. They are bounded by the air vector (0,0) and the vectors of limiting materials (e.g., Polyethylene and Lead).

**Failure Mode:** Poisson noise can push pixel values outside this feasible region. The PhysicsHead must implement **Clamp** or **ReLU** activations to enforce physical realizability, or use a "Feasible Region Projection" layer.

---

## 3. Mathematical Derivations for the Agent

### 3.1 Derivation of the Inverse Mapping (Polynomial Approximation)
Analytical inversion of the integral equations is computationally expensive. For the PhysicsHead, we will use a **polynomial approximation** of the inverse function. This allows for extremely fast, matrix-based computation on GPUs.

We approximate the basis integrals $A_1$ and $A_2$ as power series of the measured log-attenuations $L_L$ and $L_H$:

$$
A_1(L_L, L_H) \approx \sum_{i=0}^{N} \sum_{j=0}^{N-i} c_{ij}^{(1)} L_L^i L_H^j
$$

$$
A_2(L_L, L_H) \approx \sum_{i=0}^{N} \sum_{j=0}^{N-i} c_{ij}^{(2)} L_L^i L_H^j
$$

For a second-order approximation ($N=2$), the expansion terms are:

$$
\mathcal{T} = \{1, L_L, L_H, L_L^2, L_L L_H, L_H^2\}
$$

The equations become:

$$
A_1 = c_{00}^{(1)} + c_{10}^{(1)}L_L + c_{01}^{(1)}L_H + c_{20}^{(1)}L_L^2 + c_{11}^{(1)}L_L L_H + c_{02}^{(1)}L_H^2
$$

$$
A_2 = c_{00}^{(2)} + c_{10}^{(2)}L_L + c_{01}^{(2)}L_H + c_{20}^{(2)}L_L^2 + c_{11}^{(2)}L_L L_H + c_{02}^{(2)}L_H^2
$$

The coefficients $c_{ij}$ can be initialized from phantom data but then **fine-tuned via backpropagation**, allowing the network to "self-calibrate".

### 3.2 Jacobian Invertibility Conditions
For the decomposition to be stable, the Jacobian determinant of the transformation must be non-zero. Let the forward mapping be $L = F(A)$. The Jacobian matrix $J$ is:

$$
J = \begin{bmatrix} \frac{\partial L_L}{\partial A_1} & \frac{\partial L_L}{\partial A_2} \\ \frac{\partial L_H}{\partial A_1} & \frac{\partial L_H}{\partial A_2} \end{bmatrix}
$$

The partial derivatives are expectations of the basis functions over the transmitted spectrum. So the Jacobian is:

$$
J = \begin{bmatrix} \langle f_1 \rangle_L & \langle f_2 \rangle_L \\ \langle f_1 \rangle_H & \langle f_2 \rangle_H \end{bmatrix}
$$

Invertibility requires $\det(J) \neq 0$. This implies that the effective energy of the Low beam must be sufficiently different from the High beam. If the object is too thick (photon starvation), the transmitted spectra $S_L$ and $S_H$ both harden towards the same maximum energy, and the decomposition becomes singular.

---

## 4. System Architecture: The AethelgardNet

### 4.1 Class Hierarchy and Data Flow
We will define three primary classes:
* **PhysicsHead:** The differentiable decomposition layer that outputs physical maps ($A_1, A_2, Z_{eff}, \rho_e$).
* **VisionHead:** The semantic feature extractor (e.g., a truncated ResNet).
* **AethelgardNet:** The container fusing both streams using the PhysicsAttention module.

**Data Flow:**
* Input (B, 2, H, W) $\rightarrow$ PhysicsHead $\rightarrow$ Physics Maps (B, 4, H, W)
* Input (B, 2, H, W) $\rightarrow$ VisionHead $\rightarrow$ Visual Features (B, C, H', W')
* Physics Maps $\rightarrow$ Downsample $\rightarrow$ PhysicsAttention $\rightarrow$ Attention Mask
* Visual Features $\times$ Attention Mask $\rightarrow$ Weighted Features $\rightarrow$ Classifier/Detector

### 4.2 Implementation: The PhysicsHead
This module implements the polynomial decomposition derived in 3.1.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsHead(nn.Module):
    """
    Differentiable Physics Layer for Alvarez-Macovski Decomposition.
    Maps Dual-Energy Log-Attenuation -> Basis Material Maps.
    """
    def __init__(self, order=2, init_calibration=None):
        super().__init__()
        self.order = order
        # Number of terms for 2nd order: 1, L, H, L^2, LH, H^2 = 6 terms
        self.num_terms = (order + 1) * (order + 2) // 2
        
        # Learnable Coefficients for A1 (Photoelectric) and A2 (Compton)
        # Shape: [1, num_terms, 1, 1] for broadcasting across image spatial dims
        self.coeffs_a1 = nn.Parameter(torch.zeros(1, self.num_terms, 1, 1))
        self.coeffs_a2 = nn.Parameter(torch.zeros(1, self.num_terms, 1, 1))
        
        # Initialize with provided calibration or identity-like heuristic
        if init_calibration:
            self._load_calibration(init_calibration)
        else:
            self._init_heuristic()

    def _init_heuristic(self):
        # Initialize to a reasonable starting point to avoid local minima.
        # A1 (Photoelectric) is strongly correlated with L_low - L_high difference.
        # A2 (Compton) is correlated with L_high.
        # Example: Simple subtraction initialization
        nn.init.normal_(self.coeffs_a1, mean=0.0, std=0.01)
        nn.init.normal_(self.coeffs_a2, mean=0.0, std=0.01)
        
        # Set linear terms to approximate standard decomposition
        # (Placeholder values - typically derived from Phantom data)
        with torch.no_grad():
            self.coeffs_a1[0, 1, 0, 0] = 1.0   # +1 * L_low
            self.coeffs_a1[0, 2, 0, 0] = -0.5  # -0.5 * L_high
            self.coeffs_a2[0, 2, 0, 0] = 1.0   # +1 * L_high

    def forward(self, x):
        # x shape: -> Channels: [Low_Energy, High_Energy]
        # Assuming input x is already Log-Attenuation L
        L_low = x[:, 0:1, :, :]
        L_high = x[:, 1:2, :, :]
        
        # 1. Polynomial Feature Expansion
        # Terms: [1, L, H, L^2, LH, H^2]
        poly_terms = [torch.ones_like(L_low)] # bias term (c_00)
        
        for i in range(1, self.order + 1):
            for j in range(i + 1):
                # Power terms L^(i-j) * H^j
                term = (L_low ** (i - j)) * (L_high ** j)
                poly_terms.append(term)
        
        # Stack terms along channel dim:
        poly_stack = torch.cat(poly_terms, dim=1)
        
        # 2. Basis Decomposition (1x1 Convolution equivalent)
        # Apply learnable matrix multiplication per pixel via broadcasting
        A1 = torch.sum(poly_stack * self.coeffs_a1, dim=1, keepdim=True)
        A2 = torch.sum(poly_stack * self.coeffs_a2, dim=1, keepdim=True)
        
        # 3. Physical Constraints (ReLU)
        # Mass/Attenuation cannot be negative. Enforce feasible region quadrant 1.
        A1 = F.relu(A1)
        A2 = F.relu(A2)
        
        # 4. Physics Feature Derivation
        # Z_eff approximation (simplified power law)
        epsilon = 1e-6
        # Ratio of Photoelectric to Compton relates to Z^3
        # Z_eff ~ (A1 / A2)^(1/3)
        ratio = A1 / (A2 + epsilon)
        Z_eff = torch.pow(ratio, 1/3.0)
        
        # Ratio Gradient Map (Thickness Invariant Edge Detector)
        # Gradient of R = L_low / L_high
        R_map = L_low / (L_high + epsilon)
        
        # Gradient X
        grad_x = torch.abs(R_map[:, :, :, 1:] - R_map[:, :, :, :-1])
        grad_x = F.pad(grad_x, (0, 1, 0, 0))  # Padding to maintain shape
        
        # Gradient Y
        grad_y = torch.abs(R_map[:, :, 1:, :] - R_map[:, :, :-1, :])
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        
        ratio_grad = grad_x + grad_y
        
        # Return composite physics tensor
        return torch.cat([A1, A2, Z_eff, ratio_grad], dim=1)
```
### 4.3 Implementation: The PhysicsAttention Gating
This module forces the Vision features to respect the Physics findings.

```python
class PhysicsAttention(nn.Module):
    def __init__(self, vis_dim, phys_dim=4):
        super().__init__()
        # Map physics features to spatial attention mask
        # We use a small CNN to learn WHICH physics features matter for the current task
        self.attn_net = nn.Sequential(
            nn.Conv2d(phys_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # Output range for gating
        )

    def forward(self, vision_feats, physics_feats):
        # Resize physics feats to match vision resolution (downsampling)
        # Vision features are typically smaller (e.g., 1/4 or 1/8 resolution)
        if vision_feats.shape[-2:] != physics_feats.shape[-2:]:
            physics_feats_resized = F.interpolate(
                physics_feats,
                size=vision_feats.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        else:
            physics_feats_resized = physics_feats
        
        attn_map = self.attn_net(physics_feats_resized)
        
        # Gating Mechanism:
        # Residual Gating: vision_feats * (1 + attn_map)
        # This allows the network to preserve original features while boosting attended ones.
        return vision_feats * (1 + attn_map)
```
### 4.4 Global Network Assembly (AethelgardNet)

```python
class AethelgardNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.physics_head = PhysicsHead(order=2)
        
        # Standard Vision Backbone (e.g., ResNet18 truncated)
        # Assuming input is 2-channel (L_low, L_high)
        # Note: Standard ResNet expects 3 channels (RGB). We modify the first conv.
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # Modify input layer for 2 channels
        resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # We split ResNet into encoder stages to inject attention at different scales
        self.vision_stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Attention Gates for different scales
        self.att1 = PhysicsAttention(vis_dim=64, phys_dim=4)
        self.att2 = PhysicsAttention(vis_dim=128, phys_dim=4)
        self.att3 = PhysicsAttention(vis_dim=256, phys_dim=4)
        self.att4 = PhysicsAttention(vis_dim=512, phys_dim=4)
        
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # 1. Physics Stream
        phys_maps = self.physics_head(x)
        
        # 2. Vision Stream with Gating
        x_vis = self.vision_stem(x)
        
        x_vis = self.layer1(x_vis)
        x_vis = self.att1(x_vis, phys_maps)  # Gate Layer 1
        
        x_vis = self.layer2(x_vis)
        x_vis = self.att2(x_vis, phys_maps)  # Gate Layer 2
        
        x_vis = self.layer3(x_vis)
        x_vis = self.att3(x_vis, phys_maps)  # Gate Layer 3
        
        x_vis = self.layer4(x_vis)
        x_vis = self.att4(x_vis, phys_maps)  # Gate Layer 4
        
        x_vis = self.avgpool(x_vis)
        x_vis = torch.flatten(x_vis, 1)
        logits = self.fc(x_vis)
        
        return logits
```
## 5. Failure Mode Critiques & Risk Mitigation
As technical leaders, we must rigorously analyze where our architecture might fail.

### 5.1 The Monochromatic Assumption vs. Beam Hardening
**Critique:** The polynomial decomposition derived in 3.1 implicitly assumes that the beam hardening behavior is uniform or easily correctable by a global polynomial. However, the degree of hardening depends on the composition of the material it passes through. A 10cm block of water hardens the beam differently than a 10cm block of aluminum, even if they produce similar attenuation. The effective energy $E_{eff}$ shifts dynamically as the beam traverses the object.

**Failure Mode:** "Cupping" artifacts, where the center of a dense homogeneous object appears to have a lower $Z_{eff}$ than the edges.

**Mitigation:**
* **High-Order Polynomials:** Increasing the order of the PhysicsHead polynomial to 3 or 4 can capture more complex hardening curvature, though at the risk of overfitting noise.
* **Iterative Hardening Correction (Deep Unrolling):** Instead of a single-pass polynomial, we can implement the PhysicsHead as an **unrolled iterative solver**. We run $K$ iterations of a correction algorithm (e.g., Water Correction) as network layers. This allows the network to refine the effective path lengths iteratively.

### 5.2 The K-Edge Discontinuity
**Critique:** The Alvarez-Macovski model ($\mu = a_1 E^{-3} + a_2 f_{KN}$) assumes smooth energy dependence. It fails near K-absorption edges where attenuation spikes discontinuously. Common materials in security (Tin, Iodine filters, Lead shielding, some special nuclear materials) have K-edges in the diagnostic range.

**Failure Mode:** The decomposition will produce erroneous $A_1/A_2$ values near these edges. The model might interpret a K-edge jump as a sudden increase in density or atomic number that doesn't align with the smooth basis functions.

**Mitigation:** This is an inherent limitation of Dual-Energy (2 measurements). To rigorously fix this, we would need Multi-Energy (Photon Counting) data to solve for a 3rd basis function ($f_3$ for K-edge). Within the constraints of AETHELGARD (Dual-Energy), we must treat K-edge materials as noise or potential outliers. Alternatively, if we know specific K-edge materials are threats (e.g., Lead), we can train the VisionHead to recognize the specific "spectral signature" of the K-edge artifact.

### 5.3 Geometric Misalignment
**Critique:** If the Low and High energy images are not perfectly registered (e.g., patient/baggage motion between pulses in a pulsed source system), the pixel-wise operations in PhysicsHead (specifically the ratio $L_L/L_H$ and its gradient) will produce massive edge artifacts.

**Failure Mode:** The `ratio_grad` feature will trigger falsely on alignment ghosts, creating strong "edges" that are purely motion artifacts.

**Mitigation:** The preprocessing pipeline must include a **Differentiable Affine Alignment module (Spatial Transformer Network - STN)** before the PhysicsHead. This sub-network would learn to warp $L_H$ to match $L_L$ by minimizing the correlation error, ensuring pixel-perfect registration before decomposition.

---

## 6. Addendum: Real-World Adjustments 

### 6.1 Real-World Beam Hardening Correction (BHC)
In a real deployment, we cannot rely solely on the polynomial fit if the spectrum is unknown or drifting. We should implement a **Water Correction** pre-processing step. 

* **Look-up Table (LUT):** We can replace the polynomial PhysicsHead with a Differentiable Look-Up Table. The network learns the control points of a 2D surface $f(L_L, L_H) \rightarrow A_1$. This allows it to capture the complex curvature caused by beam hardening more flexibly than a polynomial.
* **Deep Beam Hardening:** Train a separate U-Net to predict "monochromatic" projections from the polychromatic raw data, trained on simulated data (Geant4/MCNP) where ground truth monochromatic projections are available.

### 6.2 Phantom-less Calibration (Self-Calibration)
You asked about calibration. In a security lane, we cannot scan a calibration step-wedge (phantom) every hour to update the decomposition matrix. We must use **Self-Calibration**.

* **Strategy:** Identify regions of "known" materials in the operational stream.
    * **Air:** $L_L \approx 0, L_H \approx 0$.
    * **Conveyor Belt:** The belt is a constant material (usually rubber/polymer) with a known $Z_{eff}$ and approximate thickness.
* **Implementation:** We can add a "Calibration Loss" term to the training. The network is penalized if the predicted $Z_{eff}$ of the background pixels (belt) deviates from the known $Z_{eff}$ of rubber. This allows the PhysicsHead coefficients to auto-tune to drift in the X-ray tube output over time.

---

## 7. Conclusion

Project AETHELGARD effectively leverages your background in signal processing to solve a domain-specific problem in Deep Learning. By explicitly modeling the physics of X-ray attenuation, we transform the problem from "learning to see patterns in noise" to "learning to interpret physical measurements".

The PhysicsHead acts as a **feature engineer** that never sleeps, providing the VisionHead with high-level, physically meaningful maps ($Z_{eff}$, Density) rather than raw, ambiguous attenuation values. This architecture is robust, explainable, and aligned with the high-trust requirements of our clients.

**Action Items for Agent Cline:**
1.  Implement `PhysicsHead` with learnable polynomial coefficients as defined in Section 4.2.
2.  Implement `PhysicsAttention` gating mechanism as defined in Section 4.3.
3.  Construct a synthetic "Salt & Sugar" dataset (using simple geometric shapes with assigned $\mu$ values) to unit test the decomposition logic before training on real data.

**Dr. Thomas Anthony**
CTO, Analytical AI

---

**Table 1: Basis Functions & Physical Dependencies** 

| Basis Function | Physical Dependency | Energy Dependency | Dominant Regime | Material Association |
| :--- | :--- | :--- | :--- | :--- |
| Photoelectric ($f_1$) | Atomic Number ($Z^3$ to $Z^4$) | $1/E^3$ | Low Energy (< 50 keV) | Metals, Bone, Iodine, Salt |
| Compton ($f_2$) | Electron Density ($\rho_e$) | Klein-Nishina $f_{KN}(E)$ | High Energy (> 50 keV) | Plastics, Water, Tissue, Explosives |

**Table 2: Proposed Tensor Dimensions for AethelgardNet** 

| Tensor Name | Shape | Description |
| :--- | :--- | :--- |
| input_raw | (B, 2, H, W) | Log-attenuation at Low and High kVp |
| poly_stack | (B, 6, H, W) | Polynomial expansion terms ($1, L, H, L^2, LH, H^2$) |
| physics_maps | (B, 4, H, W) | Decomposed Maps: $A_1, A_2, Z_{eff}, \nabla R$ |
| vision_feats | (B, C, H', W') | CNN Feature Map (e.g., ResNet Layer X) |
| attention_mask | (B, 1, H', W') | Gating mask derived from physics_maps |
| output_logits | (B, NumClasses) | Final classification probability |

### References
1.  Our Team | Analytical AI, accessed December 29, 2025, https://www.analyticalai.com/about
2.  Technology | Analytical AI, accessed December 29, 2025, https://www.analyticalai.com/technology
3.  Michael Moran Resume 12-25 (1).pdf
4.  US4029963A - X-ray spectral decomposition imaging system - Google Patents, accessed December 29, 2025, https://patents.google.com/patent/US4029963A/en
5.  Conditions for the invertibility of dual energy data - arXiv, accessed December 29, 2025, https://arxiv.org/pdf/1711.10836
6.  Projection decomposition via univariate optimization for dual-energy CT - PMC - NIH, accessed December 29, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC9427723/
7.  Material decomposition with dual- and multi-energy computed tomography | MRS Communications | Cambridge Core, accessed December 29, 2025, https://www.cambridge.org/core/journals/mrs-communications/article/material-decomposition-with-dual-and-multienergy-computed-tomography/84DE8C879159495378747FCFDC23709C
8.  Dual-energy CT | Radiology Reference Article | Radiopaedia.org, accessed December 29, 2025, https://radiopaedia.org/articles/dual-energy-ct-4
9.  Energy-selective reconstructions in X-ray computerized tomography - ResearchGate, accessed December 29, 2025, https://www.researchgate.net/publication/22186886_Energy-selective_reconstructions_in_X-ray_computerized_tomography
10. Dual Energy CT: Physics Principles - AAPM, accessed December 29, 2025, https://www.aapm.org/meetings/amos2/pdf/35-9870-60838-480.pdf
11. An image-domain, contrast material extraction method for Dual-Energy CT - PMC - NIH, accessed December 29, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC5339044/
12. The exponential edge-gradient effect in x-ray computed tomography - PubMed, accessed December 29, 2025, https://pubmed.ncbi.nlm.nih.gov/7243880/
13. Experimental feasibility of dual-energy X-ray tomography for two-phase density analysis in bentonite during water infiltration - NIH, accessed December 29, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC12589600/
14. Duality (optimization) - Wikipedia, accessed December 29, 2025, https://en.wikipedia.org/wiki/Duality_(optimization)
15. (PDF) A modelling approach to beam hardening correction - ResearchGate, accessed December 29, 2025, https://www.researchgate.net/publication/253926613_A_modelling_approach_to_beam_hardening_correction
16. MB-DECTNet: a model-based unrolling network for accurate 3D dual-energy CT reconstruction from clinically acquired helical scans - PubMed Central, accessed December 29, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC10714406/
17. CT Physics: Beam Hardening and Dual-Energy CT - XRayPhysics, accessed December 29, 2025, http://xrayphysics.com/dual_energy.html
188. A Fully Differentiable Framework for 2D/3D Registration and the Projective Spatial Transformers - PMC - NIH, accessed December 29, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC10879149/
19. A Beam Hardening Artifact Correction Method for CT Images Based on VGG Feature Extraction Networks - MDPI, accessed December 29, 2025, https://www.mdpi.com/1424-8220/25/7/2088
20. A novel beam hardening correction method requiring no prior knowledge, incorporated in an iterative reconstruction algorithm - ResearchGate, accessed December 29, 2025, https://www.researchgate.net/publication/257422345_A_novel_beam_hardening_correction_method_requiring_no_prior_knowledge_incorporated_in_an_iterative_reconstruction_algorithm
21. An automatic calibration method for dual energy material decomposition - ResearchGate, accessed December 29, 2025, https://www.researchgate.net/publication/228995401_An_automatic_calibration_method_for_dual_energy_material_decomposition