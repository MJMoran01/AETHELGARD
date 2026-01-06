"""
Project AETHELGARD: Physics-Gated Attention Network for Dual-Energy X-Ray Security Imaging

A Glass Box approach to X-ray threat detection.
"Physics filters the Signal; AI interprets the Texture."

Author: Michael Moran, Senior AI Engineer
Supervisor: Dr. Thomas Anthony, CTO, Analytical AI
"""

__version__ = "0.1.0"
__author__ = "Michael Moran"

from .preprocessing import RawToLogAttenuation
from .physics_head import PhysicsHead
from .dataset import DualEnergyDataset, SyntheticDualEnergyDataset

__all__ = [
    "RawToLogAttenuation",
    "PhysicsHead",
    "DualEnergyDataset",
    "SyntheticDualEnergyDataset",
]
