"""
Simulation environment for Space Debris Collector AI System.

Modules:
- environment: Gymnasium-compatible orbital environment
- physics: Orbital mechanics and spacecraft dynamics
- scenarios: Pre-defined training and testing scenarios
"""

from .environment import OrbitalEnv
from .physics import OrbitalMechanics, SpacecraftDynamics
from .scenarios import ScenarioGenerator, Scenario

__all__ = [
    "OrbitalEnv",
    "OrbitalMechanics",
    "SpacecraftDynamics",
    "ScenarioGenerator",
    "Scenario",
]
