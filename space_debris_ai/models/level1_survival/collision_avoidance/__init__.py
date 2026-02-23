"""
Collision Avoidance Module.

Components:
- CollisionDetector: CNN-based collision detection and risk assessment
- CollisionAvoidanceAgent: SAC-based reactive avoidance controller
"""

from .detector import CollisionDetector
from .agent import CollisionAvoidanceAgent, CollisionAvoidanceModule

__all__ = [
    "CollisionDetector",
    "CollisionAvoidanceAgent",
    "CollisionAvoidanceModule",
]
