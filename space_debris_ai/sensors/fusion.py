"""Sensor fusion module."""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FusedState:
    """Fused sensor state."""
    position: np.ndarray
    velocity: np.ndarray
    attitude: np.ndarray
    angular_velocity: np.ndarray
    confidence: float


class SensorFusion:
    """Multi-sensor fusion using weighted averaging."""
    
    def __init__(self):
        self.weights = {
            "gps": 0.4,
            "imu": 0.3,
            "star_tracker": 0.3,
        }
    
    def fuse(
        self,
        gps_data: Optional[Dict] = None,
        imu_data: Optional[Dict] = None,
        star_tracker_data: Optional[Dict] = None,
    ) -> FusedState:
        """Fuse sensor data."""
        position = np.zeros(3)
        velocity = np.zeros(3)
        attitude = np.array([1, 0, 0, 0])
        angular_velocity = np.zeros(3)
        total_weight = 0
        
        if gps_data:
            w = self.weights["gps"]
            position += w * np.asarray(gps_data.get("position", np.zeros(3)))
            velocity += w * np.asarray(gps_data.get("velocity", np.zeros(3)))
            total_weight += w
        
        if imu_data:
            angular_velocity = np.asarray(imu_data.get("angular_velocity", np.zeros(3)))
        
        if star_tracker_data:
            attitude = np.asarray(star_tracker_data.get("attitude", attitude))
        
        if total_weight > 0:
            position /= total_weight
            velocity /= total_weight
        
        return FusedState(
            position=position,
            velocity=velocity,
            attitude=attitude,
            angular_velocity=angular_velocity,
            confidence=min(1.0, total_weight),
        )
