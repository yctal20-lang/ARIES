"""Lidar sensor interface."""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LidarReading:
    """Lidar sensor reading."""
    points: np.ndarray  # [N, 3] point cloud
    intensities: np.ndarray  # [N] intensity values
    timestamp: float


class LidarSensor:
    """Lidar sensor interface."""
    
    def __init__(
        self,
        max_range: float = 1000.0,
        angular_resolution: float = 0.1,
        noise_std: float = 0.05,
    ):
        self.max_range = max_range
        self.angular_resolution = angular_resolution
        self.noise_std = noise_std
    
    def read(
        self,
        objects: List[dict],
        sensor_position: np.ndarray,
        timestamp: float = 0.0,
    ) -> LidarReading:
        """Generate lidar reading from objects."""
        points = []
        intensities = []
        
        for obj in objects:
            pos = np.asarray(obj["position"])
            rel_pos = pos - sensor_position
            distance = np.linalg.norm(rel_pos)
            
            if distance < self.max_range:
                # Add noise
                noisy_pos = rel_pos + np.random.randn(3) * self.noise_std
                points.append(noisy_pos)
                intensities.append(1.0 / (distance + 1))
        
        if points:
            return LidarReading(
                points=np.array(points),
                intensities=np.array(intensities),
                timestamp=timestamp,
            )
        return LidarReading(
            points=np.zeros((0, 3)),
            intensities=np.zeros(0),
            timestamp=timestamp,
        )
