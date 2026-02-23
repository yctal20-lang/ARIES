"""Camera sensor interface."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CameraReading:
    """Camera sensor reading."""
    image: np.ndarray
    timestamp: float


class CameraSensor:
    """Camera sensor interface."""
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        fov: float = 60.0,
        noise_std: float = 0.01,
    ):
        self.resolution = resolution
        self.fov = fov
        self.noise_std = noise_std
    
    def read(self, scene: np.ndarray, timestamp: float = 0.0) -> CameraReading:
        """Get camera reading."""
        # Add noise
        noisy = scene + np.random.randn(*scene.shape) * self.noise_std
        noisy = np.clip(noisy, 0, 1)
        
        return CameraReading(image=noisy, timestamp=timestamp)
