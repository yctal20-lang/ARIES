"""IMU sensor interface."""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class IMUReading:
    """IMU sensor reading."""
    acceleration: np.ndarray  # m/s²
    angular_velocity: np.ndarray  # rad/s
    timestamp: float


class IMUSensor:
    """IMU sensor interface with noise model."""
    
    def __init__(
        self,
        accel_noise_std: float = 0.01,
        gyro_noise_std: float = 0.001,
        accel_bias: Optional[np.ndarray] = None,
        gyro_bias: Optional[np.ndarray] = None,
    ):
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.accel_bias = accel_bias if accel_bias is not None else np.zeros(3)
        self.gyro_bias = gyro_bias if gyro_bias is not None else np.zeros(3)
    
    def read(
        self,
        true_accel: np.ndarray,
        true_angular_vel: np.ndarray,
        timestamp: float = 0.0,
    ) -> IMUReading:
        """Get noisy IMU reading."""
        accel = true_accel + self.accel_bias + np.random.randn(3) * self.accel_noise_std
        gyro = true_angular_vel + self.gyro_bias + np.random.randn(3) * self.gyro_noise_std
        
        return IMUReading(
            acceleration=accel,
            angular_velocity=gyro,
            timestamp=timestamp,
        )
