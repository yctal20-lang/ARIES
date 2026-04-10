"""Virtual star tracker sensor for attitude determination."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ...simulation.physics import EARTH_RADIUS


@dataclass
class StarTrackerReading:
    """Star tracker reading."""
    attitude: np.ndarray       # Quaternion [w, x, y, z]
    angular_rate: np.ndarray   # Estimated angular rate (rad/s)
    confidence: float          # 0-1 tracking confidence
    num_stars: int             # Number of tracked stars
    blinded: bool              # True when Sun or Earth limb in FOV
    timestamp: float


class VirtualStarTracker:
    """
    Virtual star tracker sensor.

    Adds small angular noise to the true attitude quaternion.
    Tracking degrades when the spacecraft rotates fast (>2 deg/s)
    and is lost entirely when the Sun enters the FOV.
    """

    def __init__(
        self,
        attitude_noise_deg: float = 0.01,  # 1-sigma per-axis (degrees)
        max_angular_rate: float = 0.035,   # rad/s (~2 deg/s) for valid tracking
        fov_half_angle: float = 10.0,      # degrees
        update_rate: float = 5.0,          # Hz
        seed: Optional[int] = None,
    ):
        self.attitude_noise_rad = np.radians(attitude_noise_deg)
        self.max_angular_rate = max_angular_rate
        self.fov_half_angle_rad = np.radians(fov_half_angle)
        self.update_rate = update_rate
        self._rng = np.random.RandomState(seed)
        self._last_reading: Optional[StarTrackerReading] = None
        self._last_update_time: float = -1e9

    def read(
        self,
        true_attitude: np.ndarray,
        true_angular_velocity: np.ndarray,
        spacecraft_position: np.ndarray,
        timestamp: float = 0.0,
    ) -> StarTrackerReading:
        dt_since = timestamp - self._last_update_time
        if dt_since < 1.0 / self.update_rate and self._last_reading is not None:
            return self._last_reading

        omega_mag = np.linalg.norm(true_angular_velocity)
        blinded = self._check_blinding(true_attitude, spacecraft_position)

        if blinded:
            confidence = 0.0
            num_stars = 0
            attitude = self._last_reading.attitude.copy() if self._last_reading else true_attitude.copy()
            angular_rate = true_angular_velocity.copy()
        elif omega_mag > self.max_angular_rate:
            degradation = min(1.0, omega_mag / self.max_angular_rate - 1.0)
            confidence = max(0.1, 1.0 - degradation)
            num_stars = max(3, int(15 * confidence))
            noise_scale = 1.0 / confidence
            attitude = self._add_quaternion_noise(true_attitude, noise_scale)
            angular_rate = true_angular_velocity + self._rng.randn(3) * 0.001 * noise_scale
        else:
            confidence = 0.95 + self._rng.uniform(0, 0.05)
            num_stars = self._rng.randint(10, 25)
            attitude = self._add_quaternion_noise(true_attitude, 1.0)
            angular_rate = true_angular_velocity + self._rng.randn(3) * 0.0005

        reading = StarTrackerReading(
            attitude=attitude,
            angular_rate=angular_rate,
            confidence=confidence,
            num_stars=num_stars,
            blinded=blinded,
            timestamp=timestamp,
        )
        self._last_reading = reading
        self._last_update_time = timestamp
        return reading

    def _add_quaternion_noise(self, q: np.ndarray, scale: float) -> np.ndarray:
        axis_noise = self._rng.randn(3)
        axis_noise /= np.linalg.norm(axis_noise) + 1e-12
        angle = self._rng.randn() * self.attitude_noise_rad * scale
        half = angle / 2.0
        dq = np.array([np.cos(half), *(axis_noise * np.sin(half))])
        result = self._qmul(q, dq)
        return result / np.linalg.norm(result)

    @staticmethod
    def _qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def _check_blinding(self, attitude: np.ndarray, position: np.ndarray) -> bool:
        """Check if the Sun falls within the tracker boresight FOV."""
        sun_eci = np.array([1.0, 0.0, 0.0])  # simplified Sun direction
        R = self._quat_to_matrix(attitude)
        boresight_eci = R @ np.array([0.0, 0.0, 1.0])  # tracker along +Z body
        cos_angle = np.dot(boresight_eci, sun_eci)
        return bool(cos_angle > np.cos(self.fov_half_angle_rad))

    @staticmethod
    def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ])
