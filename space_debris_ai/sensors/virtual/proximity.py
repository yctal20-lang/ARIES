"""Virtual proximity sensor for close-range debris scanning."""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ProximityReading:
    """Proximity sensor reading."""
    range_to_target: float      # Distance to nearest object (m)
    closing_velocity: float     # Radial closing speed (m/s, positive = approaching)
    bearing: np.ndarray         # Azimuth, elevation in body frame (rad)
    target_size_estimate: float # Estimated size (m)
    target_detected: bool       # Whether any target is within range
    timestamp: float


class VirtualProximitySensor:
    """
    Virtual close-range proximity sensor (0.5 - 50 m operating range).

    Maps to the Arduino HC-SR04 ultrasonic distance sensor
    (scaled from centimetres to metres/kilometres in the sketch).
    """

    def __init__(
        self,
        min_range: float = 0.5,       # metres
        max_range: float = 50.0,      # metres
        range_noise_std: float = 0.01, # metres
        angle_noise_std: float = 0.02, # rad (~1 deg)
        update_rate: float = 10.0,     # Hz
        seed: Optional[int] = None,
    ):
        self.min_range = min_range
        self.max_range = max_range
        self.range_noise_std = range_noise_std
        self.angle_noise_std = angle_noise_std
        self.update_rate = update_rate
        self._rng = np.random.RandomState(seed)
        self._last_reading: Optional[ProximityReading] = None
        self._last_update_time: float = -1e9

    def read(
        self,
        spacecraft_position: np.ndarray,
        spacecraft_velocity: np.ndarray,
        spacecraft_attitude: np.ndarray,
        debris_positions: List[np.ndarray],
        debris_velocities: List[np.ndarray],
        debris_sizes: List[float],
        timestamp: float = 0.0,
    ) -> ProximityReading:
        dt_since = timestamp - self._last_update_time
        if dt_since < 1.0 / self.update_rate and self._last_reading is not None:
            return self._last_reading

        R = self._quat_to_matrix(spacecraft_attitude)
        best_range = float("inf")
        best_vel = 0.0
        best_bearing = np.zeros(2)
        best_size = 0.0

        for pos, vel, size in zip(debris_positions, debris_velocities, debris_sizes):
            rel_eci = np.asarray(pos) - spacecraft_position
            dist_km = np.linalg.norm(rel_eci)
            dist_m = dist_km * 1000.0

            if dist_m < self.min_range or dist_m > self.max_range:
                continue

            if dist_m < best_range:
                rel_body = R.T @ rel_eci
                rel_vel_eci = np.asarray(vel) - spacecraft_velocity
                closing = -np.dot(rel_vel_eci, rel_eci / (dist_km + 1e-12)) * 1000.0  # km/s -> m/s

                az = np.arctan2(rel_body[1], rel_body[0])
                el = np.arcsin(np.clip(rel_body[2] / (dist_km + 1e-12), -1, 1))

                best_range = dist_m
                best_vel = closing
                best_bearing = np.array([az, el])
                best_size = size

        detected = bool(best_range <= self.max_range)

        if detected:
            range_noisy = max(0.0, best_range + self._rng.randn() * self.range_noise_std)
            bearing_noisy = best_bearing + self._rng.randn(2) * self.angle_noise_std
            vel_noisy = best_vel + self._rng.randn() * 0.005
            size_noisy = best_size * (1.0 + self._rng.randn() * 0.1)
        else:
            range_noisy = float("inf")
            bearing_noisy = np.zeros(2)
            vel_noisy = 0.0
            size_noisy = 0.0

        reading = ProximityReading(
            range_to_target=range_noisy,
            closing_velocity=vel_noisy,
            bearing=bearing_noisy,
            target_size_estimate=max(0.0, size_noisy),
            target_detected=detected,
            timestamp=timestamp,
        )
        self._last_reading = reading
        self._last_update_time = timestamp
        return reading

    @staticmethod
    def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ])
