"""Virtual Doppler velocity sensor for measuring relative approach speed."""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DopplerReading:
    """Doppler velocity sensor reading."""
    radial_velocity: float     # m/s (positive = receding)
    closing_rate: float        # m/s (positive = approaching)
    signal_strength: float     # dBm (higher = stronger return)
    target_detected: bool
    timestamp: float


class VirtualDopplerSensor:
    """
    Virtual Doppler velocity sensor.

    Measures the line-of-sight (radial) component of relative velocity
    between the spacecraft and the nearest debris object within range.
    """

    def __init__(
        self,
        max_range: float = 2.0,           # km
        velocity_noise_std: float = 0.01,  # m/s
        update_rate: float = 5.0,          # Hz
        seed: Optional[int] = None,
    ):
        self.max_range = max_range
        self.velocity_noise_std = velocity_noise_std
        self.update_rate = update_rate
        self._rng = np.random.RandomState(seed)
        self._last_reading: Optional[DopplerReading] = None
        self._last_update_time: float = -1e9

    def read(
        self,
        spacecraft_position: np.ndarray,
        spacecraft_velocity: np.ndarray,
        debris_positions: List[np.ndarray],
        debris_velocities: List[np.ndarray],
        timestamp: float = 0.0,
    ) -> DopplerReading:
        dt_since = timestamp - self._last_update_time
        if dt_since < 1.0 / self.update_rate and self._last_reading is not None:
            return self._last_reading

        best_dist = float("inf")
        best_radial = 0.0

        for pos, vel in zip(debris_positions, debris_velocities):
            rel_pos = np.asarray(pos) - spacecraft_position
            dist = np.linalg.norm(rel_pos)
            if dist > self.max_range or dist < 1e-9:
                continue
            if dist < best_dist:
                los = rel_pos / dist
                rel_vel = np.asarray(vel) - spacecraft_velocity
                best_radial = np.dot(rel_vel, los)  # km/s
                best_dist = dist

        detected = bool(best_dist <= self.max_range)

        if detected:
            radial_ms = best_radial * 1000.0  # km/s -> m/s
            radial_ms += self._rng.randn() * self.velocity_noise_std
            closing = -radial_ms
            signal_dbm = -50.0 + 20.0 * np.log10(1.0 / (best_dist + 1e-6))
        else:
            radial_ms = 0.0
            closing = 0.0
            signal_dbm = -120.0

        reading = DopplerReading(
            radial_velocity=radial_ms,
            closing_rate=closing,
            signal_strength=signal_dbm,
            target_detected=detected,
            timestamp=timestamp,
        )
        self._last_reading = reading
        self._last_update_time = timestamp
        return reading
