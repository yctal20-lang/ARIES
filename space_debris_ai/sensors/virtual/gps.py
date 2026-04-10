"""Virtual GPS sensor for spacecraft navigation."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ...simulation.physics import EARTH_RADIUS


@dataclass
class GPSReading:
    """GPS receiver reading."""
    position: np.ndarray       # ECI position (km)
    velocity: np.ndarray       # ECI velocity (km/s)
    hdop: float                # Horizontal dilution of precision
    num_satellites: int        # Visible satellites
    fix_valid: bool            # Whether the fix is usable
    timestamp: float


class VirtualGPS:
    """
    Virtual GPS sensor.

    Adds Gaussian noise to true position/velocity.  Signal is lost when the
    spacecraft is in Earth's geometric shadow (no line-of-sight to enough
    GPS constellation satellites at the far side of the orbit).
    """

    def __init__(
        self,
        position_noise_std: float = 0.01,   # km (~10 m)
        velocity_noise_std: float = 0.001,  # km/s (~1 m/s)
        update_rate: float = 1.0,           # Hz
        seed: Optional[int] = None,
    ):
        self.position_noise_std = position_noise_std
        self.velocity_noise_std = velocity_noise_std
        self.update_rate = update_rate
        self._rng = np.random.RandomState(seed)
        self._last_reading: Optional[GPSReading] = None
        self._last_update_time: float = -1e9

    def read(
        self,
        true_position: np.ndarray,
        true_velocity: np.ndarray,
        timestamp: float = 0.0,
    ) -> GPSReading:
        dt_since = timestamp - self._last_update_time
        if dt_since < 1.0 / self.update_rate and self._last_reading is not None:
            return self._last_reading

        in_shadow = self._is_in_earth_shadow(true_position)

        if in_shadow:
            num_sats = self._rng.randint(0, 4)
        else:
            num_sats = self._rng.randint(6, 12)

        fix_valid = bool(int(num_sats) >= 4)

        if fix_valid:
            hdop = 1.0 + self._rng.exponential(0.5)
            noise_scale = hdop
            position = true_position + self._rng.randn(3) * self.position_noise_std * noise_scale
            velocity = true_velocity + self._rng.randn(3) * self.velocity_noise_std * noise_scale
        else:
            hdop = 99.0
            position = np.full(3, np.nan)
            velocity = np.full(3, np.nan)

        reading = GPSReading(
            position=position,
            velocity=velocity,
            hdop=hdop,
            num_satellites=num_sats,
            fix_valid=fix_valid,
            timestamp=timestamp,
        )
        self._last_reading = reading
        self._last_update_time = timestamp
        return reading

    @staticmethod
    def _is_in_earth_shadow(position: np.ndarray) -> bool:
        """Simplified geometric shadow test (cylindrical shadow model)."""
        r = np.linalg.norm(position)
        if r < EARTH_RADIUS:
            return True
        # Assume Sun is along +X in ECI (simplified)
        sun_dir = np.array([1.0, 0.0, 0.0])
        proj = np.dot(position, sun_dir)
        if proj >= 0:
            return False
        perp_dist = np.sqrt(r ** 2 - proj ** 2)
        return bool(perp_dist < EARTH_RADIUS)
