"""Virtual Sun sensor for attitude determination and power estimation."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ...simulation.physics import EARTH_RADIUS, SOLAR_CONSTANT


@dataclass
class SunSensorReading:
    """Sun sensor reading."""
    sun_vector: np.ndarray    # Unit vector toward Sun in body frame
    sun_visible: bool         # False when eclipsed or outside FOV
    intensity: float          # Measured solar intensity (W/m^2)
    timestamp: float


class VirtualSunSensor:
    """
    Virtual Sun sensor.

    Computes the Sun direction in the spacecraft body frame.
    Reports ``sun_visible=False`` when the spacecraft is in Earth's
    shadow or when the Sun falls outside the sensor's hemisphere.
    """

    def __init__(
        self,
        noise_deg: float = 0.5,     # per-axis angular noise (degrees)
        update_rate: float = 10.0,   # Hz
        seed: Optional[int] = None,
    ):
        self.noise_rad = np.radians(noise_deg)
        self.update_rate = update_rate
        self._rng = np.random.RandomState(seed)
        self._last_reading: Optional[SunSensorReading] = None
        self._last_update_time: float = -1e9

    def read(
        self,
        spacecraft_position: np.ndarray,
        spacecraft_attitude: np.ndarray,
        timestamp: float = 0.0,
    ) -> SunSensorReading:
        dt_since = timestamp - self._last_update_time
        if dt_since < 1.0 / self.update_rate and self._last_reading is not None:
            return self._last_reading

        eclipsed = self._is_eclipsed(spacecraft_position)

        sun_eci = np.array([1.0, 0.0, 0.0])  # simplified constant Sun direction
        R = self._quat_to_matrix(spacecraft_attitude)
        sun_body = R.T @ sun_eci  # ECI -> body

        # Sun is "visible" if not eclipsed and in the sensor hemisphere (+X body)
        sun_visible = bool((not eclipsed) and (sun_body[0] > 0))

        if sun_visible:
            noise_axis = self._rng.randn(3)
            noise_axis /= np.linalg.norm(noise_axis) + 1e-12
            angle = self._rng.randn() * self.noise_rad
            sun_body = self._rotate_vector(sun_body, noise_axis, angle)
            sun_body /= np.linalg.norm(sun_body) + 1e-12
            intensity = SOLAR_CONSTANT * (1.0 + self._rng.randn() * 0.005)
        else:
            sun_body = np.zeros(3)
            intensity = 0.0

        reading = SunSensorReading(
            sun_vector=sun_body,
            sun_visible=sun_visible,
            intensity=intensity,
            timestamp=timestamp,
        )
        self._last_reading = reading
        self._last_update_time = timestamp
        return reading

    @staticmethod
    def _is_eclipsed(position: np.ndarray) -> bool:
        r = np.linalg.norm(position)
        if r < EARTH_RADIUS:
            return True
        sun_dir = np.array([1.0, 0.0, 0.0])
        proj = np.dot(position, sun_dir)
        if proj >= 0:
            return False
        perp_dist = np.sqrt(r ** 2 - proj ** 2)
        return bool(perp_dist < EARTH_RADIUS)

    @staticmethod
    def _rotate_vector(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)

    @staticmethod
    def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ])
