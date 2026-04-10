"""Virtual magnetometer sensor (simplified IGRF dipole model)."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ...simulation.physics import EARTH_RADIUS


@dataclass
class MagnetometerReading:
    """Three-axis magnetometer reading."""
    magnetic_field: np.ndarray   # Body-frame magnetic field (micro-Tesla)
    field_magnitude: float       # Scalar magnitude (micro-Tesla)
    timestamp: float


class VirtualMagnetometer:
    """
    Virtual magnetometer using a tilted dipole approximation of Earth's
    magnetic field (IGRF simplified).

    The geomagnetic dipole moment is ~7.94e15 T*m^3.
    On LEO (~400 km) the field is roughly 25-65 micro-Tesla depending on
    latitude and longitude.
    """

    DIPOLE_MOMENT = 7.94e15  # T * m^3

    def __init__(
        self,
        noise_std_ut: float = 0.1,   # micro-Tesla per axis
        update_rate: float = 10.0,    # Hz
        seed: Optional[int] = None,
    ):
        self.noise_std = noise_std_ut
        self.update_rate = update_rate
        self._rng = np.random.RandomState(seed)
        self._last_reading: Optional[MagnetometerReading] = None
        self._last_update_time: float = -1e9

    def read(
        self,
        spacecraft_position: np.ndarray,
        spacecraft_attitude: np.ndarray,
        timestamp: float = 0.0,
    ) -> MagnetometerReading:
        dt_since = timestamp - self._last_update_time
        if dt_since < 1.0 / self.update_rate and self._last_reading is not None:
            return self._last_reading

        b_eci = self._dipole_field_eci(spacecraft_position)

        # Transform to body frame
        R = self._quat_to_matrix(spacecraft_attitude)
        b_body = R.T @ b_eci  # ECI -> body

        b_body += self._rng.randn(3) * self.noise_std

        reading = MagnetometerReading(
            magnetic_field=b_body,
            field_magnitude=float(np.linalg.norm(b_body)),
            timestamp=timestamp,
        )
        self._last_reading = reading
        self._last_update_time = timestamp
        return reading

    def _dipole_field_eci(self, position_km: np.ndarray) -> np.ndarray:
        """Compute magnetic field at position using a centred dipole aligned with +Z (simplified)."""
        r_m = position_km * 1e3  # km -> m
        r_mag = np.linalg.norm(r_m)
        if r_mag < 1e3:
            return np.zeros(3)

        r_hat = r_m / r_mag
        # Dipole axis ~ geographic north (tilted 11 deg, but we use aligned-Z here)
        m_hat = np.array([0.0, 0.0, 1.0])

        m_dot_r = np.dot(m_hat, r_hat)
        B = (self.DIPOLE_MOMENT / r_mag ** 3) * (3.0 * m_dot_r * r_hat - m_hat)
        B_ut = B * 1e6  # Tesla -> micro-Tesla
        return B_ut

    @staticmethod
    def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ])
