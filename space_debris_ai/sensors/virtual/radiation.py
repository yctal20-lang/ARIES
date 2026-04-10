"""Virtual radiation sensor with SAA and solar-event models."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ...simulation.physics import EARTH_RADIUS


@dataclass
class RadiationReading:
    """Radiation environment reading."""
    dose_rate: float          # mSv/h
    particle_flux: float      # particles/cm^2/s
    solar_event: bool         # True during a solar proton event (SPE)
    cumulative_dose: float    # mSv since mission start
    in_saa: bool              # True when over the South Atlantic Anomaly
    timestamp: float


class VirtualRadiationSensor:
    """
    Virtual radiation sensor.

    Models two main LEO radiation sources:
    1. South Atlantic Anomaly (SAA) — persistent high-flux region.
    2. Solar Proton Events (SPE) — stochastic bursts.

    The SAA is modelled as a geographic zone (roughly lat -30, lon -45)
    projected into ECI coordinates with a simplified rotation model.
    """

    BASE_DOSE_RATE = 0.01     # mSv/h (galactic cosmic ray background on LEO)
    SAA_DOSE_RATE = 0.5       # mSv/h peak in the SAA
    SPE_DOSE_RATE = 5.0       # mSv/h during a solar event
    SPE_PROBABILITY = 1e-5    # per-second probability of a new SPE starting
    SPE_DURATION = 3600.0     # seconds (typical ~1 hour burst)

    def __init__(
        self,
        noise_std: float = 0.002,
        seed: Optional[int] = None,
    ):
        self.noise_std = noise_std
        self._rng = np.random.RandomState(seed)
        self._cumulative_dose = 0.0
        self._spe_active = False
        self._spe_end_time = 0.0
        self._last_time: Optional[float] = None

    def read(
        self,
        spacecraft_position: np.ndarray,
        timestamp: float = 0.0,
    ) -> RadiationReading:
        dt = 0.0
        if self._last_time is not None:
            dt = max(0.0, timestamp - self._last_time)
        self._last_time = timestamp

        in_saa = self._check_saa(spacecraft_position, timestamp)

        # SPE stochastic model
        if not self._spe_active:
            if self._rng.rand() < self.SPE_PROBABILITY * max(dt, 0.1):
                self._spe_active = True
                self._spe_end_time = timestamp + self.SPE_DURATION
        else:
            if timestamp >= self._spe_end_time:
                self._spe_active = False

        dose_rate = self.BASE_DOSE_RATE
        if in_saa:
            dose_rate += self.SAA_DOSE_RATE
        if self._spe_active:
            dose_rate += self.SPE_DOSE_RATE

        particle_flux = dose_rate * 1e4  # rough proportional mapping

        if dt > 0:
            self._cumulative_dose += dose_rate * (dt / 3600.0)

        dose_rate_noisy = max(0.0, dose_rate + self._rng.randn() * self.noise_std)

        return RadiationReading(
            dose_rate=dose_rate_noisy,
            particle_flux=particle_flux + self._rng.randn() * 10.0,
            solar_event=self._spe_active,
            cumulative_dose=self._cumulative_dose,
            in_saa=in_saa,
            timestamp=timestamp,
        )

    @staticmethod
    def _check_saa(position_km: np.ndarray, time_s: float) -> bool:
        """
        Simplified SAA check.

        The SAA is centred around geographic (lat=-30, lon=-45).
        We approximate by rotating the ECI position to account for
        Earth rotation and checking if the sub-satellite point falls
        within an elliptical region.
        """
        EARTH_ROT_RATE = 7.2921159e-5  # rad/s
        theta = EARTH_ROT_RATE * time_s

        # Rotate position to ECEF (approx)
        c, s = np.cos(theta), np.sin(theta)
        x_ecef = position_km[0] * c + position_km[1] * s
        y_ecef = -position_km[0] * s + position_km[1] * c
        z_ecef = position_km[2]

        r = np.sqrt(x_ecef ** 2 + y_ecef ** 2 + z_ecef ** 2)
        if r < 1e-3:
            return False

        lat = np.degrees(np.arcsin(z_ecef / r))
        lon = np.degrees(np.arctan2(y_ecef, x_ecef))

        # Elliptical SAA region
        dlat = (lat - (-30.0)) / 20.0
        dlon = (lon - (-45.0)) / 40.0
        return bool((dlat ** 2 + dlon ** 2) < 1.0)
