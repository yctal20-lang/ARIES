"""Virtual thermal sensor array for spacecraft component temperatures."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from ...simulation.physics import EARTH_RADIUS


# Component names must match the TFT failure-prediction component list
COMPONENT_NAMES = [
    "main_thruster",
    "rcs_system",
    "solar_panel_1",
    "solar_panel_2",
    "battery_1",
    "battery_2",
    "lidar",
    "radar",
    "computer",
    "communication",
]


@dataclass
class ThermalReading:
    """Thermal array reading."""
    temperatures: Dict[str, float]   # Component name -> temperature (C)
    mean_temperature: float
    max_temperature: float
    timestamp: float


class VirtualThermalArray:
    """
    Virtual thermal sensor array for 10 spacecraft components.

    Simplified thermal-balance model:
    - Solar input heats external components when not eclipsed.
    - Radiative cooling always present.
    - Internal heat from electronics / thrusters.
    - Each component has its own thermal mass and coupling coefficients.
    """

    def __init__(
        self,
        noise_std: float = 0.5,    # C per reading
        update_rate: float = 1.0,   # Hz
        seed: Optional[int] = None,
    ):
        self.noise_std = noise_std
        self.update_rate = update_rate
        self._rng = np.random.RandomState(seed)
        self._last_reading: Optional[ThermalReading] = None
        self._last_update_time: float = -1e9

        # Equilibrium temperatures in sunlight / eclipse (C)
        self._sun_eq = {
            "main_thruster": 80.0,
            "rcs_system": 40.0,
            "solar_panel_1": 65.0,
            "solar_panel_2": 65.0,
            "battery_1": 25.0,
            "battery_2": 25.0,
            "lidar": 30.0,
            "radar": 35.0,
            "computer": 45.0,
            "communication": 38.0,
        }
        self._eclipse_eq = {
            "main_thruster": -30.0,
            "rcs_system": -10.0,
            "solar_panel_1": -60.0,
            "solar_panel_2": -60.0,
            "battery_1": 10.0,
            "battery_2": 10.0,
            "lidar": 5.0,
            "radar": 8.0,
            "computer": 30.0,
            "communication": 15.0,
        }
        # Thermal time constants (seconds) — how fast each component
        # approaches equilibrium.  Large = slow change.
        self._tau = {
            "main_thruster": 300.0,
            "rcs_system": 200.0,
            "solar_panel_1": 60.0,
            "solar_panel_2": 60.0,
            "battery_1": 600.0,
            "battery_2": 600.0,
            "lidar": 150.0,
            "radar": 150.0,
            "computer": 400.0,
            "communication": 250.0,
        }

        # Current temperatures (start at a comfortable 20 C)
        self._temps = {name: 20.0 for name in COMPONENT_NAMES}

    def read(
        self,
        spacecraft_position: np.ndarray,
        thrust_active: bool = False,
        timestamp: float = 0.0,
    ) -> ThermalReading:
        dt_since = timestamp - self._last_update_time
        if 0 < dt_since < 1.0 / self.update_rate and self._last_reading is not None:
            return self._last_reading

        dt = max(0.0, dt_since) if self._last_update_time > -1e8 else 0.0

        eclipsed = self._is_eclipsed(spacecraft_position)
        eq = self._eclipse_eq if eclipsed else self._sun_eq

        if thrust_active:
            eq = dict(eq)
            eq["main_thruster"] += 120.0
            eq["rcs_system"] += 30.0

        # Exponential relaxation toward equilibrium
        for name in COMPONENT_NAMES:
            tau = self._tau[name]
            if dt > 0 and tau > 0:
                alpha = 1.0 - np.exp(-dt / tau)
                self._temps[name] += alpha * (eq[name] - self._temps[name])

        temps_noisy = {
            name: self._temps[name] + self._rng.randn() * self.noise_std
            for name in COMPONENT_NAMES
        }

        vals = list(temps_noisy.values())
        reading = ThermalReading(
            temperatures=temps_noisy,
            mean_temperature=float(np.mean(vals)),
            max_temperature=float(np.max(vals)),
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
