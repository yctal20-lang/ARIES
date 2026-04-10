"""Virtual power / energy sensor for spacecraft energy management."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ...simulation.physics import EARTH_RADIUS, SOLAR_CONSTANT


@dataclass
class PowerReading:
    """Power subsystem reading."""
    battery_soc: float         # State of charge 0-1
    solar_power: float         # Current solar panel output (W)
    bus_voltage: float         # Power bus voltage (V)
    current_draw: float        # Total current draw (A)
    power_balance: float       # Generation minus consumption (W)
    eclipse_state: bool        # True when in Earth shadow
    battery_temperature: float # Battery temperature (C)
    timestamp: float


class VirtualPowerSensor:
    """
    Virtual power subsystem sensor.

    Models:
    - Solar panel output based on panel area, efficiency, and Sun angle.
    - Eclipse detection from orbital geometry (cylindrical shadow).
    - Battery state-of-charge with simple coulomb-counting model.
    - Thermal effects on battery (simplified).
    """

    NOMINAL_BUS_VOLTAGE = 28.0  # V
    BASE_POWER_DRAW = 150.0     # W (avionics, comms, thermal)

    def __init__(
        self,
        solar_panel_area: float = 20.0,    # m^2
        solar_efficiency: float = 0.30,
        battery_capacity_wh: float = 10000.0,
        initial_soc: float = 1.0,
        noise_std: float = 0.005,
        seed: Optional[int] = None,
    ):
        self.solar_panel_area = solar_panel_area
        self.solar_efficiency = solar_efficiency
        self.battery_capacity_wh = battery_capacity_wh
        self.soc = initial_soc
        self.noise_std = noise_std
        self._rng = np.random.RandomState(seed)
        self._last_time: Optional[float] = None

    def read(
        self,
        spacecraft_position: np.ndarray,
        spacecraft_attitude: np.ndarray,
        thrust_magnitude: float = 0.0,
        timestamp: float = 0.0,
    ) -> PowerReading:
        dt = 0.0
        if self._last_time is not None:
            dt = max(0.0, timestamp - self._last_time)
        self._last_time = timestamp

        eclipse = self._is_eclipsed(spacecraft_position)

        if eclipse:
            solar_power = 0.0
        else:
            cos_angle = self._sun_incidence_angle(spacecraft_position, spacecraft_attitude)
            solar_power = (
                SOLAR_CONSTANT * self.solar_panel_area
                * self.solar_efficiency * max(0.0, cos_angle)
            )

        thruster_power = thrust_magnitude * 0.5  # rough electrical cost
        power_draw = self.BASE_POWER_DRAW + thruster_power
        power_balance = solar_power - power_draw

        if dt > 0 and self.battery_capacity_wh > 0:
            energy_delta_wh = power_balance * (dt / 3600.0)
            self.soc = np.clip(self.soc + energy_delta_wh / self.battery_capacity_wh, 0.0, 1.0)

        bus_voltage = self.NOMINAL_BUS_VOLTAGE * (0.9 + 0.1 * self.soc)
        current_draw = power_draw / bus_voltage if bus_voltage > 0 else 0.0
        battery_temp = 20.0 + 5.0 * (1.0 - self.soc) + (10.0 if not eclipse else -5.0)

        # Add sensor noise
        soc_noisy = np.clip(self.soc + self._rng.randn() * self.noise_std, 0.0, 1.0)
        solar_noisy = max(0.0, solar_power + self._rng.randn() * solar_power * 0.01)

        return PowerReading(
            battery_soc=soc_noisy,
            solar_power=solar_noisy,
            bus_voltage=bus_voltage + self._rng.randn() * 0.05,
            current_draw=current_draw + self._rng.randn() * 0.02,
            power_balance=solar_noisy - power_draw,
            eclipse_state=eclipse,
            battery_temperature=battery_temp + self._rng.randn() * 0.5,
            timestamp=timestamp,
        )

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
    def _sun_incidence_angle(position: np.ndarray, attitude: np.ndarray) -> float:
        """Cosine of angle between Sun vector and panel normal (body +Z)."""
        sun_eci = np.array([1.0, 0.0, 0.0])
        w, x, y, z = attitude
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ])
        panel_normal_eci = R @ np.array([0.0, 0.0, 1.0])
        return np.dot(panel_normal_eci, sun_eci)
