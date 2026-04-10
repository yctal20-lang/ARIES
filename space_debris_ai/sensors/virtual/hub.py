"""
VirtualSensorHub — synchronised gateway for all virtual sensors.

One call to ``read_all()`` reads every sensor with a shared timestamp
and returns both structured readings and a flat telemetry vector
ready for ML consumption.
"""

import numpy as np
from typing import Any, Dict, List, Optional

from .gps import VirtualGPS, GPSReading
from .star_tracker import VirtualStarTracker, StarTrackerReading
from .radar import VirtualRadar, RadarReading
from .power import VirtualPowerSensor, PowerReading
from .magnetometer import VirtualMagnetometer, MagnetometerReading
from .sun_sensor import VirtualSunSensor, SunSensorReading
from .proximity import VirtualProximitySensor, ProximityReading
from .thermal import VirtualThermalArray, ThermalReading, COMPONENT_NAMES
from .doppler import VirtualDopplerSensor, DopplerReading
from .radiation import VirtualRadiationSensor, RadiationReading
from .spectrometer import VirtualSpectrometer, SpectrometerReading, SPECTRUM_DIM
from .accelerometer import VirtualAccelerometer, AccelerometerReading

from ..imu import IMUSensor, IMUReading
from ..lidar import LidarSensor, LidarReading


# Dimension of the flat telemetry vector produced by ``to_telemetry_vector``.
# GPS(3+3) + StarTracker(4+3) + Radar(64) + Power(7) + Magnetometer(3+1) +
# SunSensor(3+1+1) + Proximity(1+1+2+1) + Thermal(10+1+1) + Doppler(1+1+1) +
# Radiation(1+1+1+1) + Spectrometer(32+1) + Accelerometer(3+1) + IMU(3+3) = 160
TELEMETRY_DIM = 160


class VirtualSensorHub:
    """
    Centralised sensor hub that instantiates all 12 virtual sensors plus
    the existing IMU and LiDAR, and provides a single ``read_all()`` call
    that returns synchronised data.
    """

    def __init__(
        self,
        solar_panel_area: float = 20.0,
        solar_efficiency: float = 0.30,
        battery_capacity_wh: float = 10000.0,
        seed: Optional[int] = None,
    ):
        s = seed or 0

        # Group A — critical
        self.gps = VirtualGPS(seed=s + 1)
        self.star_tracker = VirtualStarTracker(seed=s + 2)
        self.radar = VirtualRadar(seed=s + 3)
        self.power = VirtualPowerSensor(
            solar_panel_area=solar_panel_area,
            solar_efficiency=solar_efficiency,
            battery_capacity_wh=battery_capacity_wh,
            seed=s + 4,
        )

        # Group B — important
        self.magnetometer = VirtualMagnetometer(seed=s + 5)
        self.sun_sensor = VirtualSunSensor(seed=s + 6)
        self.proximity = VirtualProximitySensor(seed=s + 7)
        self.thermal = VirtualThermalArray(seed=s + 8)
        self.doppler = VirtualDopplerSensor(seed=s + 9)

        # Group C — extended
        self.radiation = VirtualRadiationSensor(seed=s + 10)
        self.spectrometer = VirtualSpectrometer(seed=s + 11)
        self.accelerometer = VirtualAccelerometer(seed=s + 12)

        # Existing sensors
        self.imu = IMUSensor()
        self.lidar = LidarSensor()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def read_all(
        self,
        spacecraft_position: np.ndarray,
        spacecraft_velocity: np.ndarray,
        spacecraft_attitude: np.ndarray,
        spacecraft_angular_velocity: np.ndarray,
        debris_positions: List[np.ndarray],
        debris_velocities: List[np.ndarray],
        debris_sizes: List[float],
        debris_types: List[str],
        thrust_magnitude: float = 0.0,
        timestamp: float = 0.0,
    ) -> Dict[str, Any]:
        """Read every sensor and return a unified dict."""

        thrust_active = bool(float(thrust_magnitude) > 0.1)

        # --- Group A ---
        gps = self.gps.read(spacecraft_position, spacecraft_velocity, timestamp)
        star = self.star_tracker.read(
            spacecraft_attitude, spacecraft_angular_velocity,
            spacecraft_position, timestamp,
        )
        radar = self.radar.read(
            spacecraft_position, spacecraft_velocity,
            debris_positions, debris_velocities, debris_sizes, timestamp,
        )
        power = self.power.read(
            spacecraft_position, spacecraft_attitude,
            thrust_magnitude, timestamp,
        )

        # --- Group B ---
        mag = self.magnetometer.read(spacecraft_position, spacecraft_attitude, timestamp)
        sun = self.sun_sensor.read(spacecraft_position, spacecraft_attitude, timestamp)
        prox = self.proximity.read(
            spacecraft_position, spacecraft_velocity, spacecraft_attitude,
            debris_positions, debris_velocities, debris_sizes, timestamp,
        )
        therm = self.thermal.read(spacecraft_position, thrust_active, timestamp)
        doppler = self.doppler.read(
            spacecraft_position, spacecraft_velocity,
            debris_positions, debris_velocities, timestamp,
        )

        # --- Group C ---
        rad = self.radiation.read(spacecraft_position, timestamp)
        spec = self.spectrometer.read(
            spacecraft_position, debris_positions, debris_types,
            sun.sun_visible, timestamp,
        )

        # Accelerometer needs body-frame acceleration.
        # Approximate from IMU true values (gravity-free residual ~ 0 in free-fall)
        true_accel_body = np.zeros(3)
        if thrust_active:
            true_accel_body[0] = thrust_magnitude / 500.0  # rough F/m
        accel = self.accelerometer.read(true_accel_body, thrust_active, timestamp)

        # --- Existing sensors ---
        # IMU uses the simulation's true values
        true_accel_eci = np.zeros(3)  # free-fall: measured accel ≈ 0
        imu = self.imu.read(true_accel_eci, spacecraft_angular_velocity, timestamp)

        objects_for_lidar = [
            {"position": p} for p in debris_positions
        ]
        lidar = self.lidar.read(objects_for_lidar, spacecraft_position, timestamp)

        result: Dict[str, Any] = {
            # Structured readings
            "gps": gps,
            "star_tracker": star,
            "radar": radar,
            "power": power,
            "magnetometer": mag,
            "sun_sensor": sun,
            "proximity": prox,
            "thermal": therm,
            "doppler": doppler,
            "radiation": rad,
            "spectrometer": spec,
            "accelerometer": accel,
            "imu": imu,
            "lidar": lidar,
            "timestamp": timestamp,

            # Convenience dicts for SensorFusion / MissionController
            "gps_data": {
                "position": gps.position if gps.fix_valid else spacecraft_position,
                "velocity": gps.velocity if gps.fix_valid else spacecraft_velocity,
            },
            "imu_data": {
                "acceleration": imu.acceleration,
                "angular_velocity": imu.angular_velocity,
            },
            "star_tracker_data": {
                "attitude": star.attitude,
            },

            # Flat ML vector
            "telemetry_vector": self.to_telemetry_vector(
                gps, star, radar, power, mag, sun, prox,
                therm, doppler, rad, spec, accel, imu,
            ),
        }
        return result

    # ------------------------------------------------------------------
    # Flat feature vector for ML
    # ------------------------------------------------------------------

    @staticmethod
    def to_telemetry_vector(
        gps: GPSReading,
        star: StarTrackerReading,
        radar: RadarReading,
        power: PowerReading,
        mag: MagnetometerReading,
        sun: SunSensorReading,
        prox: ProximityReading,
        therm: ThermalReading,
        doppler: DopplerReading,
        rad: RadiationReading,
        spec: SpectrometerReading,
        accel: AccelerometerReading,
        imu: IMUReading,
    ) -> np.ndarray:
        """Build a fixed-size telemetry vector from all sensor readings."""
        parts = []

        # GPS (6) — normalised to ~[-1, 1] range
        pos = gps.position / 7000.0 if gps.fix_valid else np.zeros(3)
        vel = gps.velocity / 8.0 if gps.fix_valid else np.zeros(3)
        parts.append(np.nan_to_num(pos))
        parts.append(np.nan_to_num(vel))

        # Star tracker (7) — quaternion already in [-1,1], angular rate normalised
        parts.append(star.attitude)           # 4
        parts.append(star.angular_rate / 1.0) # 3, rad/s (already small)

        # Radar features (64)
        parts.append(radar.features_vector)

        # Power (7)
        parts.append(np.array([
            power.battery_soc,
            power.solar_power / 1000.0,       # kW
            power.bus_voltage / 30.0,          # normalised
            power.current_draw / 10.0,
            power.power_balance / 1000.0,
            float(power.eclipse_state),
            power.battery_temperature / 100.0,
        ]))

        # Magnetometer (4)
        parts.append(mag.magnetic_field / 100.0)  # 3, normalised µT
        parts.append(np.array([mag.field_magnitude / 100.0]))

        # Sun sensor (5)
        parts.append(sun.sun_vector)          # 3
        parts.append(np.array([float(sun.sun_visible), sun.intensity / 1400.0]))

        # Proximity (5)
        r_norm = min(prox.range_to_target, 50.0) / 50.0 if prox.target_detected else 1.0
        parts.append(np.array([
            r_norm,
            prox.closing_velocity / 10.0,
        ]))
        parts.append(prox.bearing)            # 2
        parts.append(np.array([float(prox.target_detected)]))

        # Thermal (12)
        parts.append(np.array([therm.temperatures[n] / 100.0 for n in COMPONENT_NAMES]))  # 10
        parts.append(np.array([therm.mean_temperature / 100.0, therm.max_temperature / 100.0]))

        # Doppler (3)
        parts.append(np.array([
            doppler.radial_velocity / 100.0,
            doppler.closing_rate / 100.0,
            float(doppler.target_detected),
        ]))

        # Radiation (4)
        parts.append(np.array([
            rad.dose_rate,
            rad.particle_flux / 1e5,
            float(rad.solar_event),
            float(rad.in_saa),
        ]))

        # Spectrometer (33)
        parts.append(spec.spectrum)            # 32
        parts.append(np.array([spec.material_confidence]))

        # Accelerometer (4)
        parts.append(accel.acceleration * 1e3) # 3, scale up micro-g
        parts.append(np.array([float(accel.shock_detected)]))

        # IMU (6) — normalised
        parts.append(imu.acceleration / 10.0)         # 3, m/s² -> normalised
        parts.append(imu.angular_velocity / 1.0)      # 3, rad/s (already small)

        vec = np.concatenate(parts)
        vec = np.nan_to_num(vec, nan=0.0, posinf=10.0, neginf=-10.0)
        vec = np.clip(vec, -10.0, 10.0)
        vec = vec.astype(np.float32)

        # Safety: ensure length matches TELEMETRY_DIM (pad / truncate)
        if len(vec) < TELEMETRY_DIM:
            vec = np.concatenate([vec, np.zeros(TELEMETRY_DIM - len(vec), dtype=np.float32)])
        elif len(vec) > TELEMETRY_DIM:
            vec = vec[:TELEMETRY_DIM]

        return vec
