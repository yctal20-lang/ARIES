"""
Flight data snapshot at t = 12.5 s for verification.
Units: SI (m, m/s, m/s², N, kg) unless noted.
Converts to project units (km, km/s) and provides sensor_data for EKF/controller.

Reference snapshot:
  Время: t = 12.5 с, dt = 0.02 с
  Положение (м): x=184.3, y=-27.6, z=912.8
  Скорость (м/с): vx=18.4, vy=-2.1, vz=96.7
  Ускорение (м/с²): ax=1.2, ay=-0.3, az=6.8
  Ориентация (°): roll=2.4, pitch=87.1, yaw=15.6
  Масса: 1420 кг, fuel_mass=12 кг (почти пусто), fuel_flow=18.0 кг/с (утечка/ошибка)
  Тяга: 32 000 Н, Tx=8200, Ty=-1500, Tz=30600 Н
  Аэродинамика: ρ=0.38, Cd=0.75, S=14.5 м², Dx=-620, Dy=90, Dz=-2100 Н
  gravity_force = -17 440 Н
  Датчики: altimeter=913.1 м, gyro (rad/s), accelerometer (м/с²)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any


def euler_deg_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Convert Euler angles (roll, pitch, yaw) in degrees to quaternion [w, x, y, z]."""
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


@dataclass
class FlightData12_5s:
    """Raw flight data at t = 12.5 s (SI units)."""

    # Время
    t: float = 12.5
    dt: float = 0.02

    # Положение (м, мировая СК)
    x: float = 184.3
    y: float = -27.6
    z: float = 912.8

    # Скорость (м/с)
    vx: float = 18.4
    vy: float = -2.1
    vz: float = 96.7

    # Ускорение (м/с²)
    ax: float = 1.2
    ay: float = -0.3
    az: float = 6.8

    # Ориентация (углы Эйлера, градусы: крен, тангаж, рысканье)
    roll_deg: float = 2.4
    pitch_deg: float = 87.1
    yaw_deg: float = 15.6

    # Массо-инерционные: 1420 кг, топливо 12 кг (почти пусто), fuel_flow 18 кг/с (утечка/ошибка)
    mass: float = 1420.0
    fuel_mass: float = 12.0
    fuel_flow: float = 18.0

    # Двигательная установка (Н)
    thrust_total: float = 32000.0
    Tx: float = 8200.0
    Ty: float = -1500.0
    Tz: float = 30600.0

    # Аэродинамика: ρ кг/м³, Cd, S м², сила сопротивления (Н)
    air_density: float = 0.38
    drag_coefficient: float = 0.75
    reference_area: float = 14.5
    Dx: float = -620.0
    Dy: float = 90.0
    Dz: float = -2100.0

    # Внешние силы (Н)
    gravity_force: float = -17440.0

    # Датчики: высотомер (м), гиро (рад/с), акселерометр (м/с²)
    altimeter_m: float = 913.1
    gyro_x: float = 0.02
    gyro_y: float = -0.01
    gyro_z: float = 0.12
    accel_sens_x: float = 1.18
    accel_sens_y: float = -0.29
    accel_sens_z: float = 6.74

    def position_km(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64) / 1000.0

    def velocity_km_s(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz], dtype=np.float64) / 1000.0

    def attitude_quaternion(self) -> np.ndarray:
        return euler_deg_to_quaternion(self.roll_deg, self.pitch_deg, self.yaw_deg)

    def gyro_rad_s(self) -> np.ndarray:
        return np.array([self.gyro_x, self.gyro_y, self.gyro_z], dtype=np.float64)

    def accelerometer_ms2(self) -> np.ndarray:
        return np.array([self.accel_sens_x, self.accel_sens_y, self.accel_sens_z], dtype=np.float64)

    def sensor_data(self) -> Dict[str, Any]:
        """Format for MissionController / EKF."""
        return {
            "position": self.position_km(),
            "velocity": self.velocity_km_s(),
            "attitude": self.attitude_quaternion(),
            "acceleration": self.accelerometer_ms2(),
            "angular_velocity": self.gyro_rad_s(),
            "altitude": self.altimeter_m,
            "dt": self.dt,
            "timestamp": self.t,
        }


FLIGHT_DATA = FlightData12_5s()
