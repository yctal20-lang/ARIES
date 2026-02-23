"""
Verify flight data at t=12.5s: run EKF and basic physics checks.
Run from space_debris_ai: python scripts/verify_flight_data.py
"""

import sys
import os
import numpy as np

# Project root = parent of scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.flight_data_12_5s import FLIGHT_DATA


def check_f_ma():
    """F = ma: (thrust + drag + gravity_z) / m ~ acceleration."""
    d = FLIGHT_DATA
    thrust = np.array([d.Tx, d.Ty, d.Tz])
    drag = np.array([d.Dx, d.Dy, d.Dz])
    gravity_vec = np.array([0, 0, d.gravity_force])
    F = thrust + drag + gravity_vec
    m = d.mass
    a_expected = F / m
    a_given = np.array([d.ax, d.ay, d.az])
    print("Physics F=ma (world frame; gravity only on Z):")
    print("  F_total (N):    ", F)
    print("  a_expected (m/s^2):", a_expected)
    print("  a_given (m/s^2):   ", a_given)
    err = np.linalg.norm(a_expected - a_given)
    print("  |difference|:     ", round(err, 6))
    return err < 1.0


def run_ekf():
    """Initialize EKF with flight data, predict, then update with GPS-like data."""
    from models.level1_survival.navigation import ExtendedKalmanFilter

    d = FLIGHT_DATA
    ekf = ExtendedKalmanFilter(dt=d.dt)
    ekf.initialize(d.position_km(), d.velocity_km_s(), d.attitude_quaternion(), d.t)
    state = ekf.predict(
        dt=d.dt,
        acceleration=d.accelerometer_ms2(),
        angular_rate=d.gyro_rad_s(),
    )
    state = ekf.update_gps(d.position_km(), d.velocity_km_s())
    print("EKF:")
    print("  position (km):", state.position)
    print("  velocity (km/s):", state.velocity)
    return state


def main():
    print("=== Flight data t = 12.5 s ===\n")
    ok_f = check_f_ma()
    print()
    state = run_ekf()
    print()
    print("sensor_data keys:", list(FLIGHT_DATA.sensor_data().keys()))
    print("F=ma check:", "PASS" if ok_f else "FAIL (review frame/convention)")
    print("EKF run: OK")


if __name__ == "__main__":
    main()
