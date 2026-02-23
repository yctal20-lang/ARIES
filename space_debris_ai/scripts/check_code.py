"""
Full system check: flight data, physics, EKF, modules, safety.
Run: python scripts/check_code.py  (from space_debris_ai/)
"""

import sys
import os
import time
import traceback
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS = "PASS"
FAIL = "FAIL"
results = []


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    results.append((name, ok))
    line = f"  [{tag}] {name}"
    if detail:
        line += f"  -- {detail}"
    print(line)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------
# 1. Flight data
# ---------------------------------------------------------------
def check_flight_data():
    section("1. Flight data (t=12.5s)")
    from data.flight_data_12_5s import FLIGHT_DATA as d

    report("data loaded", d is not None)
    report("mass = 1420 kg", d.mass == 1420.0, f"got {d.mass}")
    report("fuel_mass = 12 kg", d.fuel_mass == 12.0, f"got {d.fuel_mass}")
    report("fuel_flow = 18.0 kg/s", d.fuel_flow == 18.0, f"got {d.fuel_flow}")

    # Fuel warning
    burn_time = d.fuel_mass / d.fuel_flow if d.fuel_flow > 0 else float("inf")
    is_critical = burn_time < 1.0
    report(
        "fuel burn time check",
        True,
        f"burn_time = {burn_time:.2f} s {'(CRITICAL: < 1 s!)' if is_critical else ''}",
    )

    pos = d.position_km()
    vel = d.velocity_km_s()
    att = d.attitude_quaternion()
    report("position_km shape", pos.shape == (3,), str(pos))
    report("velocity_km_s shape", vel.shape == (3,), str(vel))
    report("quaternion shape", att.shape == (4,), str(att))
    report("quaternion norm ~ 1", abs(np.linalg.norm(att) - 1.0) < 1e-6)

    sd = d.sensor_data()
    expected_keys = ["position", "velocity", "attitude", "acceleration",
                     "angular_velocity", "altitude", "dt", "timestamp"]
    report("sensor_data keys", all(k in sd for k in expected_keys))
    return d


# ---------------------------------------------------------------
# 2. Physics F=ma
# ---------------------------------------------------------------
def check_physics(d):
    section("2. Physics F=ma")
    thrust = np.array([d.Tx, d.Ty, d.Tz])
    drag = np.array([d.Dx, d.Dy, d.Dz])
    gravity = np.array([0, 0, d.gravity_force])
    F_total = thrust + drag + gravity
    a_calc = F_total / d.mass
    a_given = np.array([d.ax, d.ay, d.az])
    err = np.linalg.norm(a_calc - a_given)
    report("F_total (N)", True, str(F_total))
    report("a_calc  (m/s^2)", True, str(np.round(a_calc, 4)))
    report("a_given (m/s^2)", True, str(a_given))
    report("|a_calc - a_given|", err < 5.0, f"{err:.4f}")

    # Thrust magnitude
    t_mag = np.linalg.norm(thrust)
    report(
        "thrust vector magnitude ~ thrust_total",
        abs(t_mag - d.thrust_total) < 500,
        f"|T| = {t_mag:.1f}, total = {d.thrust_total}",
    )

    # Drag sanity: D = 0.5 * rho * Cd * A * v^2
    v = np.linalg.norm([d.vx, d.vy, d.vz])
    D_est = 0.5 * d.air_density * d.drag_coefficient * d.reference_area * v**2
    D_given = np.linalg.norm([d.Dx, d.Dy, d.Dz])
    report(
        "drag magnitude sanity",
        True,
        f"estimated={D_est:.1f} N, given={D_given:.1f} N",
    )


# ---------------------------------------------------------------
# 3. EKF Navigation
# ---------------------------------------------------------------
def check_ekf(d):
    section("3. EKF Navigation")
    from models.level1_survival.navigation import ExtendedKalmanFilter

    ekf = ExtendedKalmanFilter(dt=d.dt)
    ekf.initialize(d.position_km(), d.velocity_km_s(), d.attitude_quaternion(), d.t)
    state0 = ekf.get_state()
    report("EKF initialized", state0.position is not None)

    state = ekf.predict(
        dt=d.dt,
        acceleration=d.accelerometer_ms2(),
        angular_rate=d.gyro_rad_s(),
    )
    report("EKF predict OK", state.position is not None)

    state = ekf.update_gps(d.position_km(), d.velocity_km_s())
    report("EKF update_gps OK", state.position is not None)
    report("EKF position (km)", True, str(np.round(state.position, 6)))
    report("EKF velocity (km/s)", True, str(np.round(state.velocity, 6)))


# ---------------------------------------------------------------
# 4. Collision Detector
# ---------------------------------------------------------------
def check_collision_detector():
    section("4. Collision Detector")
    import torch
    from models.level1_survival.collision_avoidance import CollisionDetector

    detector = CollisionDetector(lidar_points=128, radar_features=64, imu_features=12)
    lidar = torch.randn(1, 128, 6)
    radar = torch.randn(1, 64)
    imu = torch.randn(1, 12)
    with torch.no_grad():
        out = detector(lidar, radar, imu)
    report("collision_probability present", "collision_probability" in out)
    report("time_to_collision present", "time_to_collision" in out)
    if "collision_probability" in out:
        p = out["collision_probability"].item()
        report("probability in [0,1]", 0 <= p <= 1, f"{p:.4f}")


# ---------------------------------------------------------------
# 5. Anomaly Detection
# ---------------------------------------------------------------
def check_anomaly_detection():
    section("5. Anomaly Detection")
    import torch
    from models.level2_safety.anomaly_detection import LSTMAutoencoder

    model = LSTMAutoencoder(input_dim=16, hidden_dim=32, num_layers=2, seq_len=50)
    seq = torch.randn(1, 50, 16)
    with torch.no_grad():
        reconstructed, latent = model(seq)
    report("reconstructed shape", reconstructed.shape == seq.shape)
    report("latent shape", latent.shape[0] == 1 and latent.dim() == 2)


# ---------------------------------------------------------------
# 6. Debris Recognizer
# ---------------------------------------------------------------
def check_debris_recognizer():
    section("6. Debris Recognizer")
    import torch
    from models.level4_mission_execution.debris_recognition import DebrisRecognizer

    rec = DebrisRecognizer(num_classes=5, visual_dim=256, radar_dim=128, lidar_dim=128)
    vis = torch.randn(1, 3, 224, 224)
    rad = torch.randn(1, 128)
    lid = torch.randn(1, 256, 3)
    with torch.no_grad():
        out = rec(vis, rad, lid)
    report("class_logits present", "class_logits" in out)
    report("estimated_size present", "estimated_size" in out)
    report("class_logits shape", out["class_logits"].shape == (1, 5))


# ---------------------------------------------------------------
# 7. Safety: Watchdog + Failsafe
# ---------------------------------------------------------------
def check_safety():
    section("7. Safety (Watchdog + Failsafe)")
    from safety.watchdog import Watchdog, WatchdogState
    from safety.failsafe import FailsafeController, FallbackMode

    triggered = []
    wd = Watchdog("test", timeout=0.1, on_timeout=lambda: triggered.append(True))
    wd.start()
    wd.feed()
    report("watchdog fed OK", wd.check())

    time.sleep(0.15)
    wd.check()
    report("watchdog triggered", wd.state == WatchdogState.TRIGGERED)

    fc = FailsafeController({"max_consecutive_failures": 2})
    fc.register_module("m1")
    fc.report_failure("m1", "err1")
    report("failsafe after 1 failure: NORMAL", fc.mode == FallbackMode.NORMAL)
    fc.report_failure("m1", "err2")
    report("failsafe after 2 failures: degraded", fc.mode != FallbackMode.NORMAL)


# ---------------------------------------------------------------
# 8. Simulation Environment
# ---------------------------------------------------------------
def check_simulation():
    section("8. Simulation Environment")
    from simulation.environment import OrbitalEnv

    env = OrbitalEnv()
    obs, info = env.reset()
    report("env reset OK", obs is not None)
    report("obs shape", len(obs) > 0, f"len={len(obs)}")

    action = env.action_space.sample()
    obs2, reward, done, trunc, info2 = env.step(action)
    report("env step OK", obs2 is not None)
    report("reward is float", isinstance(reward, (int, float, np.floating)))
    env.close()
    report("env closed", True)


# ---------------------------------------------------------------
# 9. Config
# ---------------------------------------------------------------
def check_config():
    section("9. Config")
    from core.config import SystemConfig, MissionMode

    cfg = SystemConfig()
    report("config created", cfg is not None)
    report("default mode = BALANCED", cfg.mission_mode == MissionMode.BALANCED)
    report("spacecraft mass > 0", cfg.spacecraft.mass > 0, f"{cfg.spacecraft.mass}")


# ---------------------------------------------------------------
# 10. Message Bus
# ---------------------------------------------------------------
def check_message_bus():
    section("10. Message Bus")
    from core.message_bus import MessageBus, MessageType

    bus = MessageBus()
    received = []
    bus.subscribe("test", lambda m: received.append(m), [MessageType.TELEMETRY])
    bus.start()
    bus.publish_sync(MessageType.TELEMETRY, "test_src", {"val": 42})
    bus.stop()
    report("message received", len(received) == 1)
    if received:
        payload = received[0].payload
        ok = payload.get("val") == 42 if isinstance(payload, dict) else False
        report("payload correct", ok)
    else:
        report("payload correct", False)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    print()
    print("  SPACE DEBRIS AI  --  FULL SYSTEM CHECK")
    print(f"  Python {sys.version.split()[0]}")
    t0 = time.time()

    try:
        d = check_flight_data()
    except Exception:
        traceback.print_exc()
        d = None

    checks = [
        lambda: check_physics(d) if d else None,
        lambda: check_ekf(d) if d else None,
        check_collision_detector,
        check_anomaly_detection,
        check_debris_recognizer,
        check_safety,
        check_simulation,
        check_config,
        check_message_bus,
    ]

    for fn in checks:
        try:
            fn()
        except Exception:
            traceback.print_exc()

    elapsed = time.time() - t0

    section("SUMMARY")
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    total = len(results)
    print(f"  {passed}/{total} passed, {failed} failed  ({elapsed:.1f} s)")
    if failed:
        print("\n  Failed checks:")
        for name, ok in results:
            if not ok:
                print(f"    - {name}")
    else:
        print("\n  All checks passed!")
    print()


if __name__ == "__main__":
    main()
