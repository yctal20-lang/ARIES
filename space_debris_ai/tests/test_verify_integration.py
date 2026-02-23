"""
Comprehensive integration verification script.
Verifies all modules integrate properly with MissionController.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from space_debris_ai.core import SystemConfig
from space_debris_ai.inference import MissionController
from space_debris_ai.models import (
    CollisionAvoidanceModule,
    NavigationModule,
    AnomalyDetector,
    EnergyManagementModule,
    StatePredictionModule,
    AttentionWarningSystem,
    SensorFilterModule,
    FailurePredictionModule,
    DebrisRecognitionModule,
    ManipulatorModule,
    TrackingModule,
    PrecisionManeuveringModule,
    RiskAssessor,
)


def verify_module_integration():
    """Verify all modules can be integrated with MissionController."""
    print("=" * 60)
    print("Integration Verification")
    print("=" * 60)
    
    config = SystemConfig()
    controller = MissionController(config)
    
    modules_registered = 0
    modules_failed = []
    
    # Level 1: Survival
    print("\n[Level 1] Registering Survival Modules...")
    
    try:
        nav_module = NavigationModule(config={}, device="cpu")
        controller.register_module("navigation", nav_module, {"timeout": 1.0})
        modules_registered += 1
        print("  ✓ Navigation module registered")
    except Exception as e:
        print(f"  ✗ Navigation module failed: {e}")
        modules_failed.append(("navigation", str(e)))
    
    try:
        col_module = CollisionAvoidanceModule(
            config={"lidar_points": 256},
            device="cpu",
        )
        controller.register_module("collision_avoidance", col_module, {"timeout": 0.5})
        modules_registered += 1
        print("  ✓ Collision avoidance module registered")
    except Exception as e:
        print(f"  ✗ Collision avoidance module failed: {e}")
        modules_failed.append(("collision_avoidance", str(e)))
    
    # Level 2: Safety
    print("\n[Level 2] Registering Safety Modules...")
    
    try:
        anomaly_module = AnomalyDetector(
            config={"input_dim": 12, "seq_len": 50},
            device="cpu",
        )
        controller.register_module("anomaly_detection", anomaly_module, {"timeout": 2.0})
        modules_registered += 1
        print("  ✓ Anomaly detection module registered")
    except Exception as e:
        print(f"  ✗ Anomaly detection module failed: {e}")
        modules_failed.append(("anomaly_detection", str(e)))
    
    try:
        energy_module = EnergyManagementModule(config={}, device="cpu")
        controller.register_module("energy_management", energy_module, {"timeout": 1.0})
        modules_registered += 1
        print("  ✓ Energy management module registered")
    except Exception as e:
        print(f"  ✗ Energy management module failed: {e}")
        modules_failed.append(("energy_management", str(e)))
    
    # Level 3: Mission Critical
    print("\n[Level 3] Registering Mission Critical Modules...")
    
    try:
        state_module = StatePredictionModule(
            config={"input_dim": 13, "output_dim": 10, "prediction_horizon": 10},
            device="cpu",
        )
        controller.register_module("state_prediction", state_module, {"timeout": 2.0})
        modules_registered += 1
        print("  ✓ State prediction module registered")
    except Exception as e:
        print(f"  ✗ State prediction module failed: {e}")
        modules_failed.append(("state_prediction", str(e)))
    
    try:
        warning_module = AttentionWarningSystem(
            input_dim=32,
            embed_dim=64,
            num_categories=5,
            num_heads=4,
            seq_len=20,
        )
        controller.register_module("early_warning", warning_module, {"timeout": 2.0})
        modules_registered += 1
        print("  ✓ Early warning module registered")
    except Exception as e:
        print(f"  ✗ Early warning module failed: {e}")
        modules_failed.append(("early_warning", str(e)))
    
    try:
        filter_module = SensorFilterModule(
            config={"input_dim": 12, "seq_len": 10},
            device="cpu",
        )
        controller.register_module("sensor_filter", filter_module, {"timeout": 1.0})
        modules_registered += 1
        print("  ✓ Sensor filter module registered")
    except Exception as e:
        print(f"  ✗ Sensor filter module failed: {e}")
        modules_failed.append(("sensor_filter", str(e)))
    
    try:
        failure_module = FailurePredictionModule(
            config={"input_dim": 20, "horizon": 48},
            device="cpu",
        )
        controller.register_module("failure_prediction", failure_module, {"timeout": 2.0})
        modules_registered += 1
        print("  ✓ Failure prediction module registered")
    except Exception as e:
        print(f"  ✗ Failure prediction module failed: {e}")
        modules_failed.append(("failure_prediction", str(e)))
    
    # Level 4: Mission Execution
    print("\n[Level 4] Registering Mission Execution Modules...")
    
    try:
        debris_module = DebrisRecognitionModule(
            config={"num_classes": 5},
            device="cpu",
        )
        controller.register_module("debris_recognition", debris_module, {"timeout": 2.0})
        modules_registered += 1
        print("  ✓ Debris recognition module registered")
    except Exception as e:
        print(f"  ✗ Debris recognition module failed: {e}")
        modules_failed.append(("debris_recognition", str(e)))
    
    try:
        manip_module = ManipulatorModule(
            config={"num_joints": 6},
            device="cpu",
        )
        controller.register_module("manipulator_control", manip_module, {"timeout": 1.0})
        modules_registered += 1
        print("  ✓ Manipulator control module registered")
    except Exception as e:
        print(f"  ✗ Manipulator control module failed: {e}")
        modules_failed.append(("manipulator_control", str(e)))
    
    try:
        track_module = TrackingModule(
            config={"input_dim": 6, "num_queries": 10},
            device="cpu",
        )
        controller.register_module("object_tracking", track_module, {"timeout": 2.0})
        modules_registered += 1
        print("  ✓ Object tracking module registered")
    except Exception as e:
        print(f"  ✗ Object tracking module failed: {e}")
        modules_failed.append(("object_tracking", str(e)))
    
    try:
        mpc_module = PrecisionManeuveringModule(
            config={"state_dim": 13, "action_dim": 6, "horizon": 20},
            device="cpu",
        )
        controller.register_module("precision_maneuvering", mpc_module, {"timeout": 2.0})
        modules_registered += 1
        print("  ✓ Precision maneuvering module registered")
    except Exception as e:
        print(f"  ✗ Precision maneuvering module failed: {e}")
        modules_failed.append(("precision_maneuvering", str(e)))
    
    # Test controller lifecycle
    print("\n[Controller] Testing lifecycle...")
    try:
        controller.start()
        print("  ✓ Controller started")
        
        # Test step
        sensor_data = {
            "position": np.array([7000.0, 0.0, 0.0]),
            "velocity": np.array([0.0, 7.5, 0.0]),
            "attitude": np.array([1.0, 0.0, 0.0, 0.0]),
            "gps_position": np.array([7000.0, 0.0, 0.0]),
            "gps_velocity": np.array([0.0, 7.5, 0.0]),
            "imu_accel": np.random.randn(3),
            "imu_gyro": np.random.randn(3),
            "star_tracker": np.array([1.0, 0.0, 0.0, 0.0]),
            "lidar": np.random.randn(256, 6).astype(np.float32),
            "radar": np.random.randn(64).astype(np.float32),
            "telemetry": np.random.randn(50, 12).astype(np.float32),
            "battery_soc": 0.8,
            "solar_power": 1000.0,
            "power_requests": np.array([100, 200, 150, 50]),
        }
        
        result = controller.step(sensor_data, dt=0.1)
        print("  ✓ Controller step executed")
        assert "commands" in result
        assert "results" in result
        assert "state" in result
        
        controller.stop()
        print("  ✓ Controller stopped")
    except Exception as e:
        print(f"  ✗ Controller lifecycle failed: {e}")
        modules_failed.append(("controller_lifecycle", str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("Integration Summary")
    print("=" * 60)
    print(f"Modules registered: {modules_registered}")
    print(f"Modules failed: {len(modules_failed)}")
    
    if modules_failed:
        print("\nFailed modules:")
        for name, error in modules_failed:
            print(f"  - {name}: {error}")
    
    print("\n" + "=" * 60)
    
    return len(modules_failed) == 0


if __name__ == "__main__":
    success = verify_module_integration()
    sys.exit(0 if success else 1)

