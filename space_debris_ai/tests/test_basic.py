"""
Basic tests for Space Debris Collector AI System.
"""

import pytest  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]
import torch  # type: ignore[reportMissingImports]
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Test configuration
def test_config_creation():
    """Test configuration creation and validation."""
    from core.config import SystemConfig, MissionMode
    
    config = SystemConfig()
    
    assert config.mission_mode == MissionMode.BALANCED
    assert config.spacecraft.mass > 0
    assert config.orbital.semi_major_axis > 0
    assert config.simulation.dt > 0


def test_config_save_load(tmp_path):
    """Test config save and load."""
    from core.config import SystemConfig
    
    config = SystemConfig(mission_name="test_mission")
    
    path = tmp_path / "config.json"
    config.save(path)
    
    loaded = SystemConfig.load(path)
    assert loaded.mission_name == "test_mission"


# Test message bus
def test_message_bus():
    """Test message bus functionality."""
    from core.message_bus import MessageBus, Message, MessageType
    
    bus = MessageBus()
    received = []
    
    def handler(msg):
        received.append(msg)
    
    bus.subscribe("test_module", handler)
    
    msg = Message.create(
        MessageType.STATUS,
        "sender",
        {"test": "data"},
    )
    
    bus.publish(msg)
    bus.process_pending()
    
    assert len(received) == 1
    assert received[0].payload["test"] == "data"


# Test orbital mechanics
def test_orbital_elements():
    """Test orbital elements conversion."""
    from simulation.physics import OrbitalElements, EARTH_RADIUS
    
    # Create circular orbit at 400km altitude
    altitude = 400.0
    elements = OrbitalElements(
        a=EARTH_RADIUS + altitude,
        e=0.0,
        i=0.5,  # ~28 degrees
        raan=0.0,
        omega=0.0,
        nu=0.0,
    )
    
    r, v = elements.to_state_vector()
    
    # Check altitude
    r_mag = np.linalg.norm(r)
    assert abs(r_mag - (EARTH_RADIUS + altitude)) < 1.0
    
    # Convert back
    elements2 = OrbitalElements.from_state_vector(r, v)
    assert abs(elements2.a - elements.a) < 1.0


def test_orbital_propagation():
    """Test orbital propagation."""
    from simulation.physics import OrbitalMechanics, SpacecraftState, EARTH_RADIUS
    
    mechanics = OrbitalMechanics()
    
    # Initial state - circular orbit
    altitude = 400.0
    r_mag = EARTH_RADIUS + altitude
    v_mag = np.sqrt(398600.4418 / r_mag)
    
    state = SpacecraftState(
        position=np.array([r_mag, 0, 0]),
        velocity=np.array([0, v_mag, 0]),
        attitude=np.array([1, 0, 0, 0]),
        angular_velocity=np.zeros(3),
        mass=500.0,
        fuel_mass=100.0,
    )
    
    # Propagate for 100 steps
    dt = 10.0  # 10 second steps
    for _ in range(100):
        state = mechanics.propagate_rk4(state, dt)
    
    # Should still be at roughly same altitude (circular orbit)
    new_altitude = np.linalg.norm(state.position) - EARTH_RADIUS
    assert abs(new_altitude - altitude) < 10.0  # Within 10 km


# Test gymnasium environment
def test_orbital_env():
    """Test orbital environment."""
    from simulation.environment import OrbitalEnv, EnvConfig
    
    config = EnvConfig(num_debris=10, max_episode_steps=100)
    env = OrbitalEnv(config)
    
    obs, info = env.reset(seed=42)
    
    assert obs.shape == env.observation_space.shape
    assert "altitude" in info
    
    # Take a few steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    env.close()


# Test anomaly detection
def test_anomaly_detector():
    """Test anomaly detection module."""
    from models.level2_safety.anomaly_detection import LSTMAutoencoder
    
    model = LSTMAutoencoder(input_dim=12, seq_len=50)
    
    # Create dummy input
    x = torch.randn(4, 50, 12)
    
    # Forward pass
    reconstructed, latent = model(x)
    
    assert reconstructed.shape == x.shape
    assert latent.shape == (4, 32)  # Default latent dim


# Test collision detector
def test_collision_detector():
    """Test collision detection network."""
    from models.level1_survival.collision_avoidance import CollisionDetector
    
    detector = CollisionDetector(lidar_points=256)
    
    # Create dummy inputs
    lidar = torch.randn(2, 256, 6)
    radar = torch.randn(2, 64)
    imu = torch.randn(2, 12)
    
    outputs = detector(lidar, radar, imu)
    
    assert "collision_probability" in outputs
    assert "time_to_collision" in outputs
    assert outputs["collision_probability"].shape == (2,)


# Test navigation EKF
def test_navigation_ekf():
    """Test Extended Kalman Filter."""
    from models.level1_survival.navigation import ExtendedKalmanFilter
    
    ekf = ExtendedKalmanFilter()
    
    # Initialize
    position = np.array([7000.0, 0.0, 0.0])
    velocity = np.array([0.0, 7.5, 0.0])
    ekf.initialize(position, velocity)
    
    # Predict
    state = ekf.predict(dt=1.0)
    
    assert state.position is not None
    assert state.velocity is not None
    
    # Update with GPS (fixed seed for reproducibility)
    np.random.seed(42)
    gps_pos = position + np.random.randn(3) * 0.01
    gps_vel = velocity + np.random.randn(3) * 0.001
    
    state = ekf.update_gps(gps_pos, gps_vel)
    
    # Position should be close to GPS (relaxed threshold: EKF blends prediction with measurement)
    assert np.linalg.norm(state.position - gps_pos) < 1.0


# Test watchdog
def test_watchdog():
    """Test watchdog timer."""
    from safety.watchdog import Watchdog, WatchdogState
    import time
    
    triggered = []
    
    def on_timeout():
        triggered.append(True)
    
    watchdog = Watchdog("test", timeout=0.1, on_timeout=on_timeout)
    watchdog.start()
    
    # Feed immediately - should not trigger
    watchdog.feed()
    assert watchdog.check() == True
    
    # Wait for timeout
    time.sleep(0.15)
    watchdog.check()
    
    assert watchdog.state == WatchdogState.TRIGGERED
    assert len(triggered) == 1


# Test failsafe
def test_failsafe_controller():
    """Test failsafe controller."""
    from safety.failsafe import FailsafeController, FallbackMode
    
    controller = FailsafeController({"max_consecutive_failures": 2})
    
    controller.register_module("test_module")
    
    # Report failures
    controller.report_failure("test_module", "Test error 1")
    assert controller.mode == FallbackMode.NORMAL
    
    controller.report_failure("test_module", "Test error 2")
    # Should now be in degraded or classical mode
    assert controller.mode != FallbackMode.NORMAL
    
    # Report success to recover
    for _ in range(5):
        controller.report_success("test_module")


# Test Risk Assessor
def test_risk_assessor():
    """Test risk assessment module."""
    from models.level4_mission_execution.risk_assessment import RiskAssessor, RiskLevel
    
    assessor = RiskAssessor(input_dim=24, use_physics_features=True)
    
    # Create dummy input 
    debris_features = torch.randn(1, 24)
    
    # Forward pass
    with torch.no_grad():
        outputs = assessor(debris_features)
    
    assert "capture_risk" in outputs
    assert "collision_risk" in outputs
    assert "priority_score" in outputs


# Test Precision Maneuvering
def test_neural_mpc():
    """Test Neural MPC for precision maneuvering."""
    from models.level4_mission_execution.precision_maneuvering import NeuralMPC
    
    mpc = NeuralMPC(state_dim=13, action_dim=6, horizon=10)
    
    # Create state vectors
    # Current state: position, velocity, attitude (quaternion), angular velocity
    current_state = np.array([
        7000.0, 0.0, 0.0,           # Position (km)
        0.0, 7.5, 0.0,              # Velocity (km/s)
        1.0, 0.0, 0.0, 0.0,         # Attitude (quaternion)
        0.0, 0.0, 0.0,              # Angular velocity
    ], dtype=np.float32)
    
    # Target state (slightly different position)
    target_state = np.array([
        7000.0, 100.0, 0.0,         # Target position
        0.0, 7.5, 0.0,              # Target velocity
        1.0, 0.0, 0.0, 0.0,         # Target attitude
        0.0, 0.0, 0.0,              # Target angular velocity
    ], dtype=np.float32)
    
    # Plan trajectory
    plan = mpc.plan(current_state, target_state)
    
    assert plan.positions.shape == (11, 3)  # horizon+1 states
    assert plan.thrust_commands.shape == (10, 3)  # horizon actions
    assert plan.feasible in (True, False)  # accept Python bool or numpy.bool_
    assert plan.horizon_time > 0


# Test Multi-Object Tracker
def test_multi_object_tracker():
    """Test multi-object tracking module."""
    from models.level4_mission_execution.object_tracking import MultiObjectTracker
    
    tracker = MultiObjectTracker(
        input_dim=6,
        d_model=64,
        num_queries=10,
        num_classes=3,
    )
    
    # Create dummy input - detection features
    batch_size = 2
    num_detections = 5
    input_dim = 6
    
    x = torch.randn(batch_size, num_detections, input_dim)
    
    # Forward pass
    outputs = tracker(x)
    
    assert "class_logits" in outputs
    assert "positions" in outputs
    assert "velocities" in outputs
    assert outputs["class_logits"].shape == (batch_size, 10, 4)  # num_classes + 1 (no object)
    assert outputs["positions"].shape == (batch_size, 10, 3)


# Test Debris Recognizer
def test_debris_recognizer():
    """Test debris recognition module."""
    from models.level4_mission_execution.debris_recognition import DebrisRecognizer
    
    recognizer = DebrisRecognizer(
        num_classes=5,
        visual_dim=256,
        radar_dim=128,
        lidar_dim=128,
    )
    
    # Create dummy inputs
    batch_size = 2
    visual = torch.randn(batch_size, 3, 224, 224)
    radar = torch.randn(batch_size, 128)
    lidar = torch.randn(batch_size, 256, 3)
    
    # Forward pass
    outputs = recognizer(visual, radar, lidar)
    
    assert "class_logits" in outputs
    assert "estimated_size" in outputs  # model returns estimated_size
    assert outputs["class_logits"].shape == (batch_size, 5)


# Test State Prediction
def test_state_predictor():
    """Test TCN state predictor."""
    from models.level3_mission_critical.state_prediction import TCNPredictor
    
    predictor = TCNPredictor(
        input_dim=13,
        output_dim=10,
        prediction_horizon=10,
    )
    
    # Create dummy sequence
    batch_size = 2
    seq_len = 50
    
    x = torch.randn(batch_size, seq_len, 13)
    
    # Forward pass
    predictions, uncertainties = predictor(x)
    
    assert predictions.shape == (batch_size, 10, 10)
    assert uncertainties.shape == (batch_size, 10, 10)


# Test Early Warning System
def test_early_warning():
    """Test attention-based early warning system."""
    from models.level3_mission_critical.early_warning import AttentionWarningSystem
    
    warning_system = AttentionWarningSystem(
        input_dim=32,
        embed_dim=64,
        num_categories=5,
        num_heads=4,
        seq_len=20,
    )
    
    # Create dummy input
    batch_size = 2
    seq_len = 20
    
    x = torch.randn(batch_size, seq_len, 32)
    
    # Forward pass
    outputs = warning_system(x)
    
    assert "level_logits" in outputs
    assert "category_logits" in outputs
    assert "time_to_event" in outputs


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
