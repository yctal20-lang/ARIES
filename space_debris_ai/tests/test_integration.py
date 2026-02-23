"""
Integration tests for Space Debris Collector AI System.
Tests module integration and end-to-end workflows.
"""

import pytest
import numpy as np
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from space_debris_ai.core import SystemConfig, MessageBus, MessageType
from space_debris_ai.simulation import OrbitalEnv, EnvConfig
from space_debris_ai.inference import MissionController
from space_debris_ai.models import (
    CollisionAvoidanceModule,
    NavigationModule,
    AnomalyDetector,
    EnergyManagementModule,
)


class TestModuleIntegration:
    """Test module integration."""
    
    def test_collision_avoidance_integration(self):
        """Test collision avoidance module integration."""
        module = CollisionAvoidanceModule(
            config={"lidar_points": 256},
            device="cpu",
        )
        
        # Test forward pass
        inputs = {
            "lidar": np.random.randn(256, 6).astype(np.float32),
            "radar": np.random.randn(64).astype(np.float32),
            "imu": np.random.randn(12).astype(np.float32),
        }
        
        result = module.forward(inputs)
        
        assert "action" in result
        assert "collision_probability" in result
        assert "avoidance_active" in result
        assert len(result["action"]) == 6  # 3D thrust + 3D torque
    
    def test_navigation_integration(self):
        """Test navigation module integration."""
        module = NavigationModule(
            config={},
            device="cpu",
        )
        
        # Initialize
        position = np.array([7000.0, 0.0, 0.0])
        velocity = np.array([0.0, 7.5, 0.0])
        
        inputs = {
            "gps_position": position,
            "gps_velocity": velocity,
            "imu_accel": np.random.randn(3),
            "imu_gyro": np.random.randn(3),
            "star_tracker": np.array([1.0, 0.0, 0.0, 0.0]),
        }
        
        result = module.forward(inputs)
        
        assert "position" in result
        assert "velocity" in result
        assert "attitude" in result
        assert len(result["position"]) == 3
    
    def test_anomaly_detection_integration(self):
        """Test anomaly detection module integration."""
        module = AnomalyDetector(
            config={"input_dim": 12, "seq_len": 50},
            device="cpu",
        )
        
        inputs = {
            "telemetry": np.random.randn(50, 12).astype(np.float32),
        }
        
        result = module.forward(inputs)
        
        assert "is_anomaly" in result
        assert "anomaly_score" in result
        assert isinstance(result["is_anomaly"], bool)
        assert 0.0 <= result["anomaly_score"] <= 1.0
    
    def test_energy_management_integration(self):
        """Test energy management module integration."""
        module = EnergyManagementModule(
            config={},
            device="cpu",
        )
        
        inputs = {
            "battery_soc": 0.8,
            "solar_power": 1000.0,
            "power_requests": np.array([100, 200, 150, 50]),
            "dt": 0.1,
        }
        
        result = module.forward(inputs)
        
        assert "power_allocations" in result
        assert "mode" in result
        assert len(result["power_allocations"]) > 0


class TestMissionController:
    """Test mission controller integration."""
    
    def test_mission_controller_initialization(self):
        """Test mission controller can be initialized."""
        config = SystemConfig()
        controller = MissionController(config)
        
        assert controller.config is not None
        assert controller.phase.value == "idle"
        assert not controller.running
    
    def test_mission_controller_lifecycle(self):
        """Test mission controller start/stop."""
        config = SystemConfig()
        controller = MissionController(config)
        
        controller.start()
        assert controller.running
        assert controller.phase.value == "search"
        
        controller.stop()
        assert not controller.running
        assert controller.phase.value == "idle"
    
    def test_mission_controller_step(self):
        """Test mission controller step function."""
        config = SystemConfig()
        controller = MissionController(config)
        
        # Register modules
        nav_module = NavigationModule(config={}, device="cpu")
        controller.register_module("navigation", nav_module)
        
        controller.start()
        
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
        }
        
        result = controller.step(sensor_data, dt=0.1)
        
        assert "commands" in result
        assert "results" in result
        assert "state" in result
        assert "thrust" in result["commands"]
        assert "torque" in result["commands"]
        
        controller.stop()


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_simulation_with_controller(self):
        """Test running simulation with mission controller."""
        config = SystemConfig()
        controller = MissionController(config)
        
        # Register modules
        nav_module = NavigationModule(config={}, device="cpu")
        controller.register_module("navigation", nav_module)
        
        # Create environment
        env_config = EnvConfig(num_debris=10, max_episode_steps=100)
        env = OrbitalEnv(env_config)
        
        controller.start()
        
        # Run a few steps
        obs, info = env.reset(seed=42)
        
        for step in range(10):
            # Prepare sensor data
            sensor_data = {
                "position": obs[:3],
                "velocity": obs[3:6],
                "attitude": obs[6:10],
                "gps_position": obs[:3],
                "gps_velocity": obs[3:6],
                "imu_accel": np.random.randn(3),
                "imu_gyro": np.random.randn(3),
                "star_tracker": obs[6:10],
            }
            
            # Get control from controller
            result = controller.step(sensor_data, dt=0.1)
            commands = result["commands"]
            
            # Convert to action
            action = np.concatenate([
                commands["thrust"],
                commands["torque"],
                [commands.get("gripper", 0.0)],
            ])
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            if done or truncated:
                break
        
        controller.stop()
        env.close()
        
        # Check that we completed without errors
        assert True


class TestMessageBusIntegration:
    """Test message bus integration."""
    
    def test_module_message_communication(self):
        """Test modules can communicate via message bus."""
        bus = MessageBus()
        received = []
        
        def handler(msg):
            received.append(msg)
        
        bus.subscribe("test_receiver", handler, [MessageType.STATUS])
        
        # Publish message
        from space_debris_ai.core.message_bus import Message
        msg = Message.create(
            MessageType.STATUS,
            "test_sender",
            {"test": "data"},
        )
        bus.publish(msg)
        bus.process_pending()
        
        assert len(received) == 1
        assert received[0].payload["test"] == "data"
        
        bus.stop()


class TestStressTests:
    """Stress tests for system robustness."""
    
    def test_rapid_module_calls(self):
        """Test modules handle rapid calls."""
        module = NavigationModule(config={}, device="cpu")
        
        inputs = {
            "gps_position": np.array([7000.0, 0.0, 0.0]),
            "gps_velocity": np.array([0.0, 7.5, 0.0]),
            "imu_accel": np.random.randn(3),
            "imu_gyro": np.random.randn(3),
            "star_tracker": np.array([1.0, 0.0, 0.0, 0.0]),
        }
        
        # Rapid calls
        for _ in range(100):
            result = module.forward(inputs)
            assert "position" in result
    
    def test_concurrent_module_access(self):
        """Test modules handle concurrent access."""
        import threading
        
        module = NavigationModule(config={}, device="cpu")
        results = []
        errors = []
        
        def worker():
            try:
                inputs = {
                    "gps_position": np.array([7000.0, 0.0, 0.0]),
                    "gps_velocity": np.array([0.0, 7.5, 0.0]),
                    "imu_accel": np.random.randn(3),
                    "imu_gyro": np.random.randn(3),
                    "star_tracker": np.array([1.0, 0.0, 0.0, 0.0]),
                }
                result = module.forward(inputs)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

