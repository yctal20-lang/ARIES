"""
Configuration management for Space Debris Collector AI System.
Uses Pydantic for validation and type safety.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class MissionMode(str, Enum):
    """Mission operation modes."""
    AGGRESSIVE = "aggressive"  # Fast debris collection, higher fuel usage
    BALANCED = "balanced"      # Balance between speed and efficiency
    ECONOMIC = "economic"      # Fuel-efficient, slower operations
    EMERGENCY = "emergency"    # Safety-first mode
    HIBERNATE = "hibernate"    # Minimal power consumption


class AlertLevel(str, Enum):
    """System alert levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class OrbitalParameters(BaseModel):
    """Orbital mechanics parameters."""
    semi_major_axis: float = Field(default=6778.0, description="Semi-major axis in km (LEO ~400km altitude)")
    eccentricity: float = Field(default=0.0, ge=0.0, lt=1.0, description="Orbital eccentricity")
    inclination: float = Field(default=51.6, ge=0.0, le=180.0, description="Inclination in degrees (ISS-like)")
    raan: float = Field(default=0.0, description="Right ascension of ascending node in degrees")
    arg_periapsis: float = Field(default=0.0, description="Argument of periapsis in degrees")
    true_anomaly: float = Field(default=0.0, description="True anomaly in degrees")
    
    # Physical constants
    mu_earth: float = Field(default=398600.4418, description="Earth gravitational parameter km³/s²")
    earth_radius: float = Field(default=6371.0, description="Earth radius in km")


class SpacecraftParameters(BaseModel):
    """Spacecraft physical parameters."""
    mass: float = Field(default=500.0, gt=0, description="Spacecraft mass in kg")
    max_thrust: float = Field(default=100.0, gt=0, description="Maximum thrust in N")
    specific_impulse: float = Field(default=300.0, gt=0, description="Specific impulse in seconds")
    fuel_mass: float = Field(default=100.0, ge=0, description="Initial fuel mass in kg")
    
    # Dimensions for collision detection
    length: float = Field(default=3.0, description="Length in meters")
    width: float = Field(default=2.0, description="Width in meters")
    height: float = Field(default=2.0, description="Height in meters")
    
    # Power system
    solar_panel_area: float = Field(default=20.0, description="Solar panel area in m²")
    solar_efficiency: float = Field(default=0.3, description="Solar panel efficiency")
    battery_capacity: float = Field(default=10000.0, description="Battery capacity in Wh")
    
    # Manipulator
    manipulator_dof: int = Field(default=6, ge=1, le=7, description="Degrees of freedom")
    manipulator_reach: float = Field(default=5.0, description="Maximum reach in meters")
    max_grip_force: float = Field(default=50.0, description="Maximum grip force in N")


class SensorConfig(BaseModel):
    """Sensor configuration."""
    # IMU
    imu_sample_rate: float = Field(default=100.0, description="IMU sample rate in Hz")
    imu_noise_std: float = Field(default=0.01, description="IMU noise standard deviation")
    
    # Lidar
    lidar_range: float = Field(default=1000.0, description="Lidar range in meters")
    lidar_resolution: float = Field(default=0.1, description="Angular resolution in degrees")
    lidar_update_rate: float = Field(default=10.0, description="Update rate in Hz")
    
    # Camera
    camera_fov: float = Field(default=60.0, description="Field of view in degrees")
    camera_resolution: tuple = Field(default=(1920, 1080), description="Resolution (width, height)")
    camera_fps: float = Field(default=30.0, description="Frames per second")
    
    # Radar
    radar_range: float = Field(default=5000.0, description="Radar range in meters")
    radar_update_rate: float = Field(default=1.0, description="Update rate in Hz")
    
    # GPS/Star tracker
    position_accuracy: float = Field(default=10.0, description="Position accuracy in meters")
    attitude_accuracy: float = Field(default=0.01, description="Attitude accuracy in degrees")


class SafetyConfig(BaseModel):
    """Safety system configuration."""
    # Collision avoidance
    min_collision_distance: float = Field(default=50.0, description="Minimum safe distance in meters")
    collision_warning_time: float = Field(default=30.0, description="Warning time before collision in seconds")
    max_reaction_time: float = Field(default=0.5, description="Maximum reaction time in seconds")
    
    # Anomaly detection
    anomaly_threshold: float = Field(default=0.8, description="Anomaly score threshold (0-1)")
    max_acceleration: float = Field(default=5.0, description="Maximum safe acceleration in m/s²")
    max_angular_rate: float = Field(default=10.0, description="Maximum safe angular rate in deg/s")
    
    # Fail-safe
    watchdog_timeout: float = Field(default=5.0, description="Watchdog timeout in seconds")
    max_consecutive_failures: int = Field(default=3, description="Max failures before fallback")
    fallback_enabled: bool = Field(default=True, description="Enable classical algorithm fallback")
    
    # Energy
    min_battery_level: float = Field(default=0.1, description="Minimum battery level (0-1)")
    emergency_battery_level: float = Field(default=0.05, description="Emergency battery level")


class NeuralNetworkConfig(BaseModel):
    """Neural network architecture configuration."""
    # General
    device: str = Field(default="cuda", description="Compute device (cuda/cpu)")
    precision: str = Field(default="float32", description="Computation precision")
    
    # Collision avoidance
    collision_backbone: str = Field(default="resnet18", description="CNN backbone")
    collision_hidden_dim: int = Field(default=256, description="Hidden dimension")
    
    # Anomaly detection
    anomaly_lstm_hidden: int = Field(default=128, description="LSTM hidden size")
    anomaly_lstm_layers: int = Field(default=2, description="Number of LSTM layers")
    anomaly_sequence_length: int = Field(default=100, description="Input sequence length")
    
    # State prediction
    prediction_horizon: int = Field(default=50, description="Prediction horizon (timesteps)")
    tcn_channels: List[int] = Field(default=[64, 128, 256], description="TCN channel sizes")
    
    # Debris recognition
    debris_backbone: str = Field(default="efficientnet_b4", description="Vision backbone")
    debris_num_classes: int = Field(default=10, description="Number of debris classes")
    
    # RL agents
    rl_hidden_sizes: List[int] = Field(default=[256, 256], description="RL network hidden sizes")
    rl_learning_rate: float = Field(default=3e-4, description="Learning rate")
    rl_gamma: float = Field(default=0.99, description="Discount factor")
    rl_buffer_size: int = Field(default=1000000, description="Replay buffer size")


class SimulationConfig(BaseModel):
    """Simulation environment configuration."""
    # Time
    dt: float = Field(default=0.1, description="Simulation timestep in seconds")
    max_episode_steps: int = Field(default=10000, description="Maximum steps per episode")
    real_time_factor: float = Field(default=1.0, description="Simulation speed multiplier")
    
    # Environment
    num_debris_objects: int = Field(default=50, description="Number of debris objects")
    debris_size_range: tuple = Field(default=(0.01, 5.0), description="Debris size range in meters")
    debris_velocity_range: tuple = Field(default=(0.0, 100.0), description="Relative velocity range in m/s")
    
    # Rendering
    render_mode: str = Field(default="rgb_array", description="Rendering mode")
    render_fps: int = Field(default=30, description="Rendering FPS")
    
    # Scenarios
    include_collisions: bool = Field(default=True, description="Include collision scenarios")
    include_failures: bool = Field(default=True, description="Include system failures")
    difficulty: str = Field(default="medium", description="Scenario difficulty")


class TrainingConfig(BaseModel):
    """Training configuration."""
    # General
    seed: int = Field(default=42, description="Random seed")
    num_envs: int = Field(default=8, description="Number of parallel environments")
    total_timesteps: int = Field(default=10000000, description="Total training timesteps")
    
    # Checkpointing
    save_freq: int = Field(default=100000, description="Checkpoint save frequency")
    eval_freq: int = Field(default=50000, description="Evaluation frequency")
    log_freq: int = Field(default=1000, description="Logging frequency")
    
    # Paths
    checkpoint_dir: str = Field(default="checkpoints", description="Checkpoint directory")
    log_dir: str = Field(default="logs", description="Log directory")
    
    # Early stopping
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    patience: int = Field(default=10, description="Early stopping patience")
    min_improvement: float = Field(default=0.01, description="Minimum improvement threshold")


class SystemConfig(BaseModel):
    """Main system configuration combining all subsystems."""
    # Mission
    mission_mode: MissionMode = Field(default=MissionMode.BALANCED)
    mission_name: str = Field(default="debris_collection_001")
    
    # Subsystem configs
    orbital: OrbitalParameters = Field(default_factory=OrbitalParameters)
    spacecraft: SpacecraftParameters = Field(default_factory=SpacecraftParameters)
    sensors: SensorConfig = Field(default_factory=SensorConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    neural_network: NeuralNetworkConfig = Field(default_factory=NeuralNetworkConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    
    # System
    debug_mode: bool = Field(default=False)
    verbose: bool = Field(default=True)
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "SystemConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# Alias for convenience
Config = SystemConfig
