"""
Gymnasium-compatible orbital environment for training RL agents.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, SupportsFloat
from dataclasses import dataclass, field
import gymnasium as gym
from gymnasium import spaces

from .physics import (
    OrbitalMechanics,
    SpacecraftDynamics,
    SpacecraftState,
    DebrisObject,
    OrbitalElements,
    EARTH_RADIUS,
    MU_EARTH,
)


@dataclass
class EnvConfig:
    """Environment configuration."""
    # Simulation
    dt: float = 0.1                    # Timestep (seconds)
    max_episode_steps: int = 10000     # Maximum steps per episode
    
    # Spacecraft
    initial_altitude: float = 400.0    # Initial altitude (km)
    spacecraft_mass: float = 500.0     # Spacecraft mass (kg)
    fuel_mass: float = 100.0           # Initial fuel (kg)
    max_thrust: float = 100.0          # Maximum thrust (N)
    max_torque: float = 10.0           # Maximum torque (N⋅m)
    
    # Debris
    num_debris: int = 50               # Number of debris objects
    debris_size_range: Tuple[float, float] = (0.01, 5.0)  # Size range (m)
    debris_altitude_range: Tuple[float, float] = (350.0, 450.0)  # Altitude range (km)
    
    # Safety
    collision_distance: float = 0.05   # Collision threshold (km = 50m)
    safe_distance: float = 0.5         # Safe distance (km = 500m)
    
    # Rewards
    collision_penalty: float = -1000.0
    capture_reward: float = 100.0
    fuel_penalty: float = -0.01
    proximity_reward: float = 1.0
    safety_bonus: float = 0.1


class OrbitalEnv(gym.Env):
    """
    Gymnasium environment for orbital debris collection.
    
    Observation Space:
        - Spacecraft state (position, velocity, attitude, angular velocity)
        - Nearest debris objects (relative position, velocity)
        - Resource levels (fuel, battery)
        - Mission status
    
    Action Space:
        - Thrust command (3D continuous)
        - Torque command (3D continuous)
        - Gripper command (1D continuous: open/close)
    
    Reward:
        - Positive for debris capture
        - Negative for collision
        - Small penalty for fuel usage
        - Bonus for maintaining safe operations
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the orbital environment.
        
        Args:
            config: Environment configuration
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        super().__init__()
        
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        
        # Initialize physics
        self.mechanics = OrbitalMechanics(include_j2=True, include_drag=True)
        self.dynamics = SpacecraftDynamics(
            mass=self.config.spacecraft_mass,
            max_thrust=self.config.max_thrust,
            max_torque=self.config.max_torque,
        )
        
        # State variables
        self.spacecraft: Optional[SpacecraftState] = None
        self.debris_objects: List[DebrisObject] = []
        self.captured_debris: List[str] = []
        self.step_count: int = 0
        self.total_fuel_used: float = 0.0
        
        # Define observation space
        # [spacecraft_pos(3), spacecraft_vel(3), attitude(4), angular_vel(3),
        #  fuel(1), nearest_debris_rel_pos(3*5), nearest_debris_rel_vel(3*5)]
        obs_dim = 3 + 3 + 4 + 3 + 1 + 15 + 15  # = 44
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Define action space
        # [thrust_x, thrust_y, thrust_z, torque_x, torque_y, torque_z, gripper]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )
        
        # Rendering
        self._fig = None
        self._ax = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        # Reset counters
        self.step_count = 0
        self.captured_debris = []
        self.total_fuel_used = 0.0
        
        # Initialize spacecraft in circular orbit
        altitude = self.config.initial_altitude
        r_mag = EARTH_RADIUS + altitude
        
        # Circular orbit velocity
        v_mag = np.sqrt(MU_EARTH / r_mag)
        
        # Random initial position on orbit
        angle = self.np_random.uniform(0, 2 * np.pi)
        
        position = np.array([
            r_mag * np.cos(angle),
            r_mag * np.sin(angle),
            0.0
        ])
        
        velocity = np.array([
            -v_mag * np.sin(angle),
            v_mag * np.cos(angle),
            0.0
        ])
        
        self.spacecraft = SpacecraftState(
            position=position,
            velocity=velocity,
            attitude=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
            angular_velocity=np.zeros(3),
            mass=self.config.spacecraft_mass + self.config.fuel_mass,
            fuel_mass=self.config.fuel_mass,
            time=0.0,
        )
        
        # Generate debris field
        self._generate_debris()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Action array [thrust(3), torque(3), gripper(1)]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        assert self.spacecraft is not None, "Call reset() first"
        
        # Parse action
        thrust_cmd = action[:3] * self.config.max_thrust
        torque_cmd = action[3:6] * self.config.max_torque
        gripper_cmd = action[6]
        
        # Store previous fuel
        prev_fuel = self.spacecraft.fuel_mass
        
        # Apply dynamics
        self.spacecraft = self.dynamics.apply_control(
            self.spacecraft,
            thrust_cmd,
            torque_cmd,
            self.config.dt,
        )
        
        # Update debris positions
        for debris in self.debris_objects:
            debris.propagate(self.config.dt, self.mechanics)
        
        # Calculate fuel used this step
        fuel_used = prev_fuel - self.spacecraft.fuel_mass
        self.total_fuel_used += fuel_used
        
        # Check for collisions and captures
        reward = 0.0
        terminated = False
        
        collision_detected, capture_possible = self._check_debris_interactions()
        
        # Collision penalty
        if collision_detected:
            reward += self.config.collision_penalty
            terminated = True
        
        # Capture reward
        if capture_possible and gripper_cmd > 0.5:
            captured = self._attempt_capture()
            if captured:
                reward += self.config.capture_reward
        
        # Fuel penalty
        reward += fuel_used * self.config.fuel_penalty * 1000  # Scale appropriately
        
        # Safety bonus for maintaining safe distance
        if not collision_detected:
            min_distance = self._get_min_debris_distance()
            if min_distance > self.config.safe_distance:
                reward += self.config.safety_bonus
        
        # Proximity reward for approaching target debris
        if len(self.debris_objects) > 0:
            nearest_dist = self._get_min_debris_distance()
            if nearest_dist < self.config.safe_distance:
                reward += self.config.proximity_reward * (1 - nearest_dist / self.config.safe_distance)
        
        # Check termination conditions
        self.step_count += 1
        truncated = self.step_count >= self.config.max_episode_steps
        
        # Check if out of fuel
        if self.spacecraft.fuel_mass <= 0:
            truncated = True
        
        # Check if all debris captured
        if len(self.debris_objects) == 0:
            terminated = True
            reward += 500.0  # Mission complete bonus
        
        # Check if crashed into Earth
        if self.spacecraft.altitude < 100:
            terminated = True
            reward += self.config.collision_penalty
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _generate_debris(self) -> None:
        """Generate random debris field."""
        self.debris_objects = []
        
        for i in range(self.config.num_debris):
            # Random altitude within range
            altitude = self.np_random.uniform(
                self.config.debris_altitude_range[0],
                self.config.debris_altitude_range[1],
            )
            r_mag = EARTH_RADIUS + altitude
            
            # Random position on orbit
            angle = self.np_random.uniform(0, 2 * np.pi)
            inclination_offset = self.np_random.uniform(-0.1, 0.1)  # Small inclination variation
            
            position = np.array([
                r_mag * np.cos(angle),
                r_mag * np.sin(angle),
                r_mag * inclination_offset,
            ])
            
            # Circular orbit velocity with small perturbation
            v_mag = np.sqrt(MU_EARTH / r_mag)
            v_perturbation = self.np_random.uniform(-0.01, 0.01, 3)
            
            velocity = np.array([
                -v_mag * np.sin(angle),
                v_mag * np.cos(angle),
                0.0,
            ]) + v_perturbation
            
            # Random debris properties
            size = self.np_random.uniform(
                self.config.debris_size_range[0],
                self.config.debris_size_range[1],
            )
            mass = size ** 3 * 1000  # Rough density estimate
            
            debris_types = ["fragment", "rocket_body", "satellite", "tool", "panel"]
            debris_type = self.np_random.choice(debris_types)
            
            angular_velocity = self.np_random.uniform(-0.5, 0.5, 3)
            
            debris = DebrisObject(
                position=position,
                velocity=velocity,
                size=size,
                mass=mass,
                debris_type=debris_type,
                angular_velocity=angular_velocity,
            )
            
            self.debris_objects.append(debris)
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        obs = []
        
        # Spacecraft state (normalized)
        obs.extend(self.spacecraft.position / 1000)  # Scale down
        obs.extend(self.spacecraft.velocity * 100)   # Scale up
        obs.extend(self.spacecraft.attitude)
        obs.extend(self.spacecraft.angular_velocity)
        obs.append(self.spacecraft.fuel_mass / self.config.fuel_mass)  # Normalized fuel
        
        # Nearest 5 debris objects (relative position and velocity)
        debris_rel_data = []
        for debris in self.debris_objects:
            rel_pos = debris.position - self.spacecraft.position
            rel_vel = debris.velocity - self.spacecraft.velocity
            dist = np.linalg.norm(rel_pos)
            debris_rel_data.append((dist, rel_pos, rel_vel))
        
        # Sort by distance and take nearest 5
        debris_rel_data.sort(key=lambda x: x[0])
        
        for i in range(5):
            if i < len(debris_rel_data):
                _, rel_pos, rel_vel = debris_rel_data[i]
                obs.extend(rel_pos)
                obs.extend(rel_vel)
            else:
                # Pad with zeros if fewer than 5 debris
                obs.extend([0.0] * 6)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        return {
            "step": self.step_count,
            "altitude": self.spacecraft.altitude,
            "speed": self.spacecraft.speed,
            "fuel_remaining": self.spacecraft.fuel_mass,
            "fuel_used": self.total_fuel_used,
            "debris_remaining": len(self.debris_objects),
            "debris_captured": len(self.captured_debris),
            "min_debris_distance": self._get_min_debris_distance(),
        }
    
    def _get_min_debris_distance(self) -> float:
        """Get distance to nearest debris."""
        if not self.debris_objects:
            return float('inf')
        
        return min(
            debris.distance_to(self.spacecraft.position)
            for debris in self.debris_objects
        )
    
    def _check_debris_interactions(self) -> Tuple[bool, bool]:
        """
        Check for debris interactions.
        
        Returns:
            (collision_detected, capture_possible)
        """
        collision_detected = False
        capture_possible = False
        
        for debris in self.debris_objects:
            dist = debris.distance_to(self.spacecraft.position)
            
            if dist < self.config.collision_distance:
                # Check relative velocity for collision vs capture
                rel_vel = debris.relative_velocity(self.spacecraft.velocity)
                
                if rel_vel > 0.01:  # High relative velocity = collision
                    collision_detected = True
                else:
                    capture_possible = True
        
        return collision_detected, capture_possible
    
    def _attempt_capture(self) -> bool:
        """
        Attempt to capture nearest debris.
        
        Returns:
            True if capture successful
        """
        if not self.debris_objects:
            return False
        
        # Find nearest capturable debris
        for i, debris in enumerate(self.debris_objects):
            dist = debris.distance_to(self.spacecraft.position)
            rel_vel = debris.relative_velocity(self.spacecraft.velocity)
            
            if dist < self.config.collision_distance and rel_vel < 0.01:
                # Capture successful
                captured = self.debris_objects.pop(i)
                self.captured_debris.append(captured.object_id)
                
                # Add debris mass to spacecraft
                self.spacecraft.mass += captured.mass
                
                return True
        
        return False
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            return None
        
        if self._fig is None:
            self._fig = plt.figure(figsize=(10, 10))
            self._ax = self._fig.add_subplot(111, projection='3d')
        
        self._ax.clear()
        
        # Draw Earth (simplified as sphere)
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
        y = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
        z = EARTH_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))
        self._ax.plot_surface(x, y, z, alpha=0.3, color='blue')
        
        # Draw spacecraft
        if self.spacecraft:
            self._ax.scatter(
                *self.spacecraft.position,
                c='green', s=100, marker='^', label='Spacecraft'
            )
        
        # Draw debris
        if self.debris_objects:
            debris_pos = np.array([d.position for d in self.debris_objects])
            debris_sizes = np.array([d.size * 20 for d in self.debris_objects])
            self._ax.scatter(
                debris_pos[:, 0], debris_pos[:, 1], debris_pos[:, 2],
                c='red', s=debris_sizes, marker='o', alpha=0.6, label='Debris'
            )
        
        self._ax.set_xlabel('X (km)')
        self._ax.set_ylabel('Y (km)')
        self._ax.set_zlabel('Z (km)')
        self._ax.legend()
        
        # Set axis limits
        if self.spacecraft:
            r = np.linalg.norm(self.spacecraft.position) * 1.2
            self._ax.set_xlim(-r, r)
            self._ax.set_ylim(-r, r)
            self._ax.set_zlim(-r, r)
        
        if self.render_mode == "human":
            plt.pause(0.01)
            return None
        else:
            self._fig.canvas.draw()
            img = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            return img
    
    def close(self) -> None:
        """Clean up resources."""
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._ax = None


# Register environment with Gymnasium
gym.register(
    id="OrbitalDebris-v0",
    entry_point="space_debris_ai.simulation.environment:OrbitalEnv",
    max_episode_steps=10000,
)
