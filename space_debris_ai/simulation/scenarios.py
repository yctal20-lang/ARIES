"""
Scenario generator for training and testing.
Provides pre-defined and procedurally generated scenarios.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from enum import Enum
import json
from pathlib import Path

from .physics import (
    SpacecraftState,
    DebrisObject,
    OrbitalElements,
    EARTH_RADIUS,
    MU_EARTH,
)


class ScenarioType(Enum):
    """Types of training/testing scenarios."""
    BASIC_COLLECTION = "basic_collection"
    COLLISION_AVOIDANCE = "collision_avoidance"
    MULTI_TARGET = "multi_target"
    FUEL_LIMITED = "fuel_limited"
    DENSE_DEBRIS_FIELD = "dense_debris_field"
    HIGH_VELOCITY = "high_velocity"
    TUMBLING_DEBRIS = "tumbling_debris"
    SYSTEM_FAILURE = "system_failure"
    MIXED_DIFFICULTY = "mixed_difficulty"


class Difficulty(Enum):
    """Scenario difficulty levels."""
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXTREME = 4


@dataclass
class Scenario:
    """
    A complete scenario specification.
    """
    name: str
    scenario_type: ScenarioType
    difficulty: Difficulty
    
    # Initial spacecraft state
    spacecraft_position: np.ndarray
    spacecraft_velocity: np.ndarray
    spacecraft_attitude: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    fuel_mass: float = 100.0
    
    # Debris objects
    debris_objects: List[Dict[str, Any]] = field(default_factory=list)
    
    # Scenario constraints
    max_time: float = 3600.0  # Maximum scenario time (seconds)
    fuel_limit: Optional[float] = None
    
    # Success criteria
    min_captures: int = 1
    max_collisions: int = 0
    
    # Injected failures (for robustness testing)
    failures: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.spacecraft_position = np.asarray(self.spacecraft_position)
        self.spacecraft_velocity = np.asarray(self.spacecraft_velocity)
        self.spacecraft_attitude = np.asarray(self.spacecraft_attitude)
    
    def get_spacecraft_state(self, total_mass: float = 500.0) -> SpacecraftState:
        """Convert scenario to SpacecraftState."""
        return SpacecraftState(
            position=self.spacecraft_position.copy(),
            velocity=self.spacecraft_velocity.copy(),
            attitude=self.spacecraft_attitude.copy(),
            angular_velocity=np.zeros(3),
            mass=total_mass + self.fuel_mass,
            fuel_mass=self.fuel_mass,
            time=0.0,
        )
    
    def get_debris_objects(self) -> List[DebrisObject]:
        """Convert debris specifications to DebrisObject instances."""
        debris_list = []
        for d in self.debris_objects:
            debris = DebrisObject(
                position=np.array(d["position"]),
                velocity=np.array(d["velocity"]),
                size=d.get("size", 1.0),
                mass=d.get("mass", 10.0),
                debris_type=d.get("type", "fragment"),
                angular_velocity=np.array(d.get("angular_velocity", [0, 0, 0])),
            )
            debris_list.append(debris)
        return debris_list
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary."""
        return {
            "name": self.name,
            "scenario_type": self.scenario_type.value,
            "difficulty": self.difficulty.value,
            "spacecraft_position": self.spacecraft_position.tolist(),
            "spacecraft_velocity": self.spacecraft_velocity.tolist(),
            "spacecraft_attitude": self.spacecraft_attitude.tolist(),
            "fuel_mass": self.fuel_mass,
            "debris_objects": self.debris_objects,
            "max_time": self.max_time,
            "fuel_limit": self.fuel_limit,
            "min_captures": self.min_captures,
            "max_collisions": self.max_collisions,
            "failures": self.failures,
            "description": self.description,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scenario":
        """Create scenario from dictionary."""
        return cls(
            name=data["name"],
            scenario_type=ScenarioType(data["scenario_type"]),
            difficulty=Difficulty(data["difficulty"]),
            spacecraft_position=np.array(data["spacecraft_position"]),
            spacecraft_velocity=np.array(data["spacecraft_velocity"]),
            spacecraft_attitude=np.array(data.get("spacecraft_attitude", [1, 0, 0, 0])),
            fuel_mass=data.get("fuel_mass", 100.0),
            debris_objects=data.get("debris_objects", []),
            max_time=data.get("max_time", 3600.0),
            fuel_limit=data.get("fuel_limit"),
            min_captures=data.get("min_captures", 1),
            max_collisions=data.get("max_collisions", 0),
            failures=data.get("failures", []),
            description=data.get("description", ""),
            tags=data.get("tags", []),
        )
    
    def save(self, path: Path) -> None:
        """Save scenario to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "Scenario":
        """Load scenario from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class ScenarioGenerator:
    """
    Generator for training and testing scenarios.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize scenario generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
    
    def generate(
        self,
        scenario_type: ScenarioType = ScenarioType.BASIC_COLLECTION,
        difficulty: Difficulty = Difficulty.MEDIUM,
        **kwargs,
    ) -> Scenario:
        """
        Generate a scenario of the specified type and difficulty.
        
        Args:
            scenario_type: Type of scenario to generate
            difficulty: Difficulty level
            **kwargs: Additional parameters
            
        Returns:
            Generated scenario
        """
        generators = {
            ScenarioType.BASIC_COLLECTION: self._generate_basic_collection,
            ScenarioType.COLLISION_AVOIDANCE: self._generate_collision_avoidance,
            ScenarioType.MULTI_TARGET: self._generate_multi_target,
            ScenarioType.FUEL_LIMITED: self._generate_fuel_limited,
            ScenarioType.DENSE_DEBRIS_FIELD: self._generate_dense_field,
            ScenarioType.HIGH_VELOCITY: self._generate_high_velocity,
            ScenarioType.TUMBLING_DEBRIS: self._generate_tumbling_debris,
            ScenarioType.SYSTEM_FAILURE: self._generate_system_failure,
            ScenarioType.MIXED_DIFFICULTY: self._generate_mixed,
        }
        
        generator = generators.get(scenario_type, self._generate_basic_collection)
        return generator(difficulty, **kwargs)
    
    def _get_circular_orbit_state(
        self,
        altitude: float,
        angle: Optional[float] = None,
    ) -> tuple:
        """Get position and velocity for circular orbit."""
        r_mag = EARTH_RADIUS + altitude
        v_mag = np.sqrt(MU_EARTH / r_mag)
        
        if angle is None:
            angle = self.rng.uniform(0, 2 * np.pi)
        
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
        
        return position, velocity
    
    def _generate_basic_collection(
        self,
        difficulty: Difficulty,
        **kwargs,
    ) -> Scenario:
        """Generate basic debris collection scenario."""
        altitude = 400.0
        sc_pos, sc_vel = self._get_circular_orbit_state(altitude)
        
        # Number of debris based on difficulty
        num_debris = {
            Difficulty.EASY: 3,
            Difficulty.MEDIUM: 5,
            Difficulty.HARD: 10,
            Difficulty.EXTREME: 20,
        }[difficulty]
        
        debris_objects = []
        for i in range(num_debris):
            # Debris in similar orbit but offset
            offset_angle = self.rng.uniform(0.01, 0.1) * (1 if i % 2 == 0 else -1)
            d_pos, d_vel = self._get_circular_orbit_state(
                altitude + self.rng.uniform(-20, 20),
            )
            
            # Add small velocity difference
            d_vel += self.rng.uniform(-0.001, 0.001, 3)
            
            debris_objects.append({
                "position": d_pos.tolist(),
                "velocity": d_vel.tolist(),
                "size": self.rng.uniform(0.1, 2.0),
                "mass": self.rng.uniform(1, 50),
                "type": self.rng.choice(["fragment", "satellite", "rocket_body"]),
                "angular_velocity": self.rng.uniform(-0.1, 0.1, 3).tolist(),
            })
        
        return Scenario(
            name=f"basic_collection_{difficulty.name.lower()}",
            scenario_type=ScenarioType.BASIC_COLLECTION,
            difficulty=difficulty,
            spacecraft_position=sc_pos,
            spacecraft_velocity=sc_vel,
            debris_objects=debris_objects,
            fuel_mass=100.0,
            min_captures=max(1, num_debris // 2),
            description=f"Basic debris collection with {num_debris} targets",
            tags=["training", "basic"],
        )
    
    def _generate_collision_avoidance(
        self,
        difficulty: Difficulty,
        **kwargs,
    ) -> Scenario:
        """Generate collision avoidance scenario."""
        altitude = 400.0
        sc_pos, sc_vel = self._get_circular_orbit_state(altitude, angle=0)
        
        # Speed factor based on difficulty
        speed_factor = {
            Difficulty.EASY: 0.5,
            Difficulty.MEDIUM: 1.0,
            Difficulty.HARD: 2.0,
            Difficulty.EXTREME: 5.0,
        }[difficulty]
        
        debris_objects = []
        
        # Create incoming debris on collision course
        num_threats = difficulty.value * 2
        
        for i in range(num_threats):
            # Debris coming from ahead
            threat_angle = self.rng.uniform(-0.05, 0.05)
            threat_distance = self.rng.uniform(5, 20)  # km ahead
            
            d_pos = sc_pos + np.array([
                threat_distance * np.cos(threat_angle),
                threat_distance * np.sin(threat_angle),
                self.rng.uniform(-1, 1),
            ])
            
            # Velocity towards spacecraft
            approach_dir = (sc_pos - d_pos) / np.linalg.norm(sc_pos - d_pos)
            d_vel = sc_vel + approach_dir * speed_factor * self.rng.uniform(0.5, 1.5)
            
            debris_objects.append({
                "position": d_pos.tolist(),
                "velocity": d_vel.tolist(),
                "size": self.rng.uniform(0.5, 3.0),
                "mass": self.rng.uniform(10, 100),
                "type": "fragment",
                "angular_velocity": [0, 0, 0],
            })
        
        return Scenario(
            name=f"collision_avoidance_{difficulty.name.lower()}",
            scenario_type=ScenarioType.COLLISION_AVOIDANCE,
            difficulty=difficulty,
            spacecraft_position=sc_pos,
            spacecraft_velocity=sc_vel,
            debris_objects=debris_objects,
            fuel_mass=50.0,
            max_time=600.0,
            min_captures=0,
            max_collisions=0,
            description=f"Avoid {num_threats} incoming debris objects",
            tags=["safety", "avoidance"],
        )
    
    def _generate_multi_target(
        self,
        difficulty: Difficulty,
        **kwargs,
    ) -> Scenario:
        """Generate multi-target collection scenario."""
        scenario = self._generate_basic_collection(difficulty)
        scenario.name = f"multi_target_{difficulty.name.lower()}"
        scenario.scenario_type = ScenarioType.MULTI_TARGET
        scenario.min_captures = len(scenario.debris_objects)
        scenario.max_time = 7200.0  # 2 hours
        scenario.description = f"Collect all {len(scenario.debris_objects)} debris objects"
        scenario.tags = ["mission", "multi-target"]
        return scenario
    
    def _generate_fuel_limited(
        self,
        difficulty: Difficulty,
        **kwargs,
    ) -> Scenario:
        """Generate fuel-limited scenario."""
        scenario = self._generate_basic_collection(difficulty)
        
        fuel_limits = {
            Difficulty.EASY: 80.0,
            Difficulty.MEDIUM: 50.0,
            Difficulty.HARD: 30.0,
            Difficulty.EXTREME: 15.0,
        }
        
        scenario.name = f"fuel_limited_{difficulty.name.lower()}"
        scenario.scenario_type = ScenarioType.FUEL_LIMITED
        scenario.fuel_mass = fuel_limits[difficulty]
        scenario.fuel_limit = fuel_limits[difficulty]
        scenario.description = f"Complete mission with only {fuel_limits[difficulty]}kg fuel"
        scenario.tags = ["efficiency", "resource-management"]
        return scenario
    
    def _generate_dense_field(
        self,
        difficulty: Difficulty,
        **kwargs,
    ) -> Scenario:
        """Generate dense debris field scenario."""
        altitude = 400.0
        sc_pos, sc_vel = self._get_circular_orbit_state(altitude)
        
        # Many debris objects in tight cluster
        num_debris = {
            Difficulty.EASY: 20,
            Difficulty.MEDIUM: 50,
            Difficulty.HARD: 100,
            Difficulty.EXTREME: 200,
        }[difficulty]
        
        debris_objects = []
        cluster_center = sc_pos + self.rng.uniform(-5, 5, 3)
        
        for i in range(num_debris):
            offset = self.rng.normal(0, 2, 3)  # Gaussian distribution
            d_pos = cluster_center + offset
            
            # Small random velocities
            d_vel = sc_vel + self.rng.uniform(-0.01, 0.01, 3)
            
            debris_objects.append({
                "position": d_pos.tolist(),
                "velocity": d_vel.tolist(),
                "size": self.rng.uniform(0.01, 0.5),  # Smaller debris
                "mass": self.rng.uniform(0.1, 5),
                "type": "fragment",
                "angular_velocity": self.rng.uniform(-0.5, 0.5, 3).tolist(),
            })
        
        return Scenario(
            name=f"dense_field_{difficulty.name.lower()}",
            scenario_type=ScenarioType.DENSE_DEBRIS_FIELD,
            difficulty=difficulty,
            spacecraft_position=sc_pos,
            spacecraft_velocity=sc_vel,
            debris_objects=debris_objects,
            fuel_mass=100.0,
            min_captures=num_debris // 4,
            description=f"Navigate and collect in dense field of {num_debris} objects",
            tags=["navigation", "dense"],
        )
    
    def _generate_high_velocity(
        self,
        difficulty: Difficulty,
        **kwargs,
    ) -> Scenario:
        """Generate high relative velocity scenario."""
        scenario = self._generate_collision_avoidance(difficulty)
        scenario.name = f"high_velocity_{difficulty.name.lower()}"
        scenario.scenario_type = ScenarioType.HIGH_VELOCITY
        
        # Increase debris velocities
        for debris in scenario.debris_objects:
            vel = np.array(debris["velocity"])
            debris["velocity"] = (vel * 1.5).tolist()
        
        scenario.description = "Handle debris with high relative velocities"
        scenario.tags = ["high-speed", "challenging"]
        return scenario
    
    def _generate_tumbling_debris(
        self,
        difficulty: Difficulty,
        **kwargs,
    ) -> Scenario:
        """Generate tumbling debris scenario."""
        scenario = self._generate_basic_collection(difficulty)
        scenario.name = f"tumbling_{difficulty.name.lower()}"
        scenario.scenario_type = ScenarioType.TUMBLING_DEBRIS
        
        # Add high angular velocities
        tumble_rates = {
            Difficulty.EASY: 0.5,
            Difficulty.MEDIUM: 1.0,
            Difficulty.HARD: 2.0,
            Difficulty.EXTREME: 5.0,
        }
        
        for debris in scenario.debris_objects:
            debris["angular_velocity"] = (
                self.rng.uniform(-1, 1, 3) * tumble_rates[difficulty]
            ).tolist()
        
        scenario.description = "Capture rapidly tumbling debris"
        scenario.tags = ["capture", "tumbling"]
        return scenario
    
    def _generate_system_failure(
        self,
        difficulty: Difficulty,
        **kwargs,
    ) -> Scenario:
        """Generate scenario with injected system failures."""
        scenario = self._generate_basic_collection(difficulty)
        scenario.name = f"system_failure_{difficulty.name.lower()}"
        scenario.scenario_type = ScenarioType.SYSTEM_FAILURE
        
        # Add failures based on difficulty
        failure_types = [
            {"type": "sensor_noise", "sensor": "imu", "magnitude": 0.1},
            {"type": "sensor_dropout", "sensor": "lidar", "duration": 10.0},
            {"type": "thruster_degradation", "thruster": "main", "efficiency": 0.8},
            {"type": "communication_delay", "delay": 2.0},
            {"type": "power_fluctuation", "magnitude": 0.2},
        ]
        
        num_failures = min(difficulty.value, len(failure_types))
        scenario.failures = self.rng.choice(
            failure_types, num_failures, replace=False
        ).tolist()
        
        scenario.description = f"Complete mission with {num_failures} system failures"
        scenario.tags = ["robustness", "failure"]
        return scenario
    
    def _generate_mixed(
        self,
        difficulty: Difficulty,
        **kwargs,
    ) -> Scenario:
        """Generate mixed difficulty scenario."""
        # Combine elements from different scenarios
        base = self._generate_basic_collection(difficulty)
        avoidance = self._generate_collision_avoidance(difficulty)
        
        # Merge debris from both
        base.debris_objects.extend(avoidance.debris_objects[:3])
        
        base.name = f"mixed_{difficulty.name.lower()}"
        base.scenario_type = ScenarioType.MIXED_DIFFICULTY
        base.description = "Mixed scenario combining collection and avoidance"
        base.tags = ["mixed", "comprehensive"]
        
        return base
    
    def generate_curriculum(
        self,
        num_scenarios: int = 100,
        start_difficulty: Difficulty = Difficulty.EASY,
    ) -> List[Scenario]:
        """
        Generate a curriculum of scenarios with increasing difficulty.
        
        Args:
            num_scenarios: Total number of scenarios
            start_difficulty: Starting difficulty level
            
        Returns:
            List of scenarios ordered by difficulty
        """
        scenarios = []
        scenario_types = list(ScenarioType)
        
        for i in range(num_scenarios):
            # Gradually increase difficulty
            progress = i / num_scenarios
            if progress < 0.25:
                difficulty = Difficulty.EASY
            elif progress < 0.5:
                difficulty = Difficulty.MEDIUM
            elif progress < 0.75:
                difficulty = Difficulty.HARD
            else:
                difficulty = Difficulty.EXTREME
            
            # Cycle through scenario types
            scenario_type = scenario_types[i % len(scenario_types)]
            
            scenario = self.generate(scenario_type, difficulty)
            scenario.name = f"curriculum_{i:04d}_{scenario.name}"
            scenarios.append(scenario)
        
        return scenarios
    
    def generate_test_suite(self) -> List[Scenario]:
        """
        Generate a comprehensive test suite covering all scenario types.
        
        Returns:
            List of test scenarios
        """
        test_scenarios = []
        
        for scenario_type in ScenarioType:
            for difficulty in Difficulty:
                scenario = self.generate(scenario_type, difficulty)
                scenario.name = f"test_{scenario_type.value}_{difficulty.name.lower()}"
                scenario.tags.append("test")
                test_scenarios.append(scenario)
        
        return test_scenarios
