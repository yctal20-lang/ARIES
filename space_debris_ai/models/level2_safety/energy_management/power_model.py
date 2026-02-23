"""
Physics-based power generation and consumption model.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class EnergyMode(Enum):
    """Energy management modes."""
    NORMAL = "normal"           # Standard operation
    ECONOMY = "economy"         # Power-saving mode
    EMERGENCY = "emergency"     # Minimum power for safety
    HIBERNATE = "hibernate"     # Deep sleep mode
    BURST = "burst"            # High power for critical operations


@dataclass
class SubsystemPower:
    """Power characteristics of a subsystem."""
    name: str
    min_power: float           # Minimum power to function (W)
    nominal_power: float       # Normal operating power (W)
    max_power: float           # Maximum power draw (W)
    priority: int              # Priority (1=highest, 10=lowest)
    essential: bool = False    # True if required for survival
    current_power: float = 0.0
    enabled: bool = True
    
    def set_power(self, fraction: float) -> float:
        """Set power as fraction of nominal. Returns actual power."""
        if not self.enabled:
            self.current_power = 0.0
            return 0.0
        
        fraction = np.clip(fraction, 0.0, 1.0)
        
        if fraction < 0.01:
            self.current_power = 0.0
        else:
            # Linear interpolation between min and max
            self.current_power = self.min_power + fraction * (self.max_power - self.min_power)
        
        return self.current_power


@dataclass
class PowerState:
    """Current power system state."""
    battery_charge: float      # Current charge (Wh)
    battery_capacity: float    # Total capacity (Wh)
    solar_generation: float    # Current solar generation (W)
    total_consumption: float   # Current total consumption (W)
    net_power: float           # Generation - consumption (W)
    time_to_empty: float       # Hours until battery empty (at current rate)
    time_to_full: float        # Hours until battery full (at current rate)
    mode: EnergyMode           # Current operating mode
    subsystem_allocation: Dict[str, float] = field(default_factory=dict)
    in_eclipse: bool = False   # True if in Earth's shadow
    
    @property
    def battery_soc(self) -> float:
        """State of charge (0-1)."""
        return self.battery_charge / self.battery_capacity
    
    @property
    def is_critical(self) -> bool:
        """True if battery critically low."""
        return self.battery_soc < 0.1
    
    @property
    def is_charging(self) -> bool:
        """True if net power is positive."""
        return self.net_power > 0


class PowerModel:
    """
    Physics-based model of spacecraft power system.
    
    Models:
    - Solar panel power generation
    - Battery charge/discharge dynamics
    - Subsystem power consumption
    - Eclipse periods
    """
    
    # Solar constant at 1 AU (W/m²)
    SOLAR_CONSTANT = 1361.0
    
    def __init__(
        self,
        solar_panel_area: float = 20.0,
        solar_efficiency: float = 0.30,
        battery_capacity: float = 10000.0,
        battery_efficiency: float = 0.95,
        initial_charge: Optional[float] = None,
    ):
        """
        Initialize power model.
        
        Args:
            solar_panel_area: Solar panel area (m²)
            solar_efficiency: Solar cell efficiency (0-1)
            battery_capacity: Battery capacity (Wh)
            battery_efficiency: Battery round-trip efficiency
            initial_charge: Initial battery charge (Wh)
        """
        self.solar_panel_area = solar_panel_area
        self.solar_efficiency = solar_efficiency
        self.battery_capacity = battery_capacity
        self.battery_efficiency = battery_efficiency
        
        self.battery_charge = initial_charge if initial_charge else battery_capacity * 0.8
        self.mode = EnergyMode.NORMAL
        
        # Define subsystems
        self.subsystems = self._init_subsystems()
        
        # Eclipse state
        self.in_eclipse = False
        self.eclipse_fraction = 0.0  # 0 = full sun, 1 = full shadow
    
    def _init_subsystems(self) -> Dict[str, SubsystemPower]:
        """Initialize subsystem power definitions."""
        return {
            "navigation": SubsystemPower(
                name="navigation",
                min_power=10, nominal_power=30, max_power=50,
                priority=1, essential=True,
            ),
            "collision_avoidance": SubsystemPower(
                name="collision_avoidance",
                min_power=20, nominal_power=50, max_power=100,
                priority=1, essential=True,
            ),
            "communication": SubsystemPower(
                name="communication",
                min_power=5, nominal_power=20, max_power=100,
                priority=2, essential=True,
            ),
            "thermal_control": SubsystemPower(
                name="thermal_control",
                min_power=15, nominal_power=40, max_power=80,
                priority=2, essential=True,
            ),
            "lidar": SubsystemPower(
                name="lidar",
                min_power=20, nominal_power=50, max_power=80,
                priority=3, essential=False,
            ),
            "radar": SubsystemPower(
                name="radar",
                min_power=30, nominal_power=80, max_power=150,
                priority=3, essential=False,
            ),
            "camera": SubsystemPower(
                name="camera",
                min_power=5, nominal_power=15, max_power=30,
                priority=4, essential=False,
            ),
            "manipulator": SubsystemPower(
                name="manipulator",
                min_power=0, nominal_power=100, max_power=300,
                priority=5, essential=False,
            ),
            "propulsion": SubsystemPower(
                name="propulsion",
                min_power=0, nominal_power=50, max_power=200,
                priority=4, essential=False,
            ),
            "computer": SubsystemPower(
                name="computer",
                min_power=30, nominal_power=80, max_power=150,
                priority=1, essential=True,
            ),
            "attitude_control": SubsystemPower(
                name="attitude_control",
                min_power=10, nominal_power=25, max_power=50,
                priority=2, essential=True,
            ),
        }
    
    def calculate_solar_generation(
        self,
        sun_angle: float = 0.0,
        distance_au: float = 1.0,
    ) -> float:
        """
        Calculate solar power generation.
        
        Args:
            sun_angle: Angle between panel normal and sun vector (radians)
            distance_au: Distance from sun in AU
            
        Returns:
            Power generation (W)
        """
        if self.in_eclipse:
            return 0.0
        
        # Intensity varies with distance squared
        intensity = self.SOLAR_CONSTANT / (distance_au ** 2)
        
        # Cosine loss from angle
        cos_angle = np.cos(sun_angle) * (1 - self.eclipse_fraction)
        cos_angle = max(0, cos_angle)
        
        # Power = Area × Efficiency × Intensity × cos(angle)
        power = self.solar_panel_area * self.solar_efficiency * intensity * cos_angle
        
        return power
    
    def update_eclipse(
        self,
        spacecraft_position: np.ndarray,
        sun_direction: np.ndarray,
    ) -> float:
        """
        Update eclipse state based on position.
        
        Args:
            spacecraft_position: Position in ECI (km)
            sun_direction: Unit vector to sun
            
        Returns:
            Eclipse fraction (0=sun, 1=shadow)
        """
        earth_radius = 6371.0  # km
        
        # Project position onto sun direction
        proj = np.dot(spacecraft_position, sun_direction)
        
        if proj > 0:
            # Spacecraft on sun side
            self.in_eclipse = False
            self.eclipse_fraction = 0.0
        else:
            # Check if in shadow cone
            perpendicular_dist = np.linalg.norm(
                spacecraft_position - proj * sun_direction
            )
            
            if perpendicular_dist < earth_radius:
                self.in_eclipse = True
                # Gradual transition at shadow boundary
                self.eclipse_fraction = 1.0 - max(0, (perpendicular_dist - earth_radius * 0.95) / (earth_radius * 0.05))
            else:
                self.in_eclipse = False
                self.eclipse_fraction = 0.0
        
        return self.eclipse_fraction
    
    def allocate_power(
        self,
        allocations: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Allocate power to subsystems.
        
        Args:
            allocations: Dict of subsystem_name -> power_fraction (0-1)
            
        Returns:
            Actual power allocated to each subsystem
        """
        actual = {}
        
        for name, subsystem in self.subsystems.items():
            fraction = allocations.get(name, 0.5)  # Default to 50%
            actual[name] = subsystem.set_power(fraction)
        
        return actual
    
    def step(
        self,
        dt: float,
        sun_angle: float = 0.0,
        allocations: Optional[Dict[str, float]] = None,
    ) -> PowerState:
        """
        Step power simulation forward.
        
        Args:
            dt: Time step (seconds)
            sun_angle: Sun angle for solar generation
            allocations: Power allocations (if None, uses mode defaults)
            
        Returns:
            Updated power state
        """
        # Apply mode-specific allocations if not provided
        if allocations is None:
            allocations = self._get_mode_allocations()
        
        # Calculate generation
        generation = self.calculate_solar_generation(sun_angle)
        
        # Calculate consumption
        actual_alloc = self.allocate_power(allocations)
        consumption = sum(actual_alloc.values())
        
        # Net power
        net_power = generation - consumption
        
        # Update battery
        energy_change = net_power * dt / 3600  # Convert W⋅s to Wh
        
        if energy_change > 0:
            # Charging (apply efficiency)
            energy_change *= self.battery_efficiency
        
        self.battery_charge = np.clip(
            self.battery_charge + energy_change,
            0,
            self.battery_capacity,
        )
        
        # Calculate time to empty/full
        if net_power < 0:
            time_to_empty = self.battery_charge / (-net_power) if net_power < 0 else float('inf')
            time_to_full = float('inf')
        else:
            time_to_empty = float('inf')
            remaining = self.battery_capacity - self.battery_charge
            time_to_full = remaining / (net_power * self.battery_efficiency) if net_power > 0 else float('inf')
        
        # Auto mode switching based on battery level
        self._update_mode()
        
        return PowerState(
            battery_charge=self.battery_charge,
            battery_capacity=self.battery_capacity,
            solar_generation=generation,
            total_consumption=consumption,
            net_power=net_power,
            time_to_empty=time_to_empty,
            time_to_full=time_to_full,
            mode=self.mode,
            subsystem_allocation=actual_alloc,
            in_eclipse=self.in_eclipse,
        )
    
    def _get_mode_allocations(self) -> Dict[str, float]:
        """Get default power allocations for current mode."""
        if self.mode == EnergyMode.NORMAL:
            return {name: 0.7 for name in self.subsystems}
        
        elif self.mode == EnergyMode.ECONOMY:
            allocs = {}
            for name, sub in self.subsystems.items():
                if sub.essential:
                    allocs[name] = 0.5
                else:
                    allocs[name] = 0.2
            return allocs
        
        elif self.mode == EnergyMode.EMERGENCY:
            allocs = {}
            for name, sub in self.subsystems.items():
                if sub.essential:
                    allocs[name] = 0.3
                else:
                    allocs[name] = 0.0
            return allocs
        
        elif self.mode == EnergyMode.HIBERNATE:
            allocs = {}
            for name, sub in self.subsystems.items():
                if sub.priority == 1:
                    allocs[name] = 0.2
                else:
                    allocs[name] = 0.0
            return allocs
        
        elif self.mode == EnergyMode.BURST:
            return {name: 1.0 for name in self.subsystems}
        
        return {name: 0.5 for name in self.subsystems}
    
    def _update_mode(self) -> None:
        """Auto-update mode based on battery level."""
        soc = self.battery_charge / self.battery_capacity
        
        if soc < 0.05:
            self.mode = EnergyMode.HIBERNATE
        elif soc < 0.1:
            self.mode = EnergyMode.EMERGENCY
        elif soc < 0.2:
            self.mode = EnergyMode.ECONOMY
        elif self.mode in [EnergyMode.EMERGENCY, EnergyMode.HIBERNATE]:
            # Recover from emergency modes when battery is sufficient
            if soc > 0.3:
                self.mode = EnergyMode.NORMAL
    
    def get_observation(self) -> np.ndarray:
        """Get observation vector for RL agent."""
        obs = [
            self.battery_charge / self.battery_capacity,  # SOC
            float(self.in_eclipse),
            self.eclipse_fraction,
        ]
        
        # Subsystem states
        for subsystem in self.subsystems.values():
            obs.append(subsystem.current_power / max(subsystem.max_power, 1))
            obs.append(float(subsystem.enabled))
        
        return np.array(obs, dtype=np.float32)
