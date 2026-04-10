"""
Orbital mechanics and spacecraft dynamics for simulation.
Implements Keplerian orbital mechanics with perturbations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from enum import Enum


# Physical constants
MU_EARTH = 398600.4418  # Earth gravitational parameter (km³/s²)
EARTH_RADIUS = 6371.0   # Earth radius (km)
J2 = 1.08263e-3         # Earth's J2 perturbation coefficient
SOLAR_CONSTANT = 1361   # W/m² at 1 AU
C_LIGHT = 299792.458    # Speed of light (km/s)


class ReferenceFrame(Enum):
    """Reference frame types."""
    ECI = "eci"         # Earth-Centered Inertial
    ECEF = "ecef"       # Earth-Centered Earth-Fixed
    LVLH = "lvlh"       # Local Vertical Local Horizontal
    BODY = "body"       # Spacecraft body frame


@dataclass
class OrbitalElements:
    """Classical orbital elements."""
    a: float         # Semi-major axis (km)
    e: float         # Eccentricity
    i: float         # Inclination (rad)
    raan: float      # Right ascension of ascending node (rad)
    omega: float     # Argument of periapsis (rad)
    nu: float        # True anomaly (rad)
    
    @classmethod
    def from_state_vector(
        cls,
        r: np.ndarray,
        v: np.ndarray,
        mu: float = MU_EARTH,
    ) -> "OrbitalElements":
        """Convert state vector (position, velocity) to orbital elements."""
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)

        if not (np.isfinite(r_mag) and np.isfinite(v_mag) and r_mag > 1.0):
            return cls(a=7000.0, e=0.0, i=0.0, raan=0.0, omega=0.0, nu=0.0)

        # Specific angular momentum
        h = np.cross(r, v)
        h_mag = np.linalg.norm(h)
        
        # Node vector
        n = np.cross(np.array([0, 0, 1]), h)
        n_mag = np.linalg.norm(n)
        
        # Eccentricity vector
        e_vec = ((v_mag**2 - mu/r_mag) * r - np.dot(r, v) * v) / mu
        e = np.linalg.norm(e_vec)
        
        # Semi-major axis
        energy = v_mag**2 / 2 - mu / r_mag
        if abs(e - 1.0) > 1e-10:
            a = -mu / (2 * energy)
        else:
            a = float('inf')  # Parabolic orbit
        
        # Inclination
        i = np.arccos(np.clip(h[2] / h_mag, -1, 1))
        
        # RAAN
        if n_mag > 1e-10:
            raan = np.arccos(np.clip(n[0] / n_mag, -1, 1))
            if n[1] < 0:
                raan = 2 * np.pi - raan
        else:
            raan = 0.0
        
        # Argument of periapsis
        if n_mag > 1e-10 and e > 1e-10:
            omega = np.arccos(np.clip(np.dot(n, e_vec) / (n_mag * e), -1, 1))
            if e_vec[2] < 0:
                omega = 2 * np.pi - omega
        else:
            omega = 0.0
        
        # True anomaly
        if e > 1e-10:
            nu = np.arccos(np.clip(np.dot(e_vec, r) / (e * r_mag), -1, 1))
            if np.dot(r, v) < 0:
                nu = 2 * np.pi - nu
        else:
            nu = np.arccos(np.clip(np.dot(n, r) / (n_mag * r_mag), -1, 1))
            if r[2] < 0:
                nu = 2 * np.pi - nu
        
        return cls(a=a, e=e, i=i, raan=raan, omega=omega, nu=nu)
    
    def to_state_vector(self, mu: float = MU_EARTH) -> Tuple[np.ndarray, np.ndarray]:
        """Convert orbital elements to state vector."""
        # Semi-latus rectum
        p = self.a * (1 - self.e**2) if self.e < 1 else self.a * (self.e**2 - 1)
        
        # Position in perifocal frame
        r_pqw = np.array([
            p * np.cos(self.nu) / (1 + self.e * np.cos(self.nu)),
            p * np.sin(self.nu) / (1 + self.e * np.cos(self.nu)),
            0.0
        ])
        
        # Velocity in perifocal frame
        v_pqw = np.array([
            -np.sqrt(mu / p) * np.sin(self.nu),
            np.sqrt(mu / p) * (self.e + np.cos(self.nu)),
            0.0
        ])
        
        # Rotation matrix from perifocal to ECI
        R = self._perifocal_to_eci_matrix()
        
        r = R @ r_pqw
        v = R @ v_pqw
        
        return r, v
    
    def _perifocal_to_eci_matrix(self) -> np.ndarray:
        """Compute rotation matrix from perifocal to ECI frame."""
        cos_raan = np.cos(self.raan)
        sin_raan = np.sin(self.raan)
        cos_i = np.cos(self.i)
        sin_i = np.sin(self.i)
        cos_omega = np.cos(self.omega)
        sin_omega = np.sin(self.omega)
        
        R = np.array([
            [cos_raan * cos_omega - sin_raan * sin_omega * cos_i,
             -cos_raan * sin_omega - sin_raan * cos_omega * cos_i,
             sin_raan * sin_i],
            [sin_raan * cos_omega + cos_raan * sin_omega * cos_i,
             -sin_raan * sin_omega + cos_raan * cos_omega * cos_i,
             -cos_raan * sin_i],
            [sin_omega * sin_i,
             cos_omega * sin_i,
             cos_i]
        ])
        
        return R
    
    @property
    def period(self) -> float:
        """Orbital period in seconds."""
        return 2 * np.pi * np.sqrt(self.a**3 / MU_EARTH)
    
    @property
    def altitude(self) -> float:
        """Current altitude above Earth surface in km."""
        r_mag = self.a * (1 - self.e**2) / (1 + self.e * np.cos(self.nu))
        return r_mag - EARTH_RADIUS


@dataclass
class SpacecraftState:
    """Complete spacecraft state."""
    position: np.ndarray      # Position in ECI frame (km)
    velocity: np.ndarray      # Velocity in ECI frame (km/s)
    attitude: np.ndarray      # Quaternion [w, x, y, z]
    angular_velocity: np.ndarray  # Angular velocity in body frame (rad/s)
    mass: float               # Current mass (kg)
    fuel_mass: float          # Remaining fuel (kg)
    time: float = 0.0         # Simulation time (s)
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.attitude = np.asarray(self.attitude, dtype=np.float64)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64)
        
        q_norm = np.linalg.norm(self.attitude)
        if np.isfinite(q_norm) and q_norm > 1e-10:
            self.attitude = self.attitude / q_norm
        else:
            self.attitude = np.array([1.0, 0.0, 0.0, 0.0])
    
    @property
    def orbital_elements(self) -> OrbitalElements:
        """Get current orbital elements."""
        return OrbitalElements.from_state_vector(self.position, self.velocity)
    
    @property
    def altitude(self) -> float:
        """Current altitude in km."""
        return np.linalg.norm(self.position) - EARTH_RADIUS
    
    @property
    def speed(self) -> float:
        """Current speed in km/s."""
        return np.linalg.norm(self.velocity)
    
    def copy(self) -> "SpacecraftState":
        """Create a deep copy of the state."""
        return SpacecraftState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            attitude=self.attitude.copy(),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass,
            fuel_mass=self.fuel_mass,
            time=self.time,
        )


class OrbitalMechanics:
    """
    Orbital mechanics calculations and propagation.
    Supports various perturbation models.
    """
    
    def __init__(
        self,
        mu: float = MU_EARTH,
        include_j2: bool = True,
        include_drag: bool = True,
        include_srp: bool = False,
    ):
        """
        Initialize orbital mechanics.
        
        Args:
            mu: Gravitational parameter
            include_j2: Include J2 perturbation
            include_drag: Include atmospheric drag
            include_srp: Include solar radiation pressure
        """
        self.mu = mu
        self.include_j2 = include_j2
        self.include_drag = include_drag
        self.include_srp = include_srp
    
    def gravitational_acceleration(self, r: np.ndarray) -> np.ndarray:
        """
        Compute gravitational acceleration at position r.
        
        Args:
            r: Position vector (km)
            
        Returns:
            Acceleration vector (km/s²)
        """
        r_mag = np.linalg.norm(r)

        if not np.isfinite(r_mag) or r_mag < 1.0:
            return np.zeros(3)

        a_grav = -self.mu * r / r_mag**3
        
        if self.include_j2:
            a_j2 = self._j2_perturbation(r, r_mag)
            a_grav += a_j2
        
        return a_grav
    
    def _j2_perturbation(self, r: np.ndarray, r_mag: float) -> np.ndarray:
        """Compute J2 perturbation acceleration."""
        z2_r2 = (r[2] / r_mag)**2
        factor = 1.5 * J2 * self.mu * EARTH_RADIUS**2 / r_mag**4
        
        a_j2 = factor * np.array([
            r[0] / r_mag * (5 * z2_r2 - 1),
            r[1] / r_mag * (5 * z2_r2 - 1),
            r[2] / r_mag * (5 * z2_r2 - 3)
        ])
        
        return a_j2
    
    def atmospheric_drag(
        self,
        r: np.ndarray,
        v: np.ndarray,
        area: float,
        mass: float,
        cd: float = 2.2,
    ) -> np.ndarray:
        """
        Compute atmospheric drag acceleration.
        
        Args:
            r: Position (km)
            v: Velocity (km/s)
            area: Cross-sectional area (m²)
            mass: Spacecraft mass (kg)
            cd: Drag coefficient
            
        Returns:
            Drag acceleration (km/s²)
        """
        if not self.include_drag:
            return np.zeros(3)
        
        altitude = np.linalg.norm(r) - EARTH_RADIUS
        
        # Exponential atmosphere model
        rho = self._atmospheric_density(altitude)
        
        if rho < 1e-20:
            return np.zeros(3)
        
        # Relative velocity (assuming atmosphere co-rotates)
        v_rel = v  # Simplified: ignoring Earth rotation
        v_rel_mag = np.linalg.norm(v_rel)
        
        # Drag force: F = -0.5 * rho * Cd * A * v² * v_hat
        # Convert area from m² to km² and adjust units
        area_km2 = area * 1e-6
        a_drag = -0.5 * rho * cd * area_km2 * v_rel_mag * v_rel / mass
        
        return a_drag
    
    def _atmospheric_density(self, altitude: float) -> float:
        """
        Simple exponential atmosphere model.
        
        Args:
            altitude: Altitude in km
            
        Returns:
            Atmospheric density in kg/km³
        """
        if altitude < 0:
            return 1.225e9  # Sea level density in kg/km³
        elif altitude > 1000:
            return 0.0
        
        # Scale heights for different altitude ranges (simplified)
        if altitude < 100:
            h0 = 0
            rho0 = 1.225e9
            H = 8.5
        elif altitude < 200:
            h0 = 100
            rho0 = 5.297e2
            H = 29.7
        elif altitude < 400:
            h0 = 200
            rho0 = 2.789e-1
            H = 37.1
        elif altitude < 600:
            h0 = 400
            rho0 = 1.585e-5
            H = 45.5
        else:
            h0 = 600
            rho0 = 2.135e-9
            H = 53.3
        
        return rho0 * np.exp(-(altitude - h0) / H)
    
    def propagate_rk4(
        self,
        state: SpacecraftState,
        dt: float,
        thrust: np.ndarray = None,
        area: float = 10.0,
    ) -> SpacecraftState:
        """
        Propagate state using RK4 integration.
        
        Args:
            state: Current spacecraft state
            dt: Time step (seconds)
            thrust: Thrust vector in body frame (N)
            area: Cross-sectional area for drag (m²)
            
        Returns:
            New spacecraft state
        """
        def state_derivative(r, v, mass):
            if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v))):
                return np.zeros(3), np.zeros(3)
            a = self.gravitational_acceleration(r)
            a += self.atmospheric_drag(r, v, area, mass)
            
            if thrust is not None and state.fuel_mass > 0:
                R = self._quaternion_to_matrix(state.attitude)
                thrust_inertial = R @ thrust / 1000
                a += thrust_inertial / mass
            
            return v, a
        
        r, v = state.position, state.velocity
        mass = state.mass
        
        # RK4 integration
        k1_r, k1_v = state_derivative(r, v, mass)
        k2_r, k2_v = state_derivative(r + 0.5*dt*k1_r, v + 0.5*dt*k1_v, mass)
        k3_r, k3_v = state_derivative(r + 0.5*dt*k2_r, v + 0.5*dt*k2_v, mass)
        k4_r, k4_v = state_derivative(r + dt*k3_r, v + dt*k3_v, mass)
        
        new_r = r + dt/6 * (k1_r + 2*k2_r + 2*k3_r + k4_r)
        new_v = v + dt/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        r_mag_new = np.linalg.norm(new_r)
        v_mag_new = np.linalg.norm(new_v)
        if not (np.isfinite(r_mag_new) and np.isfinite(v_mag_new)
                and r_mag_new > EARTH_RADIUS and r_mag_new < 1e6
                and v_mag_new < 100.0):
            new_r = r.copy()
            new_v = v.copy()
        
        # Update fuel consumption (if thrusting)
        new_fuel = state.fuel_mass
        new_mass = state.mass
        if thrust is not None and state.fuel_mass > 0:
            thrust_mag = np.linalg.norm(thrust)
            if thrust_mag > 0:
                isp = 300  # Specific impulse (s)
                g0 = 9.81e-3  # km/s²
                mdot = thrust_mag / (isp * g0 * 1000)  # kg/s
                fuel_used = mdot * dt
                new_fuel = max(0, state.fuel_mass - fuel_used)
                new_mass = state.mass - (state.fuel_mass - new_fuel)
        
        # Propagate attitude (simplified: constant angular velocity)
        new_attitude = self._propagate_quaternion(
            state.attitude, state.angular_velocity, dt
        )
        
        return SpacecraftState(
            position=new_r,
            velocity=new_v,
            attitude=new_attitude,
            angular_velocity=state.angular_velocity.copy(),
            mass=new_mass,
            fuel_mass=new_fuel,
            time=state.time + dt,
        )
    
    def _quaternion_to_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
    
    def _propagate_quaternion(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Propagate quaternion given angular velocity."""
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(omega))):
            return np.array([1.0, 0.0, 0.0, 0.0])

        omega_mag = np.linalg.norm(omega)
        
        if omega_mag < 1e-10:
            return q

        MAX_OMEGA = 10.0  # rad/s physical sanity limit
        if omega_mag > MAX_OMEGA:
            omega = omega * (MAX_OMEGA / omega_mag)
            omega_mag = MAX_OMEGA

        half_angle = 0.5 * omega_mag * dt
        axis = omega / omega_mag
        
        dq = np.array([
            np.cos(half_angle),
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle)
        ])
        
        new_q = self._quaternion_multiply(q, dq)
        q_norm = np.linalg.norm(new_q)
        if not np.isfinite(q_norm) or q_norm < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return new_q / q_norm
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])


class SpacecraftDynamics:
    """
    Complete spacecraft dynamics including attitude control.
    """
    
    def __init__(
        self,
        mass: float = 500.0,
        inertia: np.ndarray = None,
        max_thrust: float = 100.0,
        max_torque: float = 10.0,
        specific_impulse: float = 300.0,
    ):
        """
        Initialize spacecraft dynamics.
        
        Args:
            mass: Dry mass (kg)
            inertia: Moment of inertia tensor (kg⋅m²)
            max_thrust: Maximum thrust (N)
            max_torque: Maximum torque (N⋅m)
            specific_impulse: Engine specific impulse (s)
        """
        self.mass = mass
        self.inertia = inertia if inertia is not None else np.diag([100, 100, 50])
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.max_thrust = max_thrust
        self.max_torque = max_torque
        self.isp = specific_impulse
        
        self.orbital_mechanics = OrbitalMechanics()
    
    def apply_control(
        self,
        state: SpacecraftState,
        thrust_cmd: np.ndarray,
        torque_cmd: np.ndarray,
        dt: float,
    ) -> SpacecraftState:
        """
        Apply thrust and torque commands to spacecraft.
        
        Args:
            state: Current state
            thrust_cmd: Thrust command in body frame (N), shape (3,)
            torque_cmd: Torque command in body frame (N⋅m), shape (3,)
            dt: Time step (s)
            
        Returns:
            Updated spacecraft state
        """
        # Clip commands to limits
        thrust = np.clip(thrust_cmd, -self.max_thrust, self.max_thrust)
        torque = np.clip(torque_cmd, -self.max_torque, self.max_torque)
        
        # Propagate orbital state
        new_state = self.orbital_mechanics.propagate_rk4(
            state, dt, thrust=thrust
        )
        
        # Update angular velocity (Euler's equation)
        omega = np.clip(state.angular_velocity, -10.0, 10.0)
        omega_dot = self.inertia_inv @ (
            torque - np.cross(omega, self.inertia @ omega)
        )
        new_omega = np.clip(omega + omega_dot * dt, -10.0, 10.0)
        
        new_state.angular_velocity = new_omega
        
        return new_state
    
    def compute_delta_v(
        self,
        thrust: float,
        burn_time: float,
        mass: float,
    ) -> float:
        """
        Compute delta-v from a burn.
        
        Args:
            thrust: Thrust magnitude (N)
            burn_time: Burn duration (s)
            mass: Initial mass (kg)
            
        Returns:
            Delta-v (km/s)
        """
        g0 = 9.81e-3  # km/s²
        mdot = thrust / (self.isp * g0 * 1000)
        mass_ratio = mass / (mass - mdot * burn_time)
        
        return self.isp * g0 * np.log(mass_ratio)


@dataclass 
class DebrisObject:
    """Represents a piece of space debris."""
    position: np.ndarray      # Position in ECI (km)
    velocity: np.ndarray      # Velocity in ECI (km/s)
    size: float               # Characteristic size (m)
    mass: float               # Estimated mass (kg)
    debris_type: str = "unknown"  # Type classification
    object_id: str = ""       # Unique identifier
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64)
        
        if not self.object_id:
            import uuid
            self.object_id = str(uuid.uuid4())[:8]
    
    def propagate(self, dt: float, mechanics: OrbitalMechanics) -> None:
        """Propagate debris state."""
        a = mechanics.gravitational_acceleration(self.position)
        new_vel = self.velocity + a * dt
        new_pos = self.position + new_vel * dt
        if np.all(np.isfinite(new_pos)) and np.all(np.isfinite(new_vel)):
            self.velocity = new_vel
            self.position = new_pos
    
    @property
    def altitude(self) -> float:
        """Current altitude in km."""
        return np.linalg.norm(self.position) - EARTH_RADIUS
    
    def distance_to(self, other_pos: np.ndarray) -> float:
        """Distance to another position in km."""
        return np.linalg.norm(self.position - other_pos)
    
    def relative_velocity(self, other_vel: np.ndarray) -> float:
        """Relative velocity magnitude in km/s."""
        return np.linalg.norm(self.velocity - other_vel)
