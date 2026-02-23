"""
Extended Kalman Filter for spacecraft navigation.
Fuses GPS, IMU, and star tracker data.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from filterpy.kalman import ExtendedKalmanFilter as FilterPyEKF


@dataclass
class NavigationState:
    """Complete navigation state estimate."""
    position: np.ndarray       # Position [x, y, z] in km
    velocity: np.ndarray       # Velocity [vx, vy, vz] in km/s
    attitude: np.ndarray       # Quaternion [w, x, y, z]
    gyro_bias: np.ndarray      # Gyroscope bias [bx, by, bz] in rad/s
    accel_bias: np.ndarray     # Accelerometer bias [bx, by, bz] in m/s²
    covariance: np.ndarray     # State covariance matrix
    timestamp: float = 0.0     # Timestamp in seconds
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.attitude = np.asarray(self.attitude, dtype=np.float64)
        self.gyro_bias = np.asarray(self.gyro_bias, dtype=np.float64)
        self.accel_bias = np.asarray(self.accel_bias, dtype=np.float64)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)
        
        # Normalize quaternion
        self.attitude = self.attitude / np.linalg.norm(self.attitude)
    
    def to_vector(self) -> np.ndarray:
        """Convert to state vector [16 elements]."""
        return np.concatenate([
            self.position,       # 3
            self.velocity,       # 3
            self.attitude,       # 4
            self.gyro_bias,      # 3
            self.accel_bias,     # 3
        ])
    
    @classmethod
    def from_vector(
        cls,
        state: np.ndarray,
        covariance: np.ndarray,
        timestamp: float = 0.0,
    ) -> "NavigationState":
        """Create from state vector."""
        return cls(
            position=state[0:3],
            velocity=state[3:6],
            attitude=state[6:10],
            gyro_bias=state[10:13],
            accel_bias=state[13:16],
            covariance=covariance,
            timestamp=timestamp,
        )
    
    @property
    def position_uncertainty(self) -> np.ndarray:
        """Position uncertainty (1-sigma) in km."""
        return np.sqrt(np.diag(self.covariance)[0:3])
    
    @property
    def velocity_uncertainty(self) -> np.ndarray:
        """Velocity uncertainty (1-sigma) in km/s."""
        return np.sqrt(np.diag(self.covariance)[3:6])
    
    @property
    def attitude_uncertainty(self) -> np.ndarray:
        """Attitude uncertainty (1-sigma)."""
        return np.sqrt(np.diag(self.covariance)[6:10])


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for spacecraft navigation.
    
    State vector [16]:
        - Position [x, y, z] in km (3)
        - Velocity [vx, vy, vz] in km/s (3)
        - Attitude quaternion [w, x, y, z] (4)
        - Gyroscope bias [bx, by, bz] in rad/s (3)
        - Accelerometer bias [bx, by, bz] in m/s² (3)
    
    Measurements:
        - GPS: Position [x, y, z] and velocity [vx, vy, vz]
        - IMU: Acceleration [ax, ay, az] and angular rate [wx, wy, wz]
        - Star tracker: Attitude quaternion [w, x, y, z]
    """
    
    # State indices
    POS = slice(0, 3)
    VEL = slice(3, 6)
    ATT = slice(6, 10)
    GYRO_BIAS = slice(10, 13)
    ACCEL_BIAS = slice(13, 16)
    
    STATE_DIM = 16
    
    def __init__(
        self,
        dt: float = 0.1,
        process_noise: Optional[np.ndarray] = None,
        gps_noise: Optional[np.ndarray] = None,
        imu_noise: Optional[np.ndarray] = None,
        star_tracker_noise: Optional[np.ndarray] = None,
        mu: float = 398600.4418,  # Earth gravitational parameter km³/s²
    ):
        """
        Initialize EKF.
        
        Args:
            dt: Default timestep (seconds)
            process_noise: Process noise covariance [16x16]
            gps_noise: GPS measurement noise [6x6]
            imu_noise: IMU noise [6x6]
            star_tracker_noise: Star tracker noise [4x4]
            mu: Gravitational parameter
        """
        self.dt = dt
        self.mu = mu
        
        # Initialize state
        self.state = np.zeros(self.STATE_DIM)
        self.state[self.ATT] = [1, 0, 0, 0]  # Identity quaternion
        
        # Initialize covariance
        self.P = np.eye(self.STATE_DIM)
        self.P[self.POS, self.POS] *= 0.1      # 100m position uncertainty
        self.P[self.VEL, self.VEL] *= 0.001   # 1 m/s velocity uncertainty
        self.P[self.ATT, self.ATT] *= 0.01    # Small attitude uncertainty
        self.P[self.GYRO_BIAS, self.GYRO_BIAS] *= 1e-6
        self.P[self.ACCEL_BIAS, self.ACCEL_BIAS] *= 1e-4
        
        # Process noise
        if process_noise is None:
            self.Q = np.eye(self.STATE_DIM) * 1e-6
            self.Q[self.POS, self.POS] *= 1e-8    # Very small position process noise
            self.Q[self.VEL, self.VEL] *= 1e-6   # Small velocity process noise
            self.Q[self.ATT, self.ATT] *= 1e-8
            self.Q[self.GYRO_BIAS, self.GYRO_BIAS] *= 1e-10
            self.Q[self.ACCEL_BIAS, self.ACCEL_BIAS] *= 1e-8
        else:
            self.Q = process_noise
        
        # Measurement noise
        self.R_gps = gps_noise if gps_noise is not None else np.diag([
            0.01, 0.01, 0.01,    # Position noise (km)
            0.001, 0.001, 0.001, # Velocity noise (km/s)
        ])
        
        self.R_imu = imu_noise if imu_noise is not None else np.diag([
            0.01, 0.01, 0.01,    # Accel noise (m/s²)
            0.001, 0.001, 0.001, # Gyro noise (rad/s)
        ])
        
        self.R_star = star_tracker_noise if star_tracker_noise is not None else np.diag([
            0.0001, 0.0001, 0.0001, 0.0001  # Quaternion noise
        ])
        
        # Time tracking
        self.timestamp = 0.0
    
    def initialize(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        attitude: np.ndarray = None,
        timestamp: float = 0.0,
    ) -> None:
        """
        Initialize filter state.
        
        Args:
            position: Initial position [x, y, z] in km
            velocity: Initial velocity [vx, vy, vz] in km/s
            attitude: Initial quaternion [w, x, y, z]
            timestamp: Initial timestamp
        """
        self.state[self.POS] = position
        self.state[self.VEL] = velocity
        
        if attitude is not None:
            self.state[self.ATT] = attitude / np.linalg.norm(attitude)
        else:
            self.state[self.ATT] = [1, 0, 0, 0]
        
        self.state[self.GYRO_BIAS] = 0
        self.state[self.ACCEL_BIAS] = 0
        
        self.timestamp = timestamp
    
    def predict(
        self,
        dt: Optional[float] = None,
        acceleration: Optional[np.ndarray] = None,
        angular_rate: Optional[np.ndarray] = None,
    ) -> NavigationState:
        """
        Prediction step using dynamics model.
        
        Args:
            dt: Time step (uses default if None)
            acceleration: Body-frame acceleration from IMU (m/s²)
            angular_rate: Body-frame angular rate from IMU (rad/s)
            
        Returns:
            Predicted navigation state
        """
        if dt is None:
            dt = self.dt
        
        # Extract current state
        pos = self.state[self.POS].copy()
        vel = self.state[self.VEL].copy()
        att = self.state[self.ATT].copy()
        gyro_bias = self.state[self.GYRO_BIAS].copy()
        accel_bias = self.state[self.ACCEL_BIAS].copy()
        
        # Compute gravitational acceleration
        r = np.linalg.norm(pos)
        a_grav = -self.mu * pos / r**3
        
        # Process IMU measurements if available
        if acceleration is not None and angular_rate is not None:
            # Remove biases
            accel_corrected = acceleration - accel_bias
            gyro_corrected = angular_rate - gyro_bias
            
            # Convert acceleration to inertial frame
            R = self._quaternion_to_rotation(att)
            a_body = R @ (accel_corrected / 1000)  # Convert m/s² to km/s²
            
            # Total acceleration
            a_total = a_grav + a_body
            
            # Update attitude
            att = self._propagate_quaternion(att, gyro_corrected, dt)
        else:
            a_total = a_grav
        
        # Propagate position and velocity (simple Euler for now)
        new_vel = vel + a_total * dt
        new_pos = pos + vel * dt + 0.5 * a_total * dt**2
        
        # Update state
        self.state[self.POS] = new_pos
        self.state[self.VEL] = new_vel
        self.state[self.ATT] = att / np.linalg.norm(att)
        
        # Compute state transition Jacobian
        F = self._compute_state_jacobian(dt, pos, vel)
        
        # Propagate covariance
        self.P = F @ self.P @ F.T + self.Q * dt
        
        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)
        
        self.timestamp += dt
        
        return self.get_state()
    
    def update_gps(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> NavigationState:
        """
        Update with GPS measurement.
        
        Args:
            position: Measured position [x, y, z] in km
            velocity: Measured velocity [vx, vy, vz] in km/s
            
        Returns:
            Updated navigation state
        """
        # Measurement vector
        z = np.concatenate([position, velocity])
        
        # Predicted measurement
        h = np.concatenate([self.state[self.POS], self.state[self.VEL]])
        
        # Measurement Jacobian
        H = np.zeros((6, self.STATE_DIM))
        H[0:3, self.POS] = np.eye(3)
        H[3:6, self.VEL] = np.eye(3)
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        innovation = z - h
        self.state = self.state + K @ innovation
        
        # Normalize quaternion
        self.state[self.ATT] = self.state[self.ATT] / np.linalg.norm(self.state[self.ATT])
        
        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_gps @ K.T
        
        return self.get_state()
    
    def update_star_tracker(
        self,
        attitude: np.ndarray,
    ) -> NavigationState:
        """
        Update with star tracker attitude measurement.
        
        Args:
            attitude: Measured quaternion [w, x, y, z]
            
        Returns:
            Updated navigation state
        """
        # Normalize measurement
        z = attitude / np.linalg.norm(attitude)
        
        # Handle quaternion sign ambiguity
        if np.dot(z, self.state[self.ATT]) < 0:
            z = -z
        
        # Predicted measurement
        h = self.state[self.ATT]
        
        # Measurement Jacobian
        H = np.zeros((4, self.STATE_DIM))
        H[0:4, self.ATT] = np.eye(4)
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R_star
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        innovation = z - h
        self.state = self.state + K @ innovation
        
        # Normalize quaternion
        self.state[self.ATT] = self.state[self.ATT] / np.linalg.norm(self.state[self.ATT])
        
        # Update covariance
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_star @ K.T
        
        return self.get_state()
    
    def get_state(self) -> NavigationState:
        """Get current navigation state."""
        return NavigationState.from_vector(
            self.state.copy(),
            self.P.copy(),
            self.timestamp,
        )
    
    def _compute_state_jacobian(
        self,
        dt: float,
        pos: np.ndarray,
        vel: np.ndarray,
    ) -> np.ndarray:
        """Compute state transition Jacobian."""
        F = np.eye(self.STATE_DIM)
        
        r = np.linalg.norm(pos)
        
        # Position derivative w.r.t. velocity
        F[self.POS, self.VEL] = np.eye(3) * dt
        
        # Velocity derivative w.r.t. position (gravity gradient)
        r5 = r**5
        F[self.VEL, self.POS] = -self.mu * (
            np.eye(3) / r**3 - 3 * np.outer(pos, pos) / r5
        ) * dt
        
        return F
    
    def _quaternion_to_rotation(self, q: np.ndarray) -> np.ndarray:
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
        """Propagate quaternion with angular velocity."""
        omega_mag = np.linalg.norm(omega)
        
        if omega_mag < 1e-10:
            return q
        
        half_angle = 0.5 * omega_mag * dt
        axis = omega / omega_mag
        
        dq = np.array([
            np.cos(half_angle),
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle)
        ])
        
        # Quaternion multiplication q * dq
        return self._quaternion_multiply(q, dq)
    
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
