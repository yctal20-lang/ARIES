"""
Complete Navigation Module integrating EKF with NN correction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from ....core.base_module import BaseModule
from .ekf import ExtendedKalmanFilter, NavigationState
from .corrector import NNCorrector


class NavigationModule(BaseModule):
    """
    Complete navigation solution combining EKF and neural network correction.
    
    The module:
    1. Runs EKF prediction with IMU data
    2. Updates EKF with GPS and star tracker measurements
    3. Applies NN correction to reduce systematic errors
    4. Provides fused navigation state with uncertainty
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        """
        Initialize navigation module.
        
        Args:
            config: Module configuration
            device: Compute device
        """
        super().__init__(
            name="navigation",
            config=config,
            device=device,
        )
        
        self.ekf = None
        self.nn_corrector = None
        
        # Configuration
        self.dt = config.get("dt", 0.1)
        self.use_nn_correction = config.get("use_nn_correction", True)
        self.correction_threshold = config.get("correction_threshold", 0.5)
        
        # Measurement dimensions
        self.state_dim = 16
        self.measurement_dim = config.get("measurement_dim", 16)
    
    def _build_model(self) -> nn.Module:
        """Build the neural network corrector."""
        # Initialize EKF
        self.ekf = ExtendedKalmanFilter(
            dt=self.dt,
            mu=self.config.get("mu", 398600.4418),
        )
        
        # Initialize NN corrector
        self.nn_corrector = NNCorrector(
            state_dim=self.state_dim,
            measurement_dim=self.measurement_dim,
            hidden_dim=self.config.get("hidden_dim", 128),
            num_layers=self.config.get("num_layers", 3),
        ).to(self.device)
        
        return self.nn_corrector
    
    def initialize(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        attitude: Optional[np.ndarray] = None,
    ) -> NavigationState:
        """
        Initialize navigation state.
        
        Args:
            position: Initial position [x, y, z] in km
            velocity: Initial velocity [vx, vy, vz] in km/s
            attitude: Initial quaternion [w, x, y, z]
            
        Returns:
            Initial navigation state
        """
        if self.ekf is None:
            self._build_model()
        
        self.ekf.initialize(position, velocity, attitude)
        return self.ekf.get_state()
    
    def _preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess inputs."""
        processed = {}
        
        for key in ["acceleration", "angular_rate", "gps_position", 
                    "gps_velocity", "star_tracker"]:
            if key in inputs and inputs[key] is not None:
                processed[key] = np.asarray(inputs[key], dtype=np.float64)
        
        return processed
    
    def _postprocess(self, outputs: NavigationState) -> Dict[str, Any]:
        """Convert navigation state to output dictionary."""
        return {
            "position": outputs.position,
            "velocity": outputs.velocity,
            "attitude": outputs.attitude,
            "gyro_bias": outputs.gyro_bias,
            "accel_bias": outputs.accel_bias,
            "position_uncertainty": outputs.position_uncertainty,
            "velocity_uncertainty": outputs.velocity_uncertainty,
            "timestamp": outputs.timestamp,
        }
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process navigation update.
        
        Args:
            inputs: Dictionary with:
                - acceleration: IMU acceleration [ax, ay, az] in m/s²
                - angular_rate: IMU angular rate [wx, wy, wz] in rad/s
                - gps_position: GPS position [x, y, z] in km (optional)
                - gps_velocity: GPS velocity [vx, vy, vz] in km/s (optional)
                - star_tracker: Star tracker attitude quaternion (optional)
                - dt: Time step (optional, uses default)
                
        Returns:
            Navigation state dictionary
        """
        if self.ekf is None:
            raise RuntimeError("Navigation module not initialized. Call initialize() first.")
        
        processed = self._preprocess(inputs)
        dt = inputs.get("dt", self.dt)
        
        # EKF Prediction
        state = self.ekf.predict(
            dt=dt,
            acceleration=processed.get("acceleration"),
            angular_rate=processed.get("angular_rate"),
        )
        
        # EKF Updates
        if "gps_position" in processed and "gps_velocity" in processed:
            state = self.ekf.update_gps(
                processed["gps_position"],
                processed["gps_velocity"],
            )
        
        if "star_tracker" in processed:
            state = self.ekf.update_star_tracker(processed["star_tracker"])
        
        # Neural network correction
        if self.use_nn_correction and self.nn_corrector is not None:
            # Prepare measurement vector
            measurements = self._prepare_measurements(processed)
            
            # Apply NN correction
            corrected_state, confidence = self.nn_corrector.correct_state(
                state.to_vector(),
                measurements,
                device=self.device,
            )
            
            # Only apply correction if confidence is above threshold
            if confidence >= self.correction_threshold:
                # Update EKF state with correction
                self.ekf.state = corrected_state
                state = self.ekf.get_state()
        
        return self._postprocess(state)
    
    def _prepare_measurements(self, processed: Dict[str, Any]) -> np.ndarray:
        """Prepare measurement vector for NN corrector."""
        measurements = np.zeros(self.measurement_dim)
        
        idx = 0
        
        # GPS position (3)
        if "gps_position" in processed:
            measurements[idx:idx+3] = processed["gps_position"]
        idx += 3
        
        # GPS velocity (3)
        if "gps_velocity" in processed:
            measurements[idx:idx+3] = processed["gps_velocity"]
        idx += 3
        
        # Star tracker attitude (4)
        if "star_tracker" in processed:
            measurements[idx:idx+4] = processed["star_tracker"]
        idx += 4
        
        # IMU acceleration (3)
        if "acceleration" in processed:
            measurements[idx:idx+3] = processed["acceleration"] / 1000  # Normalize
        idx += 3
        
        # IMU angular rate (3)
        if "angular_rate" in processed:
            measurements[idx:idx+3] = processed["angular_rate"]
        
        return measurements
    
    def _fallback(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback to pure EKF without NN correction.
        """
        if self.ekf is None:
            # Return default state if completely uninitialized
            return {
                "position": np.zeros(3),
                "velocity": np.zeros(3),
                "attitude": np.array([1, 0, 0, 0]),
                "fallback": True,
                "error": "EKF not initialized",
            }
        
        # Run EKF without NN correction
        processed = self._preprocess(inputs)
        dt = inputs.get("dt", self.dt)
        
        state = self.ekf.predict(
            dt=dt,
            acceleration=processed.get("acceleration"),
            angular_rate=processed.get("angular_rate"),
        )
        
        if "gps_position" in processed and "gps_velocity" in processed:
            state = self.ekf.update_gps(
                processed["gps_position"],
                processed["gps_velocity"],
            )
        
        if "star_tracker" in processed:
            state = self.ekf.update_star_tracker(processed["star_tracker"])
        
        result = self._postprocess(state)
        result["fallback"] = True
        return result
    
    def save(self, path: Path) -> None:
        """Save module state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save NN corrector
        if self.nn_corrector is not None:
            torch.save(
                self.nn_corrector.state_dict(),
                path / "nn_corrector.pt"
            )
        
        # Save EKF state
        if self.ekf is not None:
            np.savez(
                path / "ekf_state.npz",
                state=self.ekf.state,
                covariance=self.ekf.P,
                timestamp=self.ekf.timestamp,
            )
    
    def load(self, path: Path) -> None:
        """Load module state."""
        path = Path(path)
        
        # Ensure models are built
        if self.nn_corrector is None:
            self._build_model()
        
        # Load NN corrector
        nn_path = path / "nn_corrector.pt"
        if nn_path.exists():
            self.nn_corrector.load_state_dict(
                torch.load(nn_path, map_location=self.device)
            )
        
        # Load EKF state
        ekf_path = path / "ekf_state.npz"
        if ekf_path.exists():
            data = np.load(ekf_path)
            self.ekf.state = data["state"]
            self.ekf.P = data["covariance"]
            self.ekf.timestamp = float(data["timestamp"])
