"""
Fail-safe mechanisms for Space Debris Collector AI System.
Provides fallback algorithms and safety controllers.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass, field
import time
import threading
from loguru import logger


class FallbackMode(Enum):
    """Fallback operation modes."""
    NORMAL = "normal"           # Normal AI operation
    DEGRADED = "degraded"       # Partial fallback
    CLASSICAL = "classical"     # Full classical algorithms
    SAFE_MODE = "safe_mode"     # Minimum operations for survival
    EMERGENCY = "emergency"     # Emergency stop/hold


@dataclass
class SafetyState:
    """Current safety system state."""
    mode: FallbackMode
    active_fallbacks: List[str]
    module_health: Dict[str, bool]
    last_update: float
    error_count: int
    consecutive_failures: int
    
    @property
    def is_safe(self) -> bool:
        return self.mode in [FallbackMode.NORMAL, FallbackMode.DEGRADED]


class ClassicalController:
    """
    Classical control algorithms as fallback.
    Implements simple but reliable control strategies.
    """
    
    def __init__(
        self,
        max_thrust: float = 100.0,
        max_torque: float = 10.0,
    ):
        self.max_thrust = max_thrust
        self.max_torque = max_torque
    
    def collision_avoidance(
        self,
        threat_direction: np.ndarray,
        threat_distance: float,
        threat_velocity: float,
    ) -> np.ndarray:
        """
        Classical collision avoidance using potential fields.
        
        Args:
            threat_direction: Unit vector towards threat
            threat_distance: Distance to threat (km)
            threat_velocity: Closing velocity (km/s)
            
        Returns:
            Thrust vector (N)
        """
        # Repulsive force inversely proportional to distance squared
        if threat_distance < 0.01:  # Very close
            force_magnitude = self.max_thrust
        else:
            force_magnitude = min(
                self.max_thrust,
                self.max_thrust * (0.1 / threat_distance) ** 2
            )
        
        # Scale by closing velocity
        velocity_factor = min(1.0, threat_velocity / 0.01)
        force_magnitude *= (0.5 + 0.5 * velocity_factor)
        
        # Thrust away from threat
        thrust = -threat_direction * force_magnitude
        
        return thrust
    
    def attitude_hold(
        self,
        current_attitude: np.ndarray,
        target_attitude: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> np.ndarray:
        """
        PD controller for attitude hold.
        
        Args:
            current_attitude: Current quaternion [w, x, y, z]
            target_attitude: Target quaternion
            angular_velocity: Current angular velocity (rad/s)
            
        Returns:
            Torque command (N·m)
        """
        # Quaternion error
        q_error = self._quaternion_error(current_attitude, target_attitude)
        
        # Extract rotation axis and angle from error quaternion
        angle = 2 * np.arccos(np.clip(q_error[0], -1, 1))
        if abs(angle) < 1e-6:
            axis = np.array([0, 0, 1])
        else:
            axis = q_error[1:4] / np.sin(angle / 2)
        
        # PD control
        kp = 5.0   # Proportional gain
        kd = 2.0   # Derivative gain
        
        torque = kp * angle * axis - kd * angular_velocity
        
        # Clip to limits
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        
        return torque
    
    def station_keeping(
        self,
        position_error: np.ndarray,
        velocity: np.ndarray,
    ) -> np.ndarray:
        """
        Simple station keeping with PD control.
        
        Args:
            position_error: Error from target position (km)
            velocity: Current velocity (km/s)
            
        Returns:
            Thrust command (N)
        """
        kp = 0.1   # Position gain
        kd = 1.0   # Velocity gain
        
        # PD control
        thrust = kp * position_error - kd * velocity
        
        # Scale to force (N)
        thrust = thrust * 1000  # km/s² to m/s² scale factor
        
        # Clip to limits
        magnitude = np.linalg.norm(thrust)
        if magnitude > self.max_thrust:
            thrust = thrust * (self.max_thrust / magnitude)
        
        return thrust
    
    def _quaternion_error(
        self,
        current: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """Compute quaternion error (target relative to current)."""
        # Conjugate of current
        current_conj = np.array([current[0], -current[1], -current[2], -current[3]])
        
        # Error = target * conj(current)
        return self._quaternion_multiply(target, current_conj)
    
    def _quaternion_multiply(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
    ) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])


class FailsafeController:
    """
    Central fail-safe controller managing all safety mechanisms.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize fail-safe controller.
        
        Args:
            config: Configuration dictionary
        """
        config = config or {}
        
        self.max_consecutive_failures = config.get("max_consecutive_failures", 3)
        self.error_threshold = config.get("error_threshold", 10)
        self.recovery_time = config.get("recovery_time", 30.0)  # seconds
        
        self.mode = FallbackMode.NORMAL
        self.module_health: Dict[str, bool] = {}
        self.module_failures: Dict[str, int] = {}
        self.active_fallbacks: List[str] = []
        self.error_count = 0
        self.consecutive_failures = 0
        self.last_failure_time = 0.0
        self.last_recovery_attempt = 0.0
        
        self.classical_controller = ClassicalController(
            max_thrust=config.get("max_thrust", 100.0),
            max_torque=config.get("max_torque", 10.0),
        )
        
        self._lock = threading.RLock()
        
        # Registered modules and their fallback functions
        self._fallback_handlers: Dict[str, Callable] = {}
        
        logger.info("FailsafeController initialized")
    
    def register_module(
        self,
        module_name: str,
        fallback_handler: Optional[Callable] = None,
    ) -> None:
        """
        Register a module with the fail-safe system.
        
        Args:
            module_name: Name of the module
            fallback_handler: Optional fallback function
        """
        with self._lock:
            self.module_health[module_name] = True
            self.module_failures[module_name] = 0
            
            if fallback_handler:
                self._fallback_handlers[module_name] = fallback_handler
            
            logger.debug(f"Registered module: {module_name}")
    
    def report_failure(
        self,
        module_name: str,
        error: Optional[str] = None,
    ) -> FallbackMode:
        """
        Report a module failure.
        
        Args:
            module_name: Name of failing module
            error: Optional error message
            
        Returns:
            Current fallback mode
        """
        with self._lock:
            self.error_count += 1
            self.consecutive_failures += 1
            self.last_failure_time = time.time()
            
            if module_name in self.module_failures:
                self.module_failures[module_name] += 1
                
                if self.module_failures[module_name] >= self.max_consecutive_failures:
                    self.module_health[module_name] = False
                    if module_name not in self.active_fallbacks:
                        self.active_fallbacks.append(module_name)
            
            logger.warning(
                f"Module failure: {module_name}, "
                f"consecutive: {self.consecutive_failures}, "
                f"error: {error}"
            )
            
            # Update mode based on failures
            self._update_mode()
            
            return self.mode
    
    def report_success(self, module_name: str) -> None:
        """
        Report successful module operation.
        
        Args:
            module_name: Name of module
        """
        with self._lock:
            self.consecutive_failures = 0
            
            if module_name in self.module_failures:
                # Gradual recovery
                self.module_failures[module_name] = max(
                    0, self.module_failures[module_name] - 1
                )
                
                if self.module_failures[module_name] == 0:
                    self.module_health[module_name] = True
                    if module_name in self.active_fallbacks:
                        self.active_fallbacks.remove(module_name)
            
            # Attempt mode recovery
            self._attempt_recovery()
    
    def _update_mode(self) -> None:
        """Update fallback mode based on system state."""
        unhealthy_count = sum(1 for h in self.module_health.values() if not h)
        total_modules = len(self.module_health)
        
        if total_modules == 0:
            return
        
        failure_ratio = unhealthy_count / total_modules
        
        if self.consecutive_failures >= self.max_consecutive_failures * 2:
            self.mode = FallbackMode.EMERGENCY
        elif failure_ratio > 0.5:
            self.mode = FallbackMode.SAFE_MODE
        elif failure_ratio > 0.2 or self.consecutive_failures >= self.max_consecutive_failures:
            self.mode = FallbackMode.CLASSICAL
        elif unhealthy_count > 0:
            self.mode = FallbackMode.DEGRADED
        else:
            self.mode = FallbackMode.NORMAL
        
        logger.info(f"Failsafe mode updated to: {self.mode.value}")
    
    def _attempt_recovery(self) -> None:
        """Attempt to recover to normal mode."""
        current_time = time.time()
        
        if current_time - self.last_failure_time > self.recovery_time:
            if current_time - self.last_recovery_attempt > self.recovery_time / 2:
                self.last_recovery_attempt = current_time
                
                # Check if all modules are healthy
                if all(self.module_health.values()):
                    self.mode = FallbackMode.NORMAL
                    self.active_fallbacks.clear()
                    logger.info("Recovered to normal mode")
    
    def get_fallback_action(
        self,
        module_name: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Get fallback action for a failed module.
        
        Args:
            module_name: Name of the module
            inputs: Input data for the module
            
        Returns:
            Fallback action/output
        """
        # Check for registered fallback handler
        if module_name in self._fallback_handlers:
            try:
                return self._fallback_handlers[module_name](inputs)
            except Exception as e:
                logger.error(f"Fallback handler failed: {e}")
        
        # Default fallbacks based on module type
        if "collision" in module_name.lower():
            return self._collision_avoidance_fallback(inputs)
        elif "navigation" in module_name.lower():
            return self._navigation_fallback(inputs)
        elif "energy" in module_name.lower():
            return self._energy_fallback(inputs)
        
        return {"fallback": True, "error": "No fallback available"}
    
    def _collision_avoidance_fallback(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fallback for collision avoidance."""
        threat_direction = inputs.get("threat_direction", np.array([1, 0, 0]))
        threat_distance = inputs.get("threat_distance", 1.0)
        threat_velocity = inputs.get("threat_velocity", 0.01)
        
        thrust = self.classical_controller.collision_avoidance(
            np.asarray(threat_direction),
            threat_distance,
            threat_velocity,
        )
        
        return {
            "action": np.concatenate([thrust / 100, np.zeros(3)]),
            "avoidance_active": True,
            "fallback": True,
        }
    
    def _navigation_fallback(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fallback for navigation (dead reckoning)."""
        # Simple dead reckoning using last known velocity
        position = inputs.get("last_position", np.zeros(3))
        velocity = inputs.get("last_velocity", np.zeros(3))
        dt = inputs.get("dt", 1.0)
        
        new_position = position + velocity * dt
        
        return {
            "position": new_position,
            "velocity": velocity,
            "attitude": inputs.get("last_attitude", np.array([1, 0, 0, 0])),
            "fallback": True,
        }
    
    def _energy_fallback(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fallback for energy management (conservative mode)."""
        return {
            "mode": "emergency",
            "subsystem_allocation": {
                "navigation": 0.3,
                "collision_avoidance": 0.5,
                "communication": 0.2,
                "thermal_control": 0.3,
                "computer": 0.3,
            },
            "fallback": True,
        }
    
    def get_state(self) -> SafetyState:
        """Get current safety system state."""
        with self._lock:
            return SafetyState(
                mode=self.mode,
                active_fallbacks=self.active_fallbacks.copy(),
                module_health=self.module_health.copy(),
                last_update=time.time(),
                error_count=self.error_count,
                consecutive_failures=self.consecutive_failures,
            )
    
    def emergency_stop(self) -> None:
        """Trigger emergency stop."""
        with self._lock:
            self.mode = FallbackMode.EMERGENCY
            logger.critical("EMERGENCY STOP triggered")
    
    def reset(self) -> None:
        """Reset fail-safe controller to normal state."""
        with self._lock:
            self.mode = FallbackMode.NORMAL
            self.module_health = {k: True for k in self.module_health}
            self.module_failures = {k: 0 for k in self.module_failures}
            self.active_fallbacks.clear()
            self.error_count = 0
            self.consecutive_failures = 0
            logger.info("FailsafeController reset to normal")
