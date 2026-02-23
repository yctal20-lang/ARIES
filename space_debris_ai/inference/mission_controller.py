"""
Central Mission Controller integrating all AI modules.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import time
import threading
from loguru import logger

from ..core.message_bus import MessageBus, Message, MessageType, get_message_bus
from ..core.config import SystemConfig, MissionMode
from ..safety.failsafe import FailsafeController, FallbackMode
from ..safety.watchdog import WatchdogManager


class MissionPhase(Enum):
    """Mission phases."""
    IDLE = "idle"
    SEARCH = "search"
    APPROACH = "approach"
    CAPTURE = "capture"
    STORE = "store"
    DISPOSAL = "disposal"
    EMERGENCY = "emergency"


@dataclass
class MissionState:
    """Current mission state."""
    phase: MissionPhase
    mode: MissionMode
    target_debris_id: Optional[str]
    captured_count: int
    fuel_remaining: float
    battery_soc: float
    position: np.ndarray
    velocity: np.ndarray
    time_elapsed: float
    alerts: List[str]


class MissionController:
    """
    Central controller orchestrating all AI modules.
    
    Manages:
    - Module lifecycle and coordination
    - Mission state machine
    - Safety monitoring
    - Decision making
    """
    
    def __init__(
        self,
        config: Optional[SystemConfig] = None,
    ):
        """
        Initialize mission controller.
        
        Args:
            config: System configuration
        """
        self.config = config or SystemConfig()
        self.device = self.config.neural_network.device
        
        # State
        self.phase = MissionPhase.IDLE
        self.mode = self.config.mission_mode
        self.running = False
        
        # Counters
        self.captured_count = 0
        self.avoided_collisions = 0
        self.time_elapsed = 0.0
        
        # Current target
        self.target_debris_id: Optional[str] = None
        
        # Infrastructure
        self.message_bus = get_message_bus()
        self.failsafe = FailsafeController(self.config.safety.model_dump())
        self.watchdog_manager = WatchdogManager(check_interval=0.1)
        
        # Modules (lazy loaded)
        self._modules: Dict[str, Any] = {}
        self._module_configs: Dict[str, Dict] = {}
        
        # Threading
        self._lock = threading.RLock()
        
        # Subscribe to messages
        self._setup_subscriptions()
        
        logger.info("MissionController initialized")
    
    def _setup_subscriptions(self):
        """Set up message bus subscriptions."""
        self.message_bus.subscribe(
            "mission_controller",
            self._handle_message,
            [MessageType.EMERGENCY, MessageType.COLLISION, MessageType.SAFETY],
        )
    
    def _handle_message(self, message: Message):
        """Handle incoming messages."""
        if message.msg_type == MessageType.EMERGENCY:
            self._handle_emergency(message)
        elif message.msg_type == MessageType.COLLISION:
            self._handle_collision_warning(message)
        elif message.msg_type == MessageType.SAFETY:
            self._handle_safety_alert(message)
    
    def _handle_emergency(self, message: Message):
        """Handle emergency message."""
        logger.critical(f"EMERGENCY: {message.payload}")
        self.phase = MissionPhase.EMERGENCY
        self.failsafe.emergency_stop()
    
    def _handle_collision_warning(self, message: Message):
        """Handle collision warning."""
        logger.warning(f"Collision warning: {message.payload}")
        self.avoided_collisions += 1
    
    def _handle_safety_alert(self, message: Message):
        """Handle safety alert."""
        logger.warning(f"Safety alert: {message.payload}")
    
    def register_module(
        self,
        name: str,
        module: Any,
        config: Optional[Dict] = None,
    ):
        """
        Register an AI module.
        
        Args:
            name: Module name
            module: Module instance
            config: Module configuration
        """
        with self._lock:
            self._modules[name] = module
            self._module_configs[name] = config or {}
            
            # Register with failsafe
            fallback = getattr(module, '_fallback', None)
            self.failsafe.register_module(name, fallback)
            
            # Add watchdog
            timeout = config.get("timeout", 5.0) if config else 5.0
            self.watchdog_manager.add_watchdog(
                name,
                timeout=timeout,
                on_timeout=lambda n=name: self._on_module_timeout(n),
            )
            
            logger.info(f"Registered module: {name}")
    
    def _on_module_timeout(self, module_name: str):
        """Handle module timeout."""
        logger.error(f"Module timeout: {module_name}")
        self.failsafe.report_failure(module_name, "Watchdog timeout")
    
    def get_module(self, name: str) -> Optional[Any]:
        """Get a registered module."""
        return self._modules.get(name)
    
    def start(self):
        """Start the mission controller."""
        if self.running:
            return
        
        self.running = True
        self.message_bus.start()
        self.watchdog_manager.start()
        self.phase = MissionPhase.SEARCH
        
        logger.info("MissionController started")
    
    def stop(self):
        """Stop the mission controller."""
        self.running = False
        self.watchdog_manager.stop()
        self.message_bus.stop()
        self.phase = MissionPhase.IDLE
        
        logger.info("MissionController stopped")
    
    def step(
        self,
        sensor_data: Dict[str, Any],
        dt: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Execute one control step.
        
        Args:
            sensor_data: Current sensor readings
            dt: Time step
            
        Returns:
            Control commands
        """
        self.time_elapsed += dt
        
        # Check safety
        safety_state = self.failsafe.get_state()
        if safety_state.mode == FallbackMode.EMERGENCY:
            return self._emergency_response(sensor_data)
        
        # Run modules based on current phase
        results = {}
        commands = {
            "thrust": np.zeros(3),
            "torque": np.zeros(3),
            "gripper": 0.0,
        }
        
        # Navigation (always active)
        if "navigation" in self._modules:
            try:
                nav_result = self._modules["navigation"].forward(sensor_data)
                results["navigation"] = nav_result
                self.watchdog_manager.feed("navigation")
                self.failsafe.report_success("navigation")
            except Exception as e:
                self.failsafe.report_failure("navigation", str(e))
                nav_result = self.failsafe.get_fallback_action("navigation", sensor_data)
                results["navigation"] = nav_result
        
        # Collision avoidance (always active)
        if "collision_avoidance" in self._modules:
            try:
                col_result = self._modules["collision_avoidance"].forward(sensor_data)
                results["collision_avoidance"] = col_result
                self.watchdog_manager.feed("collision_avoidance")
                self.failsafe.report_success("collision_avoidance")
                
                if col_result.get("avoidance_active"):
                    commands["thrust"] = col_result["action"][:3] * self.config.spacecraft.max_thrust
                    commands["torque"] = col_result["action"][3:6] * self.config.spacecraft.max_torque
            except Exception as e:
                self.failsafe.report_failure("collision_avoidance", str(e))
        
        # Energy management
        if "energy_management" in self._modules:
            try:
                energy_result = self._modules["energy_management"].forward({
                    **sensor_data,
                    "dt": dt,
                })
                results["energy_management"] = energy_result
                self.watchdog_manager.feed("energy_management")
            except Exception as e:
                self.failsafe.report_failure("energy_management", str(e))
        
        # Anomaly detection
        if "anomaly_detection" in self._modules:
            try:
                anomaly_result = self._modules["anomaly_detection"].forward(sensor_data)
                results["anomaly_detection"] = anomaly_result
                self.watchdog_manager.feed("anomaly_detection")
                
                if anomaly_result.get("is_anomaly"):
                    self.message_bus.publish_sync(
                        MessageType.SAFETY,
                        "mission_controller",
                        {"anomaly": anomaly_result},
                    )
            except Exception as e:
                self.failsafe.report_failure("anomaly_detection", str(e))
        
        # Phase-specific behavior
        if self.phase == MissionPhase.SEARCH:
            commands.update(self._search_behavior(sensor_data, results))
        elif self.phase == MissionPhase.APPROACH:
            commands.update(self._approach_behavior(sensor_data, results))
        elif self.phase == MissionPhase.CAPTURE:
            commands.update(self._capture_behavior(sensor_data, results))
        
        return {
            "commands": commands,
            "results": results,
            "state": self.get_state(sensor_data),
        }
    
    def _search_behavior(
        self,
        sensor_data: Dict[str, Any],
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Search for debris."""
        commands = {}
        
        # Object tracking
        if "object_tracking" in self._modules:
            try:
                track_result = self._modules["object_tracking"].forward(sensor_data)
                
                if track_result.get("num_tracks", 0) > 0:
                    # Select target
                    tracks = track_result["tracks"]
                    # Select closest track
                    closest = min(tracks, key=lambda t: np.linalg.norm(t["position"]))
                    self.target_debris_id = closest["id"]
                    self.phase = MissionPhase.APPROACH
                    logger.info(f"Target acquired: {self.target_debris_id}")
            except Exception as e:
                self.failsafe.report_failure("object_tracking", str(e))
        
        return commands
    
    def _approach_behavior(
        self,
        sensor_data: Dict[str, Any],
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Approach target debris."""
        commands = {}
        
        if self.target_debris_id is None:
            self.phase = MissionPhase.SEARCH
            return commands
        
        # Get target position from tracking
        if "object_tracking" in self._modules:
            track_result = self._modules["object_tracking"].forward(sensor_data)
            
            target = None
            for track in track_result.get("tracks", []):
                if track["id"] == self.target_debris_id:
                    target = track
                    break
            
            if target is None:
                # Lost target
                self.target_debris_id = None
                self.phase = MissionPhase.SEARCH
                return commands
            
            # Compute approach trajectory
            target_pos = np.array(target["position"])
            current_pos = results.get("navigation", {}).get("position", np.zeros(3))
            
            distance = np.linalg.norm(target_pos - current_pos)
            
            if distance < 0.05:  # 50m - close enough for capture
                self.phase = MissionPhase.CAPTURE
                logger.info(f"Ready for capture at distance: {distance*1000:.1f}m")
            else:
                # Simple proportional control to target
                direction = (target_pos - current_pos) / (distance + 1e-6)
                thrust_mag = min(distance * 10, self.config.spacecraft.max_thrust)
                commands["thrust"] = direction * thrust_mag
        
        return commands
    
    def _capture_behavior(
        self,
        sensor_data: Dict[str, Any],
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Capture debris."""
        commands = {}
        
        # Use manipulator
        if "manipulator_control" in self._modules:
            try:
                manip_result = self._modules["manipulator_control"].forward(sensor_data)
                commands["joint_velocities"] = manip_result.get("joint_velocities", np.zeros(6))
                commands["gripper"] = manip_result.get("gripper_command", 0.0)
                
                # Check if captured
                if sensor_data.get("gripper_contact", False):
                    self.captured_count += 1
                    self.target_debris_id = None
                    self.phase = MissionPhase.STORE
                    logger.info(f"Debris captured! Total: {self.captured_count}")
            except Exception as e:
                self.failsafe.report_failure("manipulator_control", str(e))
        
        return commands
    
    def _emergency_response(
        self,
        sensor_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Emergency response - safe mode."""
        logger.critical("Emergency response active")
        
        return {
            "commands": {
                "thrust": np.zeros(3),
                "torque": np.zeros(3),
                "gripper": 0.0,
                "mode": "emergency",
            },
            "results": {},
            "state": self.get_state(sensor_data),
        }
    
    def get_state(
        self,
        sensor_data: Optional[Dict[str, Any]] = None,
    ) -> MissionState:
        """Get current mission state."""
        sensor_data = sensor_data or {}
        
        return MissionState(
            phase=self.phase,
            mode=self.mode,
            target_debris_id=self.target_debris_id,
            captured_count=self.captured_count,
            fuel_remaining=sensor_data.get("fuel_mass", 0.0),
            battery_soc=sensor_data.get("battery_soc", 1.0),
            position=np.asarray(sensor_data.get("position", [0, 0, 0])),
            velocity=np.asarray(sensor_data.get("velocity", [0, 0, 0])),
            time_elapsed=self.time_elapsed,
            alerts=[],
        )
    
    def set_mode(self, mode: MissionMode):
        """Set mission mode."""
        self.mode = mode
        logger.info(f"Mission mode set to: {mode.value}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get mission metrics."""
        return {
            "captured_count": self.captured_count,
            "avoided_collisions": self.avoided_collisions,
            "time_elapsed": self.time_elapsed,
            "current_phase": self.phase.value,
            "failsafe_mode": self.failsafe.mode.value,
            "module_health": self.failsafe.module_health.copy(),
        }
