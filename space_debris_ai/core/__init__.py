"""
Core module for Space Debris Collector AI System.
Contains base classes, configuration, and messaging infrastructure.
"""

from .config import Config, SystemConfig, SimulationConfig, TrainingConfig
from .base_module import BaseModule, ModuleState
from .message_bus import MessageBus, Message, MessageType

__all__ = [
    "Config",
    "SystemConfig", 
    "SimulationConfig",
    "TrainingConfig",
    "BaseModule",
    "ModuleState",
    "MessageBus",
    "Message",
    "MessageType",
]
