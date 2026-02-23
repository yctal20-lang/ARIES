"""
Space Debris Collector AI System

An autonomous AI system for space debris collection using deep learning,
reinforcement learning, and advanced sensor fusion techniques.

Main components:
- core: Base classes, configuration, and messaging
- models: Neural network modules for various tasks
- sensors: Sensor interfaces and fusion
- simulation: Orbital mechanics simulation environment
- safety: Fail-safe mechanisms and watchdogs
- training: Training scripts and utilities
- inference: Real-time inference engine
"""

__version__ = "0.1.0"
__author__ = "Space Debris AI Team"

# Lazy load core (avoids loading torch when only using simulation/sensors)
__all__ = [
    "__version__",
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


def __getattr__(name):
    if name in (
        "Config", "SystemConfig", "SimulationConfig", "TrainingConfig",
        "BaseModule", "ModuleState", "MessageBus", "Message", "MessageType",
    ):
        from .core import (
            Config,
            SystemConfig,
            SimulationConfig,
            TrainingConfig,
            BaseModule,
            ModuleState,
            MessageBus,
            Message,
            MessageType,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
