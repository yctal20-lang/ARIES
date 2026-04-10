"""
Training utilities for Space Debris Collector AI System.
"""


def __getattr__(name: str):
    """Lazy imports to avoid RuntimeWarning when running sub-modules with -m."""
    _mapping = {
        "train_collision_avoidance": ".train_collision_avoidance",
        "train_energy_management": ".train_energy_management",
        "train_anomaly_detection": ".train_anomaly_detection",
        "train_state_prediction": ".train_state_prediction",
        "train_manipulator_control": ".train_manipulator_control",
    }
    if name in _mapping:
        import importlib
        module = importlib.import_module(_mapping[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "train_collision_avoidance",
    "train_energy_management",
    "train_anomaly_detection",
    "train_state_prediction",
    "train_manipulator_control",
]
