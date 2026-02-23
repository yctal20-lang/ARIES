"""
Safety systems for Space Debris Collector AI System.

Modules:
- failsafe: Fail-safe mechanisms and fallback algorithms
- watchdog: Module health monitoring and timeouts
"""

from .failsafe import FailsafeController, FallbackMode
from .watchdog import Watchdog, WatchdogManager

__all__ = [
    "FailsafeController",
    "FallbackMode",
    "Watchdog",
    "WatchdogManager",
]
