"""Arduino <-> A.R.I.E.S bridge.

Reads HC-SR04 (distance), KY-003 (Hall/magnetic), DHT11 (temp+humidity)
from Arduino Serial and exposes data via Flask blueprint.
"""

from space_debris_ai.arduino_bridge.routes import arduino_bp

__all__ = ["arduino_bp"]
