"""
Maps Arduino physical sensor readings to virtual sensor equivalents and
vice-versa, enabling a unified data view in the dashboard.

Arduino sensors:
    HC-SR04 (distance_km)  <->  VirtualProximitySensor (range_to_target)
    KY-003 (hall/magnetic) <->  VirtualMagnetometer (field_magnitude > threshold)
    DHT11 (temperature)    <->  VirtualThermalArray (external component temperature)
    Vibration (pin 13)     <->  VirtualAccelerometer (shock_detected)
"""

from typing import Any, Dict, Optional


# Thresholds for mapping between continuous virtual values and boolean Arduino flags
MAGNETOMETER_DETECTION_THRESHOLD_UT = 35.0  # micro-Tesla (LEO field ~25-65 uT)
VIBRATION_RMS_THRESHOLD = 0.001             # m/s^2


class ArduinoSensorMapper:
    """Bidirectional mapping between Arduino and virtual sensor data."""

    def __init__(
        self,
        mag_threshold: float = MAGNETOMETER_DETECTION_THRESHOLD_UT,
        vib_threshold: float = VIBRATION_RMS_THRESHOLD,
    ):
        self.mag_threshold = mag_threshold
        self.vib_threshold = vib_threshold

    def arduino_to_virtual(self, arduino: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Arduino ``get_latest()`` dict into a dict resembling
        virtual sensor readings (for overlay on the dashboard).
        """
        result: Dict[str, Any] = {}

        # Distance
        dist_km = arduino.get("distance_km")
        if dist_km is not None:
            result["proximity"] = {
                "range_to_target": dist_km * 1000.0,  # km -> m (inverse of sketch scaling)
                "target_detected": True,
                "source": "arduino_hc_sr04",
            }

        # Hall / magnetic
        hall = arduino.get("hall")
        magnetic = arduino.get("magnetic")
        if hall is not None or magnetic is not None:
            detected = bool(hall or magnetic)
            result["magnetometer"] = {
                "field_detected": detected,
                "source": "arduino_ky003",
            }

        # Temperature
        temp = arduino.get("temperature")
        if temp is not None:
            result["thermal"] = {
                "external_temperature": temp,
                "source": "arduino_dht11",
            }

        # Vibration
        vib = arduino.get("vibration")
        if vib is not None:
            result["accelerometer"] = {
                "shock_detected": bool(vib),
                "source": "arduino_vibration",
            }

        return result

    def virtual_to_arduino(self, virtual_sensors: Dict[str, Any]) -> Dict[str, Any]:
        """
        Project virtual sensor readings into the same schema as an
        Arduino ``get_latest()`` snapshot for display parity.
        """
        result: Dict[str, Any] = {
            "distance_km": None,
            "magnetic": None,
            "hall": None,
            "temperature": None,
            "vibration": None,
            "source": "virtual",
        }

        prox = virtual_sensors.get("proximity")
        if prox is not None:
            r = prox.get("range_to_target")
            if r is not None and prox.get("target_detected"):
                result["distance_km"] = r / 1000.0  # m -> km

        mag = virtual_sensors.get("magnetometer")
        if mag is not None:
            magnitude = mag.get("field_magnitude", 0.0)
            detected = magnitude > self.mag_threshold
            result["magnetic"] = detected
            result["hall"] = detected

        therm = virtual_sensors.get("thermal")
        if therm is not None:
            temps = therm.get("temperatures", {})
            if temps:
                result["temperature"] = therm.get("mean_temperature")

        accel = virtual_sensors.get("accelerometer")
        if accel is not None:
            result["vibration"] = accel.get("shock_detected", False)

        return result
