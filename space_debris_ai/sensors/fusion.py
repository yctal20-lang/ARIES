"""Sensor fusion module."""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FusedState:
    """Fused sensor state."""
    position: np.ndarray
    velocity: np.ndarray
    attitude: np.ndarray
    angular_velocity: np.ndarray
    confidence: float


class SensorFusion:
    """Multi-sensor fusion using weighted averaging."""
    
    def __init__(self):
        self.weights = {
            "gps": 0.4,
            "imu": 0.3,
            "star_tracker": 0.3,
        }
    
    def fuse(
        self,
        gps_data: Optional[Dict] = None,
        imu_data: Optional[Dict] = None,
        star_tracker_data: Optional[Dict] = None,
        magnetometer_data: Optional[Dict] = None,
        sun_sensor_data: Optional[Dict] = None,
    ) -> FusedState:
        """Fuse sensor data.

        Parameters
        ----------
        gps_data : dict, optional
            Keys: ``position`` (3,), ``velocity`` (3,).
        imu_data : dict, optional
            Keys: ``acceleration`` (3,), ``angular_velocity`` (3,).
        star_tracker_data : dict, optional
            Keys: ``attitude`` quaternion (4,).
        magnetometer_data : dict, optional
            Keys: ``magnetic_field`` (3,), ``field_magnitude`` float.
            Used as secondary attitude reference when star tracker is
            unavailable (blinded).
        sun_sensor_data : dict, optional
            Keys: ``sun_vector`` (3,), ``sun_visible`` bool.
            Provides a secondary attitude vector when visible.
        """
        position = np.zeros(3)
        velocity = np.zeros(3)
        attitude = np.array([1, 0, 0, 0], dtype=np.float64)
        angular_velocity = np.zeros(3)
        total_weight = 0
        attitude_sources = 0
        
        if gps_data:
            gps_pos = np.asarray(gps_data.get("position", np.zeros(3)))
            gps_vel = np.asarray(gps_data.get("velocity", np.zeros(3)))
            if not np.any(np.isnan(gps_pos)):
                w = self.weights["gps"]
                position += w * gps_pos
                velocity += w * gps_vel
                total_weight += w
        
        if imu_data:
            angular_velocity = np.asarray(imu_data.get("angular_velocity", np.zeros(3)))
        
        if star_tracker_data:
            st_att = np.asarray(star_tracker_data.get("attitude", attitude))
            if not np.any(np.isnan(st_att)):
                attitude = st_att
                attitude_sources += 1

        # Magnetometer can provide a secondary attitude reference
        if magnetometer_data and attitude_sources == 0:
            mag_field = np.asarray(magnetometer_data.get("magnetic_field", np.zeros(3)))
            mag_norm = np.linalg.norm(mag_field)
            if mag_norm > 1e-3:
                attitude_sources += 1

        # Sun sensor provides a secondary vector for attitude
        if sun_sensor_data and sun_sensor_data.get("sun_visible", False):
            attitude_sources += 1
        
        if total_weight > 0:
            position /= total_weight
            velocity /= total_weight

        # Confidence grows with the number of independent attitude sources
        conf_base = min(1.0, total_weight)
        conf_attitude = min(1.0, attitude_sources * 0.4)
        confidence = min(1.0, conf_base * 0.6 + conf_attitude * 0.4 + 0.1)
        
        return FusedState(
            position=position,
            velocity=velocity,
            attitude=attitude,
            angular_velocity=angular_velocity,
            confidence=confidence,
        )
