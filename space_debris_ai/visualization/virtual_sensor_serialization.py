"""JSON serialization for VirtualSensorHub.read_all() payloads (dashboard / API)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def serialize_virtual_sensor_data(sdata: dict) -> Dict[str, Any]:
    """Convert virtual sensor readings to a JSON-friendly dict."""
    out: Dict[str, Any] = {}
    gps = sdata.get("gps")
    if gps is not None:
        pos = gps.position
        vel = gps.velocity
        out["gps"] = {
            "position": pos.tolist() if hasattr(pos, "tolist") and not np.any(np.isnan(pos)) else None,
            "velocity": vel.tolist() if hasattr(vel, "tolist") and not np.any(np.isnan(vel)) else None,
            "hdop": gps.hdop,
            "num_satellites": gps.num_satellites,
            "fix_valid": bool(gps.fix_valid),
        }
    star = sdata.get("star_tracker")
    if star is not None:
        out["star_tracker"] = {
            "attitude": star.attitude.tolist(),
            "confidence": star.confidence,
            "num_stars": star.num_stars,
            "blinded": bool(star.blinded),
        }
    radar = sdata.get("radar")
    if radar is not None:
        out["radar"] = {
            "num_targets": radar.num_targets,
            "features_vector": radar.features_vector.tolist(),
        }
    power = sdata.get("power")
    if power is not None:
        out["power"] = {
            "battery_soc": power.battery_soc,
            "solar_power": power.solar_power,
            "bus_voltage": power.bus_voltage,
            "eclipse_state": power.eclipse_state,
            "battery_temperature": power.battery_temperature,
        }
    mag = sdata.get("magnetometer")
    if mag is not None:
        out["magnetometer"] = {
            "magnetic_field": mag.magnetic_field.tolist(),
            "field_magnitude": mag.field_magnitude,
        }
    sun = sdata.get("sun_sensor")
    if sun is not None:
        out["sun_sensor"] = {
            "sun_vector": sun.sun_vector.tolist(),
            "sun_visible": bool(sun.sun_visible),
            "intensity": sun.intensity,
        }
    prox = sdata.get("proximity")
    if prox is not None:
        out["proximity"] = {
            "range_to_target": prox.range_to_target if prox.target_detected else None,
            "closing_velocity": prox.closing_velocity,
            "target_detected": bool(prox.target_detected),
        }
    therm = sdata.get("thermal")
    if therm is not None:
        out["thermal"] = {
            "temperatures": therm.temperatures,
            "mean_temperature": therm.mean_temperature,
            "max_temperature": therm.max_temperature,
        }
    doppler = sdata.get("doppler")
    if doppler is not None:
        out["doppler"] = {
            "radial_velocity": doppler.radial_velocity,
            "closing_rate": doppler.closing_rate,
            "target_detected": bool(doppler.target_detected),
        }
    rad = sdata.get("radiation")
    if rad is not None:
        out["radiation"] = {
            "dose_rate": rad.dose_rate,
            "particle_flux": rad.particle_flux,
            "solar_event": bool(rad.solar_event),
            "cumulative_dose": rad.cumulative_dose,
            "in_saa": bool(rad.in_saa),
        }
    spec = sdata.get("spectrometer")
    if spec is not None:
        out["spectrometer"] = {
            "material_class": spec.material_class,
            "material_confidence": spec.material_confidence,
            "albedo": spec.albedo,
            "target_detected": bool(spec.target_detected),
        }
    accel = sdata.get("accelerometer")
    if accel is not None:
        out["accelerometer"] = {
            "acceleration": accel.acceleration.tolist(),
            "vibration_rms": accel.vibration_rms,
            "shock_detected": bool(accel.shock_detected),
        }
    return out
