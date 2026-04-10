"""Virtual high-precision accelerometer (micro-g) and vibration sensor."""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class AccelerometerReading:
    """Accelerometer / vibration sensor reading."""
    acceleration: np.ndarray     # Body-frame residual acceleration (m/s^2)
    vibration_rms: float         # RMS vibration level (m/s^2)
    shock_detected: bool         # True when sudden impulse above threshold
    timestamp: float


class VirtualAccelerometer:
    """
    Virtual high-precision accelerometer with vibration / shock detection.

    In micro-gravity the dominant accelerations are:
    - Atmospheric drag residual
    - Thruster firings
    - Reaction wheel / CMG jitter
    - Docking / capture shocks

    This sensor also serves as the virtual counterpart of the Arduino
    vibration sensor (pin 13) via the ``shock_detected`` flag.
    """

    def __init__(
        self,
        noise_std: float = 1e-5,         # m/s^2 (10 micro-g)
        vibration_base_rms: float = 5e-5, # m/s^2
        shock_threshold: float = 0.05,    # m/s^2
        update_rate: float = 100.0,       # Hz
        seed: Optional[int] = None,
    ):
        self.noise_std = noise_std
        self.vibration_base_rms = vibration_base_rms
        self.shock_threshold = shock_threshold
        self.update_rate = update_rate
        self._rng = np.random.RandomState(seed)
        self._last_reading: Optional[AccelerometerReading] = None
        self._last_update_time: float = -1e9

    def read(
        self,
        true_acceleration_body: np.ndarray,
        thrust_active: bool = False,
        timestamp: float = 0.0,
    ) -> AccelerometerReading:
        dt_since = timestamp - self._last_update_time
        if dt_since < 1.0 / self.update_rate and self._last_reading is not None:
            return self._last_reading

        # Base micro-vibration environment
        vib_rms = self.vibration_base_rms
        if thrust_active:
            vib_rms += 0.01  # thrusters add significant vibration

        vibration = self._rng.randn(3) * vib_rms
        accel = true_acceleration_body + vibration + self._rng.randn(3) * self.noise_std

        measured_rms = float(np.sqrt(np.mean(vibration ** 2)))
        shock = float(np.max(np.abs(accel))) > self.shock_threshold

        reading = AccelerometerReading(
            acceleration=accel,
            vibration_rms=measured_rms,
            shock_detected=shock,
            timestamp=timestamp,
        )
        self._last_reading = reading
        self._last_update_time = timestamp
        return reading
