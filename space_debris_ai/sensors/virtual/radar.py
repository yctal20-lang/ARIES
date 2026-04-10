"""Virtual radar sensor for long-range debris detection."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RadarDetection:
    """Single radar detection."""
    range_km: float
    azimuth_rad: float
    elevation_rad: float
    radial_velocity: float     # km/s (positive = receding)
    rcs: float                 # Radar cross-section estimate (m^2)
    snr: float                 # Signal-to-noise ratio (dB)


@dataclass
class RadarReading:
    """Radar sensor reading."""
    detections: List[RadarDetection]
    features_vector: np.ndarray   # Fixed 64-dim feature vector for ML
    num_targets: int
    timestamp: float


class VirtualRadar:
    """
    Virtual radar sensor for long-range debris detection.

    Generates detections for objects within range, computes per-target
    observables (range, angles, radial velocity, RCS), and compresses
    the variable-length detection list into a fixed 64-dim feature vector
    suitable for the CollisionDetector neural network.
    """

    FEATURES_DIM = 64

    def __init__(
        self,
        max_range: float = 5.0,          # km
        range_noise_std: float = 0.005,   # km (~5 m)
        angle_noise_std: float = 0.0087,  # rad (~0.5 deg)
        velocity_noise_std: float = 0.0001,  # km/s
        false_alarm_rate: float = 0.02,
        update_rate: float = 1.0,         # Hz
        max_detections: int = 32,
        seed: Optional[int] = None,
    ):
        self.max_range = max_range
        self.range_noise_std = range_noise_std
        self.angle_noise_std = angle_noise_std
        self.velocity_noise_std = velocity_noise_std
        self.false_alarm_rate = false_alarm_rate
        self.update_rate = update_rate
        self.max_detections = max_detections
        self._rng = np.random.RandomState(seed)
        self._last_reading: Optional[RadarReading] = None
        self._last_update_time: float = -1e9

    def read(
        self,
        spacecraft_position: np.ndarray,
        spacecraft_velocity: np.ndarray,
        debris_positions: List[np.ndarray],
        debris_velocities: List[np.ndarray],
        debris_sizes: List[float],
        timestamp: float = 0.0,
    ) -> RadarReading:
        dt_since = timestamp - self._last_update_time
        if dt_since < 1.0 / self.update_rate and self._last_reading is not None:
            return self._last_reading

        detections: List[RadarDetection] = []

        for pos, vel, size in zip(debris_positions, debris_velocities, debris_sizes):
            rel_pos = np.asarray(pos) - spacecraft_position
            rel_vel = np.asarray(vel) - spacecraft_velocity
            dist = np.linalg.norm(rel_pos)

            if dist > self.max_range or dist < 1e-9:
                continue

            rcs = self._estimate_rcs(size)
            snr = self._compute_snr(dist, rcs)

            # Detection probability based on SNR
            p_detect = min(1.0, max(0.0, 1.0 - np.exp(-snr / 5.0)))
            if self._rng.rand() > p_detect:
                continue

            r_noisy = dist + self._rng.randn() * self.range_noise_std
            az = np.arctan2(rel_pos[1], rel_pos[0]) + self._rng.randn() * self.angle_noise_std
            el = np.arcsin(np.clip(rel_pos[2] / dist, -1, 1)) + self._rng.randn() * self.angle_noise_std
            radial_vel = np.dot(rel_vel, rel_pos / dist) + self._rng.randn() * self.velocity_noise_std

            detections.append(RadarDetection(
                range_km=max(0.0, r_noisy),
                azimuth_rad=az,
                elevation_rad=el,
                radial_velocity=radial_vel,
                rcs=rcs,
                snr=snr,
            ))

        # Add false alarms
        n_false = self._rng.poisson(self.false_alarm_rate * self.max_range)
        for _ in range(n_false):
            detections.append(RadarDetection(
                range_km=self._rng.uniform(0.1, self.max_range),
                azimuth_rad=self._rng.uniform(-np.pi, np.pi),
                elevation_rad=self._rng.uniform(-np.pi / 2, np.pi / 2),
                radial_velocity=self._rng.randn() * 0.01,
                rcs=self._rng.exponential(0.01),
                snr=self._rng.uniform(0, 5),
            ))

        detections.sort(key=lambda d: d.range_km)
        detections = detections[:self.max_detections]

        features = self._build_features(detections)

        reading = RadarReading(
            detections=detections,
            features_vector=features,
            num_targets=len(detections),
            timestamp=timestamp,
        )
        self._last_reading = reading
        self._last_update_time = timestamp
        return reading

    def _build_features(self, detections: List[RadarDetection]) -> np.ndarray:
        """Compress detections into a fixed 64-dim feature vector."""
        features = np.zeros(self.FEATURES_DIM)

        if not detections:
            return features

        # First 8 slots: aggregate statistics
        ranges = np.array([d.range_km for d in detections])
        rv = np.array([d.radial_velocity for d in detections])
        snrs = np.array([d.snr for d in detections])

        features[0] = len(detections)
        features[1] = np.min(ranges)
        features[2] = np.mean(ranges)
        features[3] = np.std(ranges) if len(ranges) > 1 else 0
        features[4] = np.min(rv) if len(rv) else 0
        features[5] = np.max(rv) if len(rv) else 0
        features[6] = np.mean(snrs)
        features[7] = np.sum([d.rcs for d in detections])

        # Next 56 slots: top-8 detections, 7 features each
        for i, det in enumerate(detections[:8]):
            base = 8 + i * 7
            features[base + 0] = det.range_km / self.max_range
            features[base + 1] = det.azimuth_rad / np.pi
            features[base + 2] = det.elevation_rad / (np.pi / 2)
            features[base + 3] = det.radial_velocity
            features[base + 4] = np.log1p(det.rcs)
            features[base + 5] = det.snr / 30.0
            features[base + 6] = 1.0  # detection-present flag

        return features

    @staticmethod
    def _estimate_rcs(size_m: float) -> float:
        """Rough RCS from characteristic size (sphere approximation)."""
        return np.pi * (size_m / 2.0) ** 2

    @staticmethod
    def _compute_snr(distance_km: float, rcs_m2: float) -> float:
        """Simplified radar equation SNR (arbitrary scale)."""
        if distance_km < 1e-6:
            return 50.0
        return 10.0 * np.log10(rcs_m2 / (distance_km ** 4) + 1e-30) + 40.0
