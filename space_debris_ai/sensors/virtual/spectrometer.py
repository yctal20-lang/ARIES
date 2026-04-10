"""Virtual spectrometer for debris material classification."""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


MATERIAL_CLASSES = [
    "aluminum",
    "titanium",
    "steel",
    "carbon_composite",
    "mli",          # multi-layer insulation (Kapton / Mylar)
    "solar_cell",
    "glass",
    "painted_surface",
    "copper",
    "unknown",
]

# Mapping from simulation debris_type to likely material distribution
_TYPE_TO_MATERIALS = {
    "fragment":     {"aluminum": 0.4, "steel": 0.25, "titanium": 0.15, "copper": 0.1, "unknown": 0.1},
    "rocket_body":  {"aluminum": 0.6, "steel": 0.2, "titanium": 0.1, "painted_surface": 0.1},
    "satellite":    {"aluminum": 0.3, "solar_cell": 0.2, "mli": 0.15, "carbon_composite": 0.15,
                     "titanium": 0.1, "glass": 0.1},
    "tool":         {"steel": 0.5, "titanium": 0.3, "aluminum": 0.2},
    "panel":        {"solar_cell": 0.4, "mli": 0.3, "glass": 0.2, "aluminum": 0.1},
    "unknown":      {"unknown": 0.5, "aluminum": 0.3, "steel": 0.2},
}

SPECTRUM_DIM = 32  # number of spectral bins


@dataclass
class SpectrometerReading:
    """Spectrometer reading."""
    spectrum: np.ndarray             # Reflectance spectrum (SPECTRUM_DIM,)
    material_class: str              # Most likely material
    material_confidence: float       # 0-1
    albedo: float                    # Overall reflectance
    target_detected: bool
    timestamp: float


class VirtualSpectrometer:
    """
    Virtual spectrometer for classifying debris material.

    Generates a synthetic reflectance spectrum based on the known
    ``debris_type`` from the simulation and a probabilistic material model.
    The spectrum is only available when a target is close enough and
    illuminated by the Sun (not eclipsed).
    """

    def __init__(
        self,
        max_range: float = 0.5,      # km (500 m)
        noise_std: float = 0.05,
        update_rate: float = 1.0,     # Hz
        seed: Optional[int] = None,
    ):
        self.max_range = max_range
        self.noise_std = noise_std
        self.update_rate = update_rate
        self._rng = np.random.RandomState(seed)
        self._last_reading: Optional[SpectrometerReading] = None
        self._last_update_time: float = -1e9

        # Pre-generate canonical spectra for each material (deterministic)
        gen = np.random.RandomState(12345)
        self._canonical_spectra = {}
        for mat in MATERIAL_CLASSES:
            base = gen.rand(SPECTRUM_DIM) * 0.5 + 0.25
            # Each material gets a distinct spectral "fingerprint"
            peak = gen.randint(0, SPECTRUM_DIM)
            base[max(0, peak - 2):peak + 3] += 0.3
            self._canonical_spectra[mat] = np.clip(base, 0, 1)

    def read(
        self,
        spacecraft_position: np.ndarray,
        debris_positions: List[np.ndarray],
        debris_types: List[str],
        sun_visible: bool = True,
        timestamp: float = 0.0,
    ) -> SpectrometerReading:
        dt_since = timestamp - self._last_update_time
        if dt_since < 1.0 / self.update_rate and self._last_reading is not None:
            return self._last_reading

        best_dist = float("inf")
        best_type = "unknown"

        for pos, dtype in zip(debris_positions, debris_types):
            dist = np.linalg.norm(np.asarray(pos) - spacecraft_position)
            if dist < best_dist and dist < self.max_range:
                best_dist = dist
                best_type = dtype

        detected = bool(best_dist <= self.max_range and sun_visible)

        if detected:
            material = self._sample_material(best_type)
            spectrum = self._canonical_spectra.get(material, self._canonical_spectra["unknown"]).copy()
            spectrum += self._rng.randn(SPECTRUM_DIM) * self.noise_std
            spectrum = np.clip(spectrum, 0, 1)
            albedo = float(np.mean(spectrum))
            confidence = max(0.3, 1.0 - best_dist / self.max_range)
            confidence *= (1.0 - self.noise_std)
        else:
            material = "unknown"
            spectrum = np.zeros(SPECTRUM_DIM)
            albedo = 0.0
            confidence = 0.0

        reading = SpectrometerReading(
            spectrum=spectrum,
            material_class=material,
            material_confidence=confidence,
            albedo=albedo,
            target_detected=detected,
            timestamp=timestamp,
        )
        self._last_reading = reading
        self._last_update_time = timestamp
        return reading

    def _sample_material(self, debris_type: str) -> str:
        dist = _TYPE_TO_MATERIALS.get(debris_type, _TYPE_TO_MATERIALS["unknown"])
        materials = list(dist.keys())
        probs = np.array(list(dist.values()))
        probs /= probs.sum()
        return self._rng.choice(materials, p=probs)
