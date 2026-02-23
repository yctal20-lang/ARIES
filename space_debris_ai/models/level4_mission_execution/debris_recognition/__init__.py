"""
Debris Recognition Module.

Components:
- DebrisRecognizer: Multi-modal debris classification
- VisualEncoder: EfficientNet-based visual encoder
- RadarEncoder: 1D CNN for radar signatures
"""

from .recognizer import DebrisRecognizer, DebrisRecognitionModule, DebrisInfo

__all__ = [
    "DebrisRecognizer",
    "DebrisRecognitionModule",
    "DebrisInfo",
]
