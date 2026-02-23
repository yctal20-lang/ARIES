"""
Failure Prediction Module.

Components:
- TemporalFusionTransformer: TFT for remaining useful life prediction
- FailurePredictionModule: Complete failure prediction system
"""

from .tft import TemporalFusionTransformer, FailurePredictionModule

__all__ = [
    "TemporalFusionTransformer",
    "FailurePredictionModule",
]
