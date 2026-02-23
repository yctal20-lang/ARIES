"""
Level 3: Mission-Critical Systems
Системы критичные для выполнения миссии.

Приоритет: СРЕДНИЙ-ВЫСОКИЙ
Время отклика: < 1s
Требования: 99.9% надёжность

Модули:
- state_prediction: Прогнозирование состояния (TCN + Physics-Informed)
- early_warning: Система раннего предупреждения (Attention)
- sensor_filter: Фильтрация и очистка сенсорных данных (Denoising AE)
- failure_prediction: Прогнозирование отказов (Temporal Fusion Transformer)
"""

from .state_prediction import TCNPredictor, PhysicsInformedLoss, StatePredictionModule
from .early_warning import AttentionWarningSystem, AlertManager, AlertLevel, Alert
from .sensor_filter import DenoisingAutoencoder, AdaptiveFilter, SensorFilterModule
from .failure_prediction import TemporalFusionTransformer, FailurePredictionModule

__all__ = [
    # State Prediction
    "TCNPredictor",
    "PhysicsInformedLoss",
    "StatePredictionModule",
    # Early Warning
    "AttentionWarningSystem",
    "AlertManager",
    "AlertLevel",
    "Alert",
    # Sensor Filter
    "DenoisingAutoencoder",
    "AdaptiveFilter",
    "SensorFilterModule",
    # Failure Prediction
    "TemporalFusionTransformer",
    "FailurePredictionModule",
]
