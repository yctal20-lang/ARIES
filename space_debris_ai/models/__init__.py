"""
Neural Network Models for Space Debris Collector AI System.

Organized by priority levels:

Level 1 - Survival (Критичные для выживания):
    - collision_avoidance: CNN + DRL для уклонения от столкновений
    - navigation: EKF + NN коррекция для навигации

Level 2 - Safety (Безопасность):
    - anomaly_detection: LSTM Autoencoder для обнаружения аномалий
    - energy_management: PPO для управления энергией

Level 3 - Mission Critical (Критичные для миссии):
    - state_prediction: TCN для прогнозирования состояния
    - early_warning: Attention-based система предупреждений
    - sensor_filter: Denoising Autoencoder для фильтрации
    - failure_prediction: TFT для прогнозирования отказов

Level 4 - Mission Execution (Выполнение миссии):
    - debris_recognition: EfficientNet + Multimodal для распознавания
    - manipulator_control: SAC для управления манипулятором
    - object_tracking: DETR-like tracker
"""

# Level 1: Survival Critical
from .level1_survival import (
    CollisionDetector,
    CollisionAvoidanceAgent,
    CollisionAvoidanceModule,
    ExtendedKalmanFilter,
    NNCorrector,
    NavigationModule,
)

# Level 2: Safety
from .level2_safety import (
    LSTMAutoencoder,
    AnomalyDetector,
    AnomalyClassifier,
    AnomalyType,
    PowerModel,
    PowerState,
    EnergyMode,
    EnergyAgent,
    EnergyManagementModule,
)

# Level 3: Mission Critical
from .level3_mission_critical import (
    TCNPredictor,
    PhysicsInformedLoss,
    StatePredictionModule,
    AttentionWarningSystem,
    AlertManager,
    AlertLevel,
    Alert,
    DenoisingAutoencoder,
    AdaptiveFilter,
    SensorFilterModule,
    TemporalFusionTransformer,
    FailurePredictionModule,
)

# Level 4: Mission Execution
from .level4_mission_execution import (
    DebrisRecognizer,
    DebrisRecognitionModule,
    DebrisInfo,
    ManipulatorController,
    ManipulatorModule,
    MultiObjectTracker,
    TrackingModule,
    TrackedObject,
)

__all__ = [
    # Level 1
    "CollisionDetector",
    "CollisionAvoidanceAgent",
    "CollisionAvoidanceModule",
    "ExtendedKalmanFilter",
    "NNCorrector",
    "NavigationModule",
    # Level 2
    "LSTMAutoencoder",
    "AnomalyDetector",
    "AnomalyClassifier",
    "AnomalyType",
    "PowerModel",
    "PowerState",
    "EnergyMode",
    "EnergyAgent",
    "EnergyManagementModule",
    # Level 3
    "TCNPredictor",
    "PhysicsInformedLoss",
    "StatePredictionModule",
    "AttentionWarningSystem",
    "AlertManager",
    "AlertLevel",
    "Alert",
    "DenoisingAutoencoder",
    "AdaptiveFilter",
    "SensorFilterModule",
    "TemporalFusionTransformer",
    "FailurePredictionModule",
    # Level 4
    "DebrisRecognizer",
    "DebrisRecognitionModule",
    "DebrisInfo",
    "ManipulatorController",
    "ManipulatorModule",
    "MultiObjectTracker",
    "TrackingModule",
    "TrackedObject",
]
