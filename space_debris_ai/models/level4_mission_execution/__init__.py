"""
Level 4: Mission Execution Systems
Системы непосредственного выполнения миссии.

Приоритет: СРЕДНИЙ
Время отклика: < 2s
Требования: 99% надёжность

Модули:
- debris_recognition: Распознавание и классификация мусора (EfficientNet + Multimodal)
- risk_assessment: Оценка риска захвата объектов (MLP Scorer)
- manipulator_control: Управление роботизированным манипулятором (SAC)
- precision_maneuvering: Точное маневрирование (Neural MPC)
- object_tracking: Многообъектное отслеживание (DETR-like Tracker)
"""

from .debris_recognition import DebrisRecognizer, DebrisRecognitionModule, DebrisInfo
from .risk_assessment import RiskAssessor, RiskAssessmentModule, RiskAssessment, RiskLevel
from .manipulator_control import ManipulatorController, ManipulatorModule
from .precision_maneuvering import NeuralMPC, PrecisionManeuveringModule
from .object_tracking import MultiObjectTracker, TrackingModule, TrackedObject

__all__ = [
    # Debris Recognition
    "DebrisRecognizer",
    "DebrisRecognitionModule",
    "DebrisInfo",
    # Risk Assessment
    "RiskAssessor",
    "RiskAssessmentModule",
    "RiskAssessment",
    "RiskLevel",
    # Manipulator Control
    "ManipulatorController",
    "ManipulatorModule",
    # Precision Maneuvering
    "NeuralMPC",
    "PrecisionManeuveringModule",
    # Object Tracking
    "MultiObjectTracker",
    "TrackingModule",
    "TrackedObject",
]
