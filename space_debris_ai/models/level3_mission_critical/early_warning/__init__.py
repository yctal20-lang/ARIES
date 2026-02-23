"""
Early Warning System Module.

Components:
- AttentionWarningSystem: Attention-based early warning classifier
- AlertManager: Alert generation and management
"""

from .warning_system import AttentionWarningSystem, AlertManager, AlertLevel, Alert

__all__ = [
    "AttentionWarningSystem",
    "AlertManager",
    "AlertLevel",
    "Alert",
]
