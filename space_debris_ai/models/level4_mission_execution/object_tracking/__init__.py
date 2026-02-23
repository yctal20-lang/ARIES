"""
Multi-Object Tracking Module.

Components:
- MultiObjectTracker: DETR-like architecture for tracking
- TrackingModule: Complete tracking system
"""

from .tracker import MultiObjectTracker, TrackingModule, TrackedObject

__all__ = [
    "MultiObjectTracker",
    "TrackingModule",
    "TrackedObject",
]
