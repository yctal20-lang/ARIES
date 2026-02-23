"""
Anomaly type classifier for categorizing detected anomalies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass


class AnomalyType(Enum):
    """Types of anomalies."""
    NORMAL = "normal"
    SUDDEN_ACCELERATION = "sudden_acceleration"
    RAPID_ROTATION = "rapid_rotation"
    SENSOR_DRIFT = "sensor_drift"
    SENSOR_NOISE = "sensor_noise"
    IMPACT = "impact"
    THRUSTER_MALFUNCTION = "thruster_malfunction"
    POWER_FLUCTUATION = "power_fluctuation"
    COMMUNICATION_LOSS = "communication_loss"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Anomaly classification result."""
    anomaly_type: AnomalyType
    confidence: float
    probabilities: Dict[str, float]
    features_importance: Dict[str, float]


class AnomalyClassifier(nn.Module):
    """
    Neural network for classifying anomaly types.
    
    Takes latent representation from autoencoder and feature errors
    to classify the type of anomaly.
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        feature_dim: int = 12,
        hidden_dim: int = 64,
        num_classes: int = 10,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Latent feature processor
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        
        # Error feature processor
        self.error_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )
        
        # Feature importance head
        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, feature_dim),
            nn.Sigmoid(),
        )
        
        self.anomaly_types = list(AnomalyType)
    
    def forward(
        self,
        latent: torch.Tensor,
        feature_errors: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            latent: Latent representation from autoencoder [batch, latent_dim]
            feature_errors: Per-feature reconstruction errors [batch, feature_dim]
            
        Returns:
            Dictionary with logits, probabilities, and feature importance
        """
        latent_features = self.latent_encoder(latent)
        error_features = self.error_encoder(feature_errors)
        
        combined = torch.cat([latent_features, error_features], dim=-1)
        
        logits = self.classifier(combined)
        probs = F.softmax(logits, dim=-1)
        importance = self.importance_head(combined)
        
        return {
            "logits": logits,
            "probabilities": probs,
            "feature_importance": importance,
        }
    
    def classify(
        self,
        latent: np.ndarray,
        feature_errors: np.ndarray,
        device: str = "cpu",
    ) -> ClassificationResult:
        """
        Classify an anomaly.
        
        Args:
            latent: Latent vector
            feature_errors: Feature errors
            device: Compute device
            
        Returns:
            ClassificationResult
        """
        self.eval()
        
        with torch.no_grad():
            latent_t = torch.FloatTensor(latent).unsqueeze(0).to(device)
            errors_t = torch.FloatTensor(feature_errors).unsqueeze(0).to(device)
            
            outputs = self.forward(latent_t, errors_t)
        
        probs = outputs["probabilities"].cpu().numpy().squeeze()
        importance = outputs["feature_importance"].cpu().numpy().squeeze()
        
        class_idx = np.argmax(probs)
        anomaly_type = self.anomaly_types[class_idx]
        
        prob_dict = {
            self.anomaly_types[i].value: float(probs[i])
            for i in range(len(self.anomaly_types))
        }
        
        importance_dict = {
            f"feature_{i}": float(importance[i])
            for i in range(len(importance))
        }
        
        return ClassificationResult(
            anomaly_type=anomaly_type,
            confidence=float(probs[class_idx]),
            probabilities=prob_dict,
            features_importance=importance_dict,
        )


class RuleBasedClassifier:
    """
    Rule-based anomaly classifier as fallback.
    Uses domain knowledge to classify anomalies without neural network.
    """
    
    def __init__(
        self,
        accel_threshold: float = 5.0,
        rotation_threshold: float = 10.0,
        drift_threshold: float = 0.1,
    ):
        self.accel_threshold = accel_threshold
        self.rotation_threshold = rotation_threshold
        self.drift_threshold = drift_threshold
    
    def classify(
        self,
        feature_errors: np.ndarray,
        raw_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> AnomalyType:
        """
        Classify anomaly using rules.
        
        Args:
            feature_errors: Per-feature reconstruction errors
            raw_data: Optional raw telemetry data
            
        Returns:
            Classified anomaly type
        """
        if raw_data is None:
            # Use feature error patterns
            max_error_idx = np.argmax(feature_errors)
            
            # Heuristic: first 3 features are acceleration
            if max_error_idx < 3:
                return AnomalyType.SUDDEN_ACCELERATION
            # Next 3 are angular velocity
            elif max_error_idx < 6:
                return AnomalyType.RAPID_ROTATION
            else:
                return AnomalyType.UNKNOWN
        
        # Use raw data for more accurate classification
        if "acceleration" in raw_data:
            accel = np.asarray(raw_data["acceleration"])
            accel_mag = np.linalg.norm(accel)
            
            if accel_mag > self.accel_threshold:
                # Check if it's impact-like (sudden spike)
                return AnomalyType.IMPACT
        
        if "angular_velocity" in raw_data:
            omega = np.asarray(raw_data["angular_velocity"])
            omega_mag = np.linalg.norm(omega) * 180 / np.pi  # Convert to deg/s
            
            if omega_mag > self.rotation_threshold:
                return AnomalyType.RAPID_ROTATION
        
        # Check for sensor drift (gradual increase in errors)
        error_gradient = np.gradient(feature_errors)
        if np.all(error_gradient > self.drift_threshold):
            return AnomalyType.SENSOR_DRIFT
        
        # Check for noise (high variance in errors)
        if np.std(feature_errors) > np.mean(feature_errors):
            return AnomalyType.SENSOR_NOISE
        
        return AnomalyType.UNKNOWN
