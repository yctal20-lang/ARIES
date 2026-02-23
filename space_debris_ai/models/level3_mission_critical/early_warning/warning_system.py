"""
Attention-based early warning system for predicting critical events.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List
from enum import IntEnum
from dataclasses import dataclass, field
from collections import deque
import time

from ....core.base_module import BaseModule


class AlertLevel(IntEnum):
    """Alert severity levels."""
    INFO = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


@dataclass
class Alert:
    """Alert data structure."""
    level: AlertLevel
    category: str
    message: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    predicted_time: Optional[float] = None  # Time until event
    recommended_action: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for sequence processing."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply self-attention.
        
        Returns:
            output, attention_weights
        """
        attn_out, attn_weights = self.attention(x, x, x, attn_mask=mask)
        return self.norm(x + attn_out), attn_weights


class AttentionWarningSystem(nn.Module):
    """
    Attention-based neural network for early warning.
    
    Analyzes telemetry sequences to detect precursors of critical events.
    """
    
    def __init__(
        self,
        input_dim: int = 24,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        num_categories: int = 6,  # Types of warnings
        seq_len: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_categories = num_categories
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.1)
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Feature aggregation
        self.feature_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )
        
        # Alert level classifier (multi-label)
        self.level_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 4),  # 4 alert levels
        )
        
        # Category classifier (multi-label)
        self.category_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_categories),
        )
        
        # Time-to-event regressor
        self.time_regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
            nn.Softplus(),  # Positive time
        )
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # Warning categories
        self.categories = [
            "collision_risk",
            "system_failure",
            "power_critical",
            "attitude_instability",
            "debris_approach",
            "communication_loss",
        ]
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input sequence [batch, seq_len, input_dim]
            
        Returns:
            Dictionary with predictions
        """
        batch_size = x.shape[0]
        
        # Embed input
        x = self.input_embed(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.shape[1], :]
        
        # Apply attention layers
        attention_maps = []
        for layer in self.attention_layers:
            x, attn = layer(x)
            attention_maps.append(attn)
        
        # Global average pooling
        features = x.mean(dim=1)  # [batch, embed_dim]
        features = self.feature_pool(features)
        
        # Predictions
        level_logits = self.level_classifier(features)
        category_logits = self.category_classifier(features)
        time_to_event = self.time_regressor(features)
        confidence = self.confidence_head(features)
        
        return {
            "level_logits": level_logits,
            "level_probs": F.softmax(level_logits, dim=-1),
            "category_logits": category_logits,
            "category_probs": torch.sigmoid(category_logits),
            "time_to_event": time_to_event.squeeze(-1),
            "confidence": confidence.squeeze(-1),
            "attention_maps": attention_maps,
        }
    
    def predict(
        self,
        sequence: np.ndarray,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Make prediction from numpy input."""
        self.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            outputs = self.forward(x)
        
        level_probs = outputs["level_probs"].cpu().numpy().squeeze()
        category_probs = outputs["category_probs"].cpu().numpy().squeeze()
        
        predicted_level = AlertLevel(np.argmax(level_probs))
        active_categories = [
            self.categories[i] for i, p in enumerate(category_probs) if p > 0.5
        ]
        
        return {
            "level": predicted_level,
            "level_probabilities": {
                AlertLevel(i).name: float(level_probs[i])
                for i in range(len(level_probs))
            },
            "active_categories": active_categories,
            "category_probabilities": {
                cat: float(category_probs[i])
                for i, cat in enumerate(self.categories)
            },
            "time_to_event": outputs["time_to_event"].item(),
            "confidence": outputs["confidence"].item(),
        }


class AlertManager:
    """
    Manages alert generation, deduplication, and escalation.
    """
    
    def __init__(
        self,
        cooldown_time: float = 5.0,
        escalation_threshold: int = 3,
    ):
        """
        Initialize alert manager.
        
        Args:
            cooldown_time: Minimum time between similar alerts
            escalation_threshold: Number of repeated alerts before escalation
        """
        self.cooldown_time = cooldown_time
        self.escalation_threshold = escalation_threshold
        
        self.active_alerts: List[Alert] = []
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_counts: Dict[str, int] = {}
        self.last_alert_time: Dict[str, float] = {}
    
    def process_prediction(
        self,
        prediction: Dict[str, Any],
    ) -> Optional[Alert]:
        """
        Process prediction and generate alert if needed.
        
        Args:
            prediction: Output from AttentionWarningSystem.predict()
            
        Returns:
            Alert if generated, None otherwise
        """
        level = prediction["level"]
        confidence = prediction["confidence"]
        
        # Only alert for WARNING or higher
        if level < AlertLevel.WARNING:
            return None
        
        # Check confidence threshold
        if confidence < 0.5:
            return None
        
        # Determine category
        if prediction["active_categories"]:
            category = prediction["active_categories"][0]
        else:
            category = "unknown"
        
        # Check cooldown
        alert_key = f"{level.name}_{category}"
        current_time = time.time()
        
        if alert_key in self.last_alert_time:
            if current_time - self.last_alert_time[alert_key] < self.cooldown_time:
                return None
        
        # Create alert
        alert = Alert(
            level=level,
            category=category,
            message=self._generate_message(level, category, prediction),
            confidence=confidence,
            predicted_time=prediction["time_to_event"],
            recommended_action=self._get_recommended_action(category),
            data=prediction,
        )
        
        # Track alert
        self.last_alert_time[alert_key] = current_time
        self.alert_counts[alert_key] = self.alert_counts.get(alert_key, 0) + 1
        
        # Escalate if repeated
        if self.alert_counts[alert_key] >= self.escalation_threshold:
            if alert.level < AlertLevel.EMERGENCY:
                alert.level = AlertLevel(alert.level + 1)
                alert.message = f"[ESCALATED] {alert.message}"
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        return alert
    
    def _generate_message(
        self,
        level: AlertLevel,
        category: str,
        prediction: Dict[str, Any],
    ) -> str:
        """Generate human-readable alert message."""
        time_str = f"{prediction['time_to_event']:.1f}s" if prediction['time_to_event'] < 60 else f"{prediction['time_to_event']/60:.1f}min"
        
        messages = {
            "collision_risk": f"Potential collision detected. Time to event: {time_str}",
            "system_failure": f"System failure predicted. Time to event: {time_str}",
            "power_critical": f"Power system critical. Time to event: {time_str}",
            "attitude_instability": f"Attitude instability detected. Time to event: {time_str}",
            "debris_approach": f"Debris approach detected. Time to event: {time_str}",
            "communication_loss": f"Communication loss predicted. Time to event: {time_str}",
        }
        
        return messages.get(category, f"Warning: {category}. Time: {time_str}")
    
    def _get_recommended_action(self, category: str) -> str:
        """Get recommended action for category."""
        actions = {
            "collision_risk": "Initiate avoidance maneuver",
            "system_failure": "Switch to backup systems",
            "power_critical": "Enter power saving mode",
            "attitude_instability": "Activate attitude hold",
            "debris_approach": "Monitor and prepare avoidance",
            "communication_loss": "Store data locally, retry connection",
        }
        return actions.get(category, "Monitor situation")
    
    def clear_alerts(self, max_age: float = 300.0) -> None:
        """Clear old alerts."""
        current_time = time.time()
        self.active_alerts = [
            a for a in self.active_alerts
            if current_time - a.timestamp < max_age
        ]
    
    def get_highest_alert(self) -> Optional[Alert]:
        """Get the highest severity active alert."""
        if not self.active_alerts:
            return None
        return max(self.active_alerts, key=lambda a: a.level)


from typing import Tuple


class EarlyWarningModule(BaseModule):
    """Complete early warning module."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        super().__init__(
            name="early_warning",
            config=config,
            device=device,
        )
        
        self.warning_system = None
        self.alert_manager = AlertManager()
        self.history_buffer = deque(maxlen=config.get("seq_len", 100))
    
    def _build_model(self) -> nn.Module:
        self.warning_system = AttentionWarningSystem(
            input_dim=self.config.get("input_dim", 24),
            embed_dim=self.config.get("embed_dim", 128),
            num_heads=self.config.get("num_heads", 8),
            seq_len=self.config.get("seq_len", 100),
        ).to(self.device)
        
        return self.warning_system
    
    def _preprocess(self, inputs: Dict[str, Any]) -> torch.Tensor:
        # Build telemetry vector
        telemetry = []
        for key in ["position", "velocity", "attitude", "angular_velocity",
                    "acceleration", "battery_soc", "temperature"]:
            if key in inputs:
                val = np.asarray(inputs[key]).flatten()
                telemetry.extend(val.tolist())
        
        # Pad to expected dimension
        while len(telemetry) < self.config.get("input_dim", 24):
            telemetry.append(0.0)
        telemetry = np.array(telemetry[:self.config.get("input_dim", 24)], dtype=np.float32)
        
        self.history_buffer.append(telemetry)
        
        # Build sequence
        if len(self.history_buffer) < self.history_buffer.maxlen:
            padding = [np.zeros_like(telemetry)] * (self.history_buffer.maxlen - len(self.history_buffer))
            sequence = np.array(padding + list(self.history_buffer))
        else:
            sequence = np.array(list(self.history_buffer))
        
        return sequence
    
    def _postprocess(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        return outputs
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        sequence = self._preprocess(inputs)
        prediction = self.warning_system.predict(sequence, self.device)
        
        alert = self.alert_manager.process_prediction(prediction)
        
        result = prediction.copy()
        result["alert"] = alert
        result["active_alerts"] = len(self.alert_manager.active_alerts)
        
        return result
    
    def _fallback(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "level": AlertLevel.INFO,
            "active_categories": [],
            "confidence": 0.0,
            "fallback": True,
        }
