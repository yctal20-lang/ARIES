"""
Temporal Fusion Transformer for failure prediction and RUL estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque

from ....core.base_module import BaseModule


@dataclass
class FailurePrediction:
    """Failure prediction result."""
    component: str
    failure_probability: float
    remaining_useful_life: float  # Hours
    confidence: float
    degradation_trend: str  # "stable", "degrading", "critical"
    recommended_action: str


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for TFT."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)
        
        hidden = F.elu(self.fc1(x))
        hidden = self.dropout(hidden)
        
        output = self.fc2(hidden)
        gate = torch.sigmoid(self.gate(hidden))
        
        gated_output = gate * output + (1 - gate) * residual
        return self.norm(gated_output)


class VariableSelectionNetwork(nn.Module):
    """Variable selection for TFT."""
    
    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.grns = nn.ModuleList([
            GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_vars)
        ])
        
        self.softmax_grn = GatedResidualNetwork(
            num_vars * hidden_dim, hidden_dim, num_vars, dropout
        )
    
    def forward(
        self,
        embeddings: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select important variables.
        
        Args:
            embeddings: List of variable embeddings [batch, hidden_dim]
            
        Returns:
            combined: Weighted combination
            weights: Variable importance weights
        """
        # Process each variable
        processed = [grn(emb) for grn, emb in zip(self.grns, embeddings)]
        
        # Concatenate for weight computation
        concat = torch.cat(embeddings, dim=-1)
        weights = F.softmax(self.softmax_grn(concat), dim=-1)
        
        # Weighted sum
        stacked = torch.stack(processed, dim=-1)  # [batch, hidden, num_vars]
        combined = (stacked * weights.unsqueeze(1)).sum(dim=-1)
        
        return combined, weights


class TemporalFusionTransformer(nn.Module):
    """
    Simplified Temporal Fusion Transformer for failure prediction.
    """
    
    def __init__(
        self,
        num_components: int = 10,
        input_dim: int = 16,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        seq_len: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_components = num_components
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # Input embedding per component
        self.input_embed = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_components)
        ])
        
        # Variable selection
        self.var_selection = VariableSelectionNetwork(
            num_components, hidden_dim, dropout
        )
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output heads
        self.failure_head = nn.Sequential(
            GatedResidualNetwork(hidden_dim * 2, hidden_dim, hidden_dim, dropout),
            nn.Linear(hidden_dim, num_components),
            nn.Sigmoid(),
        )
        
        self.rul_head = nn.Sequential(
            GatedResidualNetwork(hidden_dim * 2, hidden_dim, hidden_dim, dropout),
            nn.Linear(hidden_dim, num_components),
            nn.Softplus(),
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_components),
            nn.Sigmoid(),
        )
        
        # Component names
        self.components = [
            "main_thruster", "rcs_system", "solar_panel_1", "solar_panel_2",
            "battery_1", "battery_2", "lidar", "radar", "computer", "communication"
        ][:num_components]
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Component telemetry [batch, seq_len, num_components, input_dim]
            
        Returns:
            Predictions for each component
        """
        batch_size, seq_len, num_comp, input_dim = x.shape
        
        # Embed each component's data
        embeddings = []
        for i in range(num_comp):
            comp_data = x[:, :, i, :]  # [batch, seq_len, input_dim]
            # Average over time for variable selection
            comp_avg = comp_data.mean(dim=1)  # [batch, input_dim]
            emb = self.input_embed[i](comp_avg)  # [batch, hidden_dim]
            embeddings.append(emb)
        
        # Variable selection
        selected, var_weights = self.var_selection(embeddings)
        
        # Prepare sequence for LSTM (combine all components)
        # [batch, seq_len, num_comp * input_dim]
        combined_seq = x.reshape(batch_size, seq_len, -1)
        
        # Project to hidden dim
        proj_seq = nn.functional.linear(
            combined_seq,
            torch.eye(self.hidden_dim, combined_seq.shape[-1], device=x.device)[:, :combined_seq.shape[-1]]
        )
        
        # Use selected features to weight the sequence
        proj_seq = proj_seq * selected.unsqueeze(1)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(proj_seq)
        
        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take final features
        features = attn_out[:, -1, :]  # [batch, hidden_dim * 2]
        
        # Predictions
        failure_probs = self.failure_head(features)
        rul_estimates = self.rul_head(features)
        confidence = self.confidence_head(features)
        
        return {
            "failure_probability": failure_probs,
            "remaining_useful_life": rul_estimates,
            "confidence": confidence,
            "variable_importance": var_weights,
            "attention_weights": attn_weights,
        }
    
    def predict_components(
        self,
        telemetry: np.ndarray,
        device: str = "cpu",
    ) -> List[FailurePrediction]:
        """
        Predict failure for all components.
        
        Args:
            telemetry: Component telemetry data
            device: Compute device
            
        Returns:
            List of FailurePrediction for each component
        """
        self.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(telemetry).unsqueeze(0).to(device)
            outputs = self.forward(x)
        
        failure_probs = outputs["failure_probability"].cpu().numpy().squeeze()
        rul_estimates = outputs["remaining_useful_life"].cpu().numpy().squeeze()
        confidence = outputs["confidence"].cpu().numpy().squeeze()
        
        predictions = []
        for i, comp in enumerate(self.components):
            prob = failure_probs[i]
            rul = rul_estimates[i]
            conf = confidence[i]
            
            # Determine trend
            if prob < 0.2:
                trend = "stable"
                action = "Continue normal operation"
            elif prob < 0.5:
                trend = "degrading"
                action = "Schedule maintenance"
            else:
                trend = "critical"
                action = "Immediate attention required"
            
            predictions.append(FailurePrediction(
                component=comp,
                failure_probability=float(prob),
                remaining_useful_life=float(rul),
                confidence=float(conf),
                degradation_trend=trend,
                recommended_action=action,
            ))
        
        return predictions


class FailurePredictionModule(BaseModule):
    """Complete failure prediction module."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        super().__init__(
            name="failure_prediction",
            config=config,
            device=device,
        )
        
        self.tft = None
        self.num_components = config.get("num_components", 10)
        self.input_dim = config.get("input_dim", 16)
        self.seq_len = config.get("seq_len", 100)
        
        # History buffer per component
        self.history = {
            f"component_{i}": deque(maxlen=self.seq_len)
            for i in range(self.num_components)
        }
    
    def _build_model(self) -> nn.Module:
        self.tft = TemporalFusionTransformer(
            num_components=self.num_components,
            input_dim=self.input_dim,
            hidden_dim=self.config.get("hidden_dim", 64),
            seq_len=self.seq_len,
        ).to(self.device)
        
        return self.tft
    
    def _preprocess(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Build telemetry tensor from inputs."""
        # Update history for each component
        for i in range(self.num_components):
            key = f"component_{i}"
            data_key = inputs.get(key, inputs.get("telemetry", {}).get(key))
            
            if data_key is not None:
                self.history[key].append(np.asarray(data_key)[:self.input_dim])
            else:
                # Use zeros if no data
                self.history[key].append(np.zeros(self.input_dim))
        
        # Build tensor [seq_len, num_components, input_dim]
        telemetry = []
        for i in range(self.num_components):
            key = f"component_{i}"
            hist = list(self.history[key])
            
            # Pad if needed
            while len(hist) < self.seq_len:
                hist.insert(0, np.zeros(self.input_dim))
            
            telemetry.append(np.array(hist))
        
        # [seq_len, num_components, input_dim] -> [num_components, seq_len, input_dim]
        telemetry = np.array(telemetry).transpose(1, 0, 2)
        
        return torch.FloatTensor(telemetry).unsqueeze(0).to(self.device)
    
    def _postprocess(self, outputs: List[FailurePrediction]) -> Dict[str, Any]:
        """Convert predictions to output dict."""
        result = {
            "predictions": [],
            "highest_risk_component": None,
            "overall_health": "good",
        }
        
        max_prob = 0
        for pred in outputs:
            result["predictions"].append({
                "component": pred.component,
                "failure_probability": pred.failure_probability,
                "remaining_useful_life_hours": pred.remaining_useful_life,
                "confidence": pred.confidence,
                "trend": pred.degradation_trend,
                "action": pred.recommended_action,
            })
            
            if pred.failure_probability > max_prob:
                max_prob = pred.failure_probability
                result["highest_risk_component"] = pred.component
        
        # Overall health assessment
        if max_prob > 0.7:
            result["overall_health"] = "critical"
        elif max_prob > 0.4:
            result["overall_health"] = "warning"
        elif max_prob > 0.2:
            result["overall_health"] = "degraded"
        
        return result
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Predict component failures."""
        telemetry = self._preprocess(inputs)
        predictions = self.tft.predict_components(
            telemetry.squeeze(0).cpu().numpy(),
            self.device,
        )
        return self._postprocess(predictions)
    
    def _fallback(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Simple threshold-based fallback."""
        return {
            "predictions": [],
            "highest_risk_component": None,
            "overall_health": "unknown",
            "fallback": True,
        }
