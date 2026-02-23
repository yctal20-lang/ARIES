"""
Temporal Convolutional Network for state prediction with physics constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

from ....core.base_module import BaseModule


class CausalConv1d(nn.Module):
    """Causal 1D convolution (no future information leakage)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class TCNBlock(nn.Module):
    """Temporal Convolutional Network residual block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, seq_len]
        residual = x
        
        out = self.conv1(x)
        out = out.transpose(1, 2)  # [batch, seq_len, channels]
        out = self.norm1(out)
        out = out.transpose(1, 2)  # [batch, channels, seq_len]
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = F.relu(out)
        out = self.dropout(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        return F.relu(out + residual)


class TCNPredictor(nn.Module):
    """
    Temporal Convolutional Network for spacecraft state prediction.
    
    Predicts future states (position, velocity, attitude) from history.
    """
    
    def __init__(
        self,
        input_dim: int = 13,        # pos(3) + vel(3) + att(4) + accel(3)
        output_dim: int = 10,       # pos(3) + vel(3) + att(4)
        hidden_channels: List[int] = [64, 128, 256],
        kernel_size: int = 3,
        prediction_horizon: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prediction_horizon = prediction_horizon
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_channels[0])
        
        # TCN blocks with exponentially increasing dilation
        self.tcn_blocks = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(zip(
            hidden_channels[:-1], hidden_channels[1:]
        )):
            dilation = 2 ** i
            self.tcn_blocks.append(
                TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
        
        # Add initial block
        self.tcn_blocks.insert(0, TCNBlock(
            hidden_channels[0], hidden_channels[0], kernel_size, 1, dropout
        ))
        
        # Output projection for multi-step prediction
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels[-1], output_dim * prediction_horizon),
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels[-1] // 2, output_dim * prediction_horizon),
            nn.Softplus(),  # Ensure positive uncertainty
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input sequence [batch, seq_len, input_dim]
            
        Returns:
            predictions: [batch, prediction_horizon, output_dim]
            uncertainties: [batch, prediction_horizon, output_dim]
        """
        batch_size = x.shape[0]
        
        # Project input
        x = self.input_proj(x)  # [batch, seq_len, hidden_channels[0]]
        x = x.transpose(1, 2)    # [batch, channels, seq_len]
        
        # TCN blocks
        for block in self.tcn_blocks:
            x = block(x)
        
        # Take last timestep features
        features = x[:, :, -1]  # [batch, hidden_channels[-1]]
        
        # Generate predictions
        pred_flat = self.output_proj(features)
        predictions = pred_flat.view(batch_size, self.prediction_horizon, self.output_dim)
        
        # Generate uncertainties
        unc_flat = self.uncertainty_head(features)
        uncertainties = unc_flat.view(batch_size, self.prediction_horizon, self.output_dim)
        
        return predictions, uncertainties
    
    def predict_sequence(
        self,
        history: np.ndarray,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict future state sequence.
        
        Args:
            history: Historical states [seq_len, input_dim]
            device: Compute device
            
        Returns:
            predictions, uncertainties
        """
        self.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(history).unsqueeze(0).to(device)
            predictions, uncertainties = self.forward(x)
        
        return predictions.cpu().numpy().squeeze(0), uncertainties.cpu().numpy().squeeze(0)


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss function for orbital mechanics.
    
    Penalizes predictions that violate physical laws.
    """
    
    def __init__(
        self,
        mu: float = 398600.4418,  # Earth gravitational parameter
        physics_weight: float = 0.1,
        smoothness_weight: float = 0.01,
    ):
        super().__init__()
        self.mu = mu
        self.physics_weight = physics_weight
        self.smoothness_weight = smoothness_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        dt: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss.
        
        Args:
            predictions: Predicted states [batch, horizon, state_dim]
            targets: True states [batch, horizon, state_dim]
            dt: Time step between predictions
            
        Returns:
            Dictionary of loss components
        """
        # MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Physics loss: velocity should be consistent with position change
        pred_pos = predictions[:, :, :3]  # Position
        pred_vel = predictions[:, :, 3:6]  # Velocity
        
        # Numerical derivative of position
        pos_diff = (pred_pos[:, 1:, :] - pred_pos[:, :-1, :]) / dt
        
        # Should match velocity
        physics_loss = F.mse_loss(pos_diff, pred_vel[:, :-1, :])
        
        # Gravitational constraint: a ≈ -μr/|r|³
        r_mag = torch.norm(pred_pos, dim=-1, keepdim=True)
        expected_accel_mag = self.mu / (r_mag ** 2 + 1e-6)
        
        # Velocity change should be consistent with gravity
        vel_diff = (pred_vel[:, 1:, :] - pred_vel[:, :-1, :]) / dt
        actual_accel_mag = torch.norm(vel_diff, dim=-1, keepdim=True)
        
        gravity_loss = F.mse_loss(
            actual_accel_mag,
            expected_accel_mag[:, :-1, :],
        )
        
        # Smoothness loss (penalize jerky predictions)
        accel_diff = vel_diff[:, 1:, :] - vel_diff[:, :-1, :]
        smoothness_loss = torch.mean(accel_diff ** 2)
        
        # Quaternion normalization loss
        if predictions.shape[-1] >= 10:
            pred_quat = predictions[:, :, 6:10]
            quat_norm = torch.norm(pred_quat, dim=-1)
            quat_loss = F.mse_loss(quat_norm, torch.ones_like(quat_norm))
        else:
            quat_loss = torch.tensor(0.0, device=predictions.device)
        
        # Total loss
        total_loss = (
            mse_loss +
            self.physics_weight * (physics_loss + gravity_loss + quat_loss) +
            self.smoothness_weight * smoothness_loss
        )
        
        return {
            "total": total_loss,
            "mse": mse_loss,
            "physics": physics_loss,
            "gravity": gravity_loss,
            "quaternion": quat_loss,
            "smoothness": smoothness_loss,
        }


class StatePredictionModule(BaseModule):
    """Complete state prediction module."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        super().__init__(
            name="state_prediction",
            config=config,
            device=device,
        )
        
        self.predictor = None
        self.history_buffer = deque(maxlen=config.get("history_length", 100))
        self.prediction_horizon = config.get("prediction_horizon", 50)
    
    def _build_model(self) -> nn.Module:
        """Build the TCN predictor."""
        self.predictor = TCNPredictor(
            input_dim=self.config.get("input_dim", 13),
            output_dim=self.config.get("output_dim", 10),
            hidden_channels=self.config.get("hidden_channels", [64, 128, 256]),
            prediction_horizon=self.prediction_horizon,
        ).to(self.device)
        
        return self.predictor
    
    def _preprocess(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Build history sequence from inputs."""
        # Construct state vector
        state = []
        for key in ["position", "velocity", "attitude", "acceleration"]:
            if key in inputs:
                state.extend(np.asarray(inputs[key]).flatten().tolist())
        
        state = np.array(state, dtype=np.float32)
        self.history_buffer.append(state)
        
        # Pad if needed
        if len(self.history_buffer) < self.history_buffer.maxlen:
            padding = [np.zeros_like(state)] * (self.history_buffer.maxlen - len(self.history_buffer))
            history = np.array(padding + list(self.history_buffer))
        else:
            history = np.array(list(self.history_buffer))
        
        return torch.FloatTensor(history).unsqueeze(0).to(self.device)
    
    def _postprocess(self, outputs: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        """Process predictions."""
        predictions, uncertainties = outputs
        pred_np = predictions.cpu().numpy().squeeze(0)
        unc_np = uncertainties.cpu().numpy().squeeze(0)
        
        return {
            "predicted_trajectory": pred_np,
            "predicted_position": pred_np[:, :3],
            "predicted_velocity": pred_np[:, 3:6],
            "predicted_attitude": pred_np[:, 6:10] if pred_np.shape[1] >= 10 else None,
            "uncertainties": unc_np,
            "horizon_steps": self.prediction_horizon,
        }
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future states."""
        history = self._preprocess(inputs)
        
        with torch.no_grad():
            predictions, uncertainties = self.predictor(history)
        
        return self._postprocess((predictions, uncertainties))
    
    def _fallback(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Simple linear extrapolation fallback."""
        position = np.asarray(inputs.get("position", np.zeros(3)))
        velocity = np.asarray(inputs.get("velocity", np.zeros(3)))
        
        dt = inputs.get("dt", 0.1)
        
        # Linear extrapolation
        predictions = []
        for t in range(self.prediction_horizon):
            future_pos = position + velocity * (t + 1) * dt
            predictions.append(np.concatenate([future_pos, velocity]))
        
        return {
            "predicted_trajectory": np.array(predictions),
            "predicted_position": np.array([p[:3] for p in predictions]),
            "fallback": True,
        }
