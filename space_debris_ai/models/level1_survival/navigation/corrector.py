"""
Neural Network Corrector for EKF.
Learns to correct systematic errors in the EKF estimates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional


class NNCorrector(nn.Module):
    """
    Neural network for correcting EKF systematic errors.
    
    Takes EKF state estimate and raw measurements as input,
    outputs corrections to the state estimate.
    
    Architecture: Residual MLP with attention to measurement reliability.
    """
    
    def __init__(
        self,
        state_dim: int = 16,
        measurement_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize corrector network.
        
        Args:
            state_dim: Dimension of state vector
            measurement_dim: Dimension of measurement vector
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # Input embedding
        input_dim = state_dim + measurement_dim
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Measurement attention (weigh measurement reliability)
        self.measurement_attention = nn.Sequential(
            nn.Linear(measurement_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, measurement_dim),
            nn.Sigmoid(),
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output heads for different state components
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
        )
        
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
        )
        
        self.attitude_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 4),
        )
        
        self.bias_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 6),  # gyro (3) + accel (3) biases
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # Initialize output layers with small weights
        self._init_output_weights()
    
    def _make_residual_block(
        self,
        dim: int,
        dropout: float,
    ) -> nn.Module:
        """Create a residual block."""
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
    
    def _init_output_weights(self):
        """Initialize output layer weights to be small."""
        for head in [self.position_head, self.velocity_head, 
                     self.attitude_head, self.bias_head]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
    
    def forward(
        self,
        state: torch.Tensor,
        measurements: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: EKF state estimate [batch, state_dim]
            measurements: Raw measurements [batch, measurement_dim]
            
        Returns:
            Dictionary of state corrections and confidence
        """
        # Apply attention to measurements
        meas_weights = self.measurement_attention(measurements)
        weighted_meas = measurements * meas_weights
        
        # Concatenate state and weighted measurements
        x = torch.cat([state, weighted_meas], dim=-1)
        
        # Input embedding
        x = self.input_embed(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = F.relu(x + block(x))
        
        # Generate corrections
        pos_correction = self.position_head(x)
        vel_correction = self.velocity_head(x)
        att_correction = self.attitude_head(x)
        bias_correction = self.bias_head(x)
        confidence = self.confidence_head(x)
        
        return {
            "position_correction": pos_correction,
            "velocity_correction": vel_correction,
            "attitude_correction": att_correction,
            "gyro_bias_correction": bias_correction[:, :3],
            "accel_bias_correction": bias_correction[:, 3:],
            "confidence": confidence.squeeze(-1),
            "measurement_weights": meas_weights,
        }
    
    def correct_state(
        self,
        state: np.ndarray,
        measurements: np.ndarray,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, float]:
        """
        Apply correction to state estimate.
        
        Args:
            state: EKF state estimate [16]
            measurements: Raw measurements [measurement_dim]
            device: Compute device
            
        Returns:
            (corrected_state, confidence)
        """
        self.eval()
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            meas_t = torch.FloatTensor(measurements).unsqueeze(0).to(device)
            
            corrections = self.forward(state_t, meas_t)
        
        # Apply corrections
        corrected = state.copy()
        corrected[0:3] += corrections["position_correction"].cpu().numpy().squeeze()
        corrected[3:6] += corrections["velocity_correction"].cpu().numpy().squeeze()
        
        # Attitude correction (additive in tangent space, then normalize)
        att_corr = corrections["attitude_correction"].cpu().numpy().squeeze()
        corrected[6:10] += att_corr * 0.1  # Scale down attitude corrections
        corrected[6:10] /= np.linalg.norm(corrected[6:10])
        
        # Bias corrections
        corrected[10:13] += corrections["gyro_bias_correction"].cpu().numpy().squeeze()
        corrected[13:16] += corrections["accel_bias_correction"].cpu().numpy().squeeze()
        
        confidence = corrections["confidence"].item()
        
        return corrected, confidence


class CorrectorLoss(nn.Module):
    """
    Loss function for training the NN corrector.
    """
    
    def __init__(
        self,
        position_weight: float = 10.0,
        velocity_weight: float = 5.0,
        attitude_weight: float = 2.0,
        bias_weight: float = 1.0,
        confidence_weight: float = 0.5,
    ):
        super().__init__()
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.attitude_weight = attitude_weight
        self.bias_weight = bias_weight
        self.confidence_weight = confidence_weight
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            predictions: Model corrections
            targets: Ground truth corrections (state_true - state_ekf)
            
        Returns:
            Dictionary of loss components
        """
        # Position correction loss
        pos_loss = F.mse_loss(
            predictions["position_correction"],
            targets["position_correction"],
        )
        
        # Velocity correction loss
        vel_loss = F.mse_loss(
            predictions["velocity_correction"],
            targets["velocity_correction"],
        )
        
        # Attitude correction loss
        att_loss = F.mse_loss(
            predictions["attitude_correction"],
            targets["attitude_correction"],
        )
        
        # Bias correction loss
        gyro_loss = F.mse_loss(
            predictions["gyro_bias_correction"],
            targets["gyro_bias_correction"],
        )
        accel_loss = F.mse_loss(
            predictions["accel_bias_correction"],
            targets["accel_bias_correction"],
        )
        bias_loss = gyro_loss + accel_loss
        
        # Confidence calibration loss
        # Confidence should correlate with actual correction accuracy
        correction_error = (
            (predictions["position_correction"] - targets["position_correction"]).pow(2).mean(dim=-1) +
            (predictions["velocity_correction"] - targets["velocity_correction"]).pow(2).mean(dim=-1)
        )
        # Low error -> high confidence
        target_confidence = torch.exp(-correction_error * 10)
        confidence_loss = F.mse_loss(predictions["confidence"], target_confidence)
        
        # Total loss
        total_loss = (
            self.position_weight * pos_loss +
            self.velocity_weight * vel_loss +
            self.attitude_weight * att_loss +
            self.bias_weight * bias_loss +
            self.confidence_weight * confidence_loss
        )
        
        return {
            "total": total_loss,
            "position": pos_loss,
            "velocity": vel_loss,
            "attitude": att_loss,
            "bias": bias_loss,
            "confidence": confidence_loss,
        }


class SequentialCorrector(nn.Module):
    """
    Sequential corrector using LSTM to process measurement history.
    Useful when measurements have temporal correlations.
    """
    
    def __init__(
        self,
        state_dim: int = 16,
        measurement_dim: int = 16,
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # LSTM for measurement sequence
        self.measurement_lstm = nn.LSTM(
            input_size=measurement_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.1,
        )
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Output (full state correction)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, state_dim),
        )
        
        # Initialize output with small weights
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
    
    def forward(
        self,
        state: torch.Tensor,
        measurement_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: Current EKF state [batch, state_dim]
            measurement_sequence: Measurement history [batch, seq_len, measurement_dim]
            
        Returns:
            State correction vector [batch, state_dim]
        """
        # Process measurement sequence
        lstm_out, _ = self.measurement_lstm(measurement_sequence)
        meas_features = lstm_out[:, -1, :]  # Take last output
        
        # Encode state
        state_features = self.state_encoder(state)
        
        # Fuse
        fused = torch.cat([state_features, meas_features], dim=-1)
        fused = self.fusion(fused)
        
        # Generate correction
        correction = self.output(fused)
        
        return correction
