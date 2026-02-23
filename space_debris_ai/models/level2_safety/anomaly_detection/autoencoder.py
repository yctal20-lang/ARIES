"""
LSTM Autoencoder for telemetry anomaly detection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from collections import deque

from ....core.base_module import BaseModule


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    is_anomaly: bool
    anomaly_score: float           # 0-1, higher = more anomalous
    reconstruction_error: float    # Raw reconstruction error
    threshold: float               # Current anomaly threshold
    confidence: float              # Detection confidence
    anomalous_features: List[int]  # Indices of anomalous features
    timestamp: float = 0.0


class LSTMEncoder(nn.Module):
    """LSTM encoder for sequence compression."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        
        # Project to latent space
        self.fc = nn.Linear(hidden_dim * 2, latent_dim)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode sequence.
        
        Args:
            x: Input sequence [batch, seq_len, input_dim]
            
        Returns:
            latent: Latent representation [batch, latent_dim]
            hidden: LSTM hidden states for decoder
        """
        # LSTM encoding
        output, (hidden, cell) = self.lstm(x)
        
        # Use last output from both directions
        last_forward = output[:, -1, :self.lstm.hidden_size]
        last_backward = output[:, 0, self.lstm.hidden_size:]
        combined = torch.cat([last_forward, last_backward], dim=-1)
        
        # Project to latent
        latent = self.fc(combined)
        
        return latent, (hidden, cell)


class LSTMDecoder(nn.Module):
    """LSTM decoder for sequence reconstruction."""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_len: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Project latent to initial hidden state
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.fc_cell = nn.Linear(latent_dim, hidden_dim * num_layers)
        
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.num_layers = num_layers
    
    def forward(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent to sequence.
        
        Args:
            latent: Latent representation [batch, latent_dim]
            
        Returns:
            Reconstructed sequence [batch, seq_len, output_dim]
        """
        batch_size = latent.shape[0]
        
        # Initialize hidden states from latent
        hidden = self.fc_hidden(latent).view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()
        
        cell = self.fc_cell(latent).view(batch_size, self.num_layers, self.hidden_dim)
        cell = cell.permute(1, 0, 2).contiguous()
        
        # Repeat latent as input for each timestep
        decoder_input = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Decode
        output, _ = self.lstm(decoder_input, (hidden, cell))
        
        # Project to output dimension
        reconstructed = self.output_layer(output)
        
        return reconstructed


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for anomaly detection in telemetry sequences.
    
    Trained on normal operation data to learn typical patterns.
    High reconstruction error indicates anomalous behavior.
    """
    
    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        seq_len: int = 100,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        self.decoder = LSTMDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout,
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
            reconstructed: Reconstructed sequence
            latent: Latent representation
        """
        latent, _ = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute reconstruction error.
        
        Args:
            x: Input sequence
            reduction: Error reduction method
            
        Returns:
            Reconstruction error per sample
        """
        reconstructed, _ = self.forward(x)
        
        # MSE per feature per timestep
        error = (x - reconstructed) ** 2
        
        if reduction == "mean":
            # Mean over features and timesteps
            return error.mean(dim=(1, 2))
        elif reduction == "sum":
            return error.sum(dim=(1, 2))
        elif reduction == "none":
            return error
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    def feature_errors(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction error per feature.
        
        Args:
            x: Input sequence [batch, seq_len, input_dim]
            
        Returns:
            Error per feature [batch, input_dim]
        """
        reconstructed, _ = self.forward(x)
        error = (x - reconstructed) ** 2
        return error.mean(dim=1)  # Mean over timesteps


class AnomalyDetector(BaseModule):
    """
    Complete anomaly detection system.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        super().__init__(
            name="anomaly_detection",
            config=config,
            device=device,
        )
        
        self.autoencoder = None
        self.input_dim = config.get("input_dim", 12)
        self.seq_len = config.get("seq_len", 100)
        
        # Anomaly detection parameters
        self.threshold = config.get("initial_threshold", 0.1)
        self.threshold_percentile = config.get("threshold_percentile", 95)
        
        # Adaptive threshold tracking
        self.error_history = deque(maxlen=config.get("history_size", 1000))
        self.adapt_threshold = config.get("adapt_threshold", True)
        
        # Input buffer for sequence building
        self.input_buffer = deque(maxlen=self.seq_len)
        
        # Running statistics for normalization
        self.running_mean = None
        self.running_std = None
        self.n_samples = 0
    
    def _build_model(self) -> nn.Module:
        """Build the autoencoder model."""
        self.autoencoder = LSTMAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.config.get("hidden_dim", 128),
            latent_dim=self.config.get("latent_dim", 32),
            seq_len=self.seq_len,
            num_layers=self.config.get("num_layers", 2),
        ).to(self.device)
        
        return self.autoencoder
    
    def _preprocess(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Preprocess telemetry data."""
        # Get telemetry vector
        if "telemetry" in inputs:
            data = np.asarray(inputs["telemetry"], dtype=np.float32)
        else:
            # Construct from individual components
            components = []
            for key in ["acceleration", "angular_velocity", "position", "velocity"]:
                if key in inputs:
                    components.append(np.asarray(inputs[key]).flatten())
            data = np.concatenate(components) if components else np.zeros(self.input_dim)
        
        # Normalize if statistics available
        if self.running_mean is not None:
            data = (data - self.running_mean) / (self.running_std + 1e-8)
        
        # Add to buffer
        self.input_buffer.append(data)
        
        # Build sequence
        if len(self.input_buffer) >= self.seq_len:
            sequence = np.array(list(self.input_buffer))
        else:
            # Pad with zeros if not enough data
            padding = np.zeros((self.seq_len - len(self.input_buffer), self.input_dim))
            sequence = np.vstack([padding, np.array(list(self.input_buffer))])
        
        return torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
    
    def _postprocess(self, outputs: Tuple[torch.Tensor, torch.Tensor]) -> AnomalyResult:
        """Process autoencoder outputs."""
        reconstructed, latent = outputs
        
        # Get original sequence from buffer
        if len(self.input_buffer) >= self.seq_len:
            original = torch.FloatTensor(
                np.array(list(self.input_buffer))
            ).unsqueeze(0).to(self.device)
        else:
            padding = np.zeros((self.seq_len - len(self.input_buffer), self.input_dim))
            sequence = np.vstack([padding, np.array(list(self.input_buffer))])
            original = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Compute errors
        with torch.no_grad():
            mse = ((original - reconstructed) ** 2).mean().item()
            feature_errors = ((original - reconstructed) ** 2).mean(dim=1).squeeze().cpu().numpy()
        
        # Update error history
        self.error_history.append(mse)
        
        # Adaptive threshold
        if self.adapt_threshold and len(self.error_history) > 100:
            self.threshold = np.percentile(
                list(self.error_history),
                self.threshold_percentile
            )
        
        # Determine if anomaly
        is_anomaly = mse > self.threshold
        
        # Compute anomaly score (0-1)
        if self.threshold > 0:
            anomaly_score = min(1.0, mse / (2 * self.threshold))
        else:
            anomaly_score = 0.0
        
        # Find anomalous features
        feature_threshold = np.mean(feature_errors) + 2 * np.std(feature_errors)
        anomalous_features = np.where(feature_errors > feature_threshold)[0].tolist()
        
        # Confidence based on how far from threshold
        if is_anomaly:
            confidence = min(1.0, (mse - self.threshold) / self.threshold)
        else:
            confidence = min(1.0, (self.threshold - mse) / self.threshold)
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            reconstruction_error=mse,
            threshold=self.threshold,
            confidence=confidence,
            anomalous_features=anomalous_features,
        )
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect anomalies in telemetry.
        
        Args:
            inputs: Telemetry data dictionary
            
        Returns:
            Anomaly detection results
        """
        sequence = self._preprocess(inputs)
        
        with torch.no_grad():
            reconstructed, latent = self.autoencoder(sequence)
        
        result = self._postprocess((reconstructed, latent))
        
        return {
            "is_anomaly": result.is_anomaly,
            "anomaly_score": result.anomaly_score,
            "reconstruction_error": result.reconstruction_error,
            "threshold": result.threshold,
            "confidence": result.confidence,
            "anomalous_features": result.anomalous_features,
        }
    
    def update_statistics(self, data: np.ndarray) -> None:
        """Update running mean and std for normalization."""
        if self.running_mean is None:
            self.running_mean = data.copy()
            self.running_std = np.abs(data) + 1e-8
            self.n_samples = 1
        else:
            self.n_samples += 1
            delta = data - self.running_mean
            self.running_mean += delta / self.n_samples
            delta2 = data - self.running_mean
            self.running_std = np.sqrt(
                (self.running_std ** 2 * (self.n_samples - 1) + delta * delta2) / self.n_samples
            )
    
    def _fallback(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback: simple threshold-based detection."""
        # Use simple statistical checks
        values = []
        for key in ["acceleration", "angular_velocity"]:
            if key in inputs:
                values.extend(np.asarray(inputs[key]).flatten().tolist())
        
        if not values:
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "fallback": True,
            }
        
        # Simple z-score check
        values = np.array(values)
        if self.running_mean is not None:
            z_scores = np.abs((values - self.running_mean[:len(values)]) / 
                             (self.running_std[:len(values)] + 1e-8))
            is_anomaly = np.any(z_scores > 3)
            anomaly_score = min(1.0, np.max(z_scores) / 5)
        else:
            is_anomaly = False
            anomaly_score = 0.0
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "fallback": True,
        }
