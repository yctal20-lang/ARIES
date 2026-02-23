"""
Denoising Autoencoder for sensor data filtering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque

from ....core.base_module import BaseModule


class DenoisingAutoencoder(nn.Module):
    """
    Denoising autoencoder for sensor data.
    Trained on pairs of (noisy, clean) data to learn noise removal.
    """
    
    def __init__(
        self,
        input_dim: int = 12,
        hidden_dims: list = [64, 32, 16],
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (mirror of encoder)
        decoder_layers = []
        for h_dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Noise level estimator
        self.noise_estimator = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, input_dim),
            nn.Softplus(),
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to output."""
        return self.decoder(z)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Noisy input [batch, input_dim]
            
        Returns:
            denoised: Cleaned signal
            noise_estimate: Estimated noise level per feature
        """
        z = self.encode(x)
        denoised = self.decode(z)
        noise_estimate = self.noise_estimator(z)
        return denoised, noise_estimate
    
    def filter(
        self,
        noisy_data: np.ndarray,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter noisy data.
        
        Args:
            noisy_data: Noisy sensor data [input_dim]
            device: Compute device
            
        Returns:
            (filtered_data, noise_estimate)
        """
        self.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(noisy_data).unsqueeze(0).to(device)
            denoised, noise = self.forward(x)
        
        return denoised.cpu().numpy().squeeze(), noise.cpu().numpy().squeeze()


class AdaptiveFilter(nn.Module):
    """
    Context-aware adaptive filter.
    Adjusts filtering strength based on operating mode and conditions.
    """
    
    def __init__(
        self,
        input_dim: int = 12,
        context_dim: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        
        # Signal processor
        self.signal_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Filter weights generator
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # Weights between 0 and 1
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
    
    def forward(
        self,
        signal: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply adaptive filtering.
        
        Args:
            signal: Input signal [batch, input_dim]
            context: Operating context [batch, context_dim]
            
        Returns:
            filtered_signal, filter_weights
        """
        # Encode
        ctx_features = self.context_encoder(context)
        sig_features = self.signal_encoder(signal)
        
        # Fuse
        combined = torch.cat([sig_features, ctx_features], dim=-1)
        fused = self.fusion(combined)
        
        # Generate adaptive weights
        weights = self.weight_generator(fused)
        
        # Generate filtered signal
        filtered = self.output_proj(fused)
        
        # Blend based on weights: high weight = more filtering
        output = weights * filtered + (1 - weights) * signal
        
        return output, weights


class TemporalFilter(nn.Module):
    """
    Temporal filter using 1D convolutions.
    Filters noise while preserving temporal dynamics.
    """
    
    def __init__(
        self,
        input_dim: int = 12,
        kernel_sizes: list = [3, 5, 7],
        num_filters: int = 32,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Multi-scale convolutions
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, num_filters, ks, padding=ks//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(inplace=True),
            )
            for ks in kernel_sizes
        ])
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(num_filters * len(kernel_sizes), input_dim, 1),
            nn.BatchNorm1d(input_dim),
        )
        
        # Residual connection weight
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal filtering.
        
        Args:
            x: Input sequence [batch, seq_len, input_dim]
            
        Returns:
            Filtered sequence [batch, seq_len, input_dim]
        """
        # [batch, seq_len, input_dim] -> [batch, input_dim, seq_len]
        x_conv = x.transpose(1, 2)
        
        # Multi-scale processing
        branch_outputs = [branch(x_conv) for branch in self.conv_branches]
        
        # Concatenate branches
        combined = torch.cat(branch_outputs, dim=1)
        
        # Fuse to original dimension
        filtered = self.fusion(combined)
        
        # Residual connection
        filtered = filtered.transpose(1, 2)  # [batch, seq_len, input_dim]
        
        # Blend
        alpha = torch.sigmoid(self.alpha)
        output = alpha * filtered + (1 - alpha) * x
        
        return output


class SensorFilterModule(BaseModule):
    """
    Complete sensor filtering module.
    Combines denoising autoencoder with adaptive and temporal filtering.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        super().__init__(
            name="sensor_filter",
            config=config,
            device=device,
        )
        
        self.denoiser = None
        self.adaptive_filter = None
        self.temporal_filter = None
        
        self.input_dim = config.get("input_dim", 12)
        self.use_temporal = config.get("use_temporal", True)
        self.history_buffer = deque(maxlen=config.get("buffer_size", 20))
    
    def _build_model(self) -> nn.Module:
        """Build filtering models."""
        self.denoiser = DenoisingAutoencoder(
            input_dim=self.input_dim,
            hidden_dims=self.config.get("hidden_dims", [64, 32, 16]),
        ).to(self.device)
        
        self.adaptive_filter = AdaptiveFilter(
            input_dim=self.input_dim,
            context_dim=self.config.get("context_dim", 8),
        ).to(self.device)
        
        if self.use_temporal:
            self.temporal_filter = TemporalFilter(
                input_dim=self.input_dim,
            ).to(self.device)
        
        return self.denoiser
    
    def _preprocess(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare inputs."""
        # Get sensor data
        sensor_data = inputs.get("sensor_data", np.zeros(self.input_dim))
        sensor_data = np.asarray(sensor_data, dtype=np.float32)[:self.input_dim]
        
        # Pad if needed
        if len(sensor_data) < self.input_dim:
            sensor_data = np.pad(sensor_data, (0, self.input_dim - len(sensor_data)))
        
        # Get context
        context = inputs.get("context", np.zeros(8))
        context = np.asarray(context, dtype=np.float32)[:8]
        if len(context) < 8:
            context = np.pad(context, (0, 8 - len(context)))
        
        # Add to history
        self.history_buffer.append(sensor_data)
        
        return {
            "sensor": torch.FloatTensor(sensor_data).unsqueeze(0).to(self.device),
            "context": torch.FloatTensor(context).unsqueeze(0).to(self.device),
        }
    
    def _postprocess(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Convert outputs to numpy."""
        return {
            k: v.cpu().numpy().squeeze() if isinstance(v, torch.Tensor) else v
            for k, v in outputs.items()
        }
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply filtering pipeline.
        
        Args:
            inputs: Dictionary with sensor_data and optional context
            
        Returns:
            Filtered sensor data
        """
        processed = self._preprocess(inputs)
        
        with torch.no_grad():
            # Step 1: Denoise
            denoised, noise_estimate = self.denoiser(processed["sensor"])
            
            # Step 2: Adaptive filtering based on context
            adaptive_filtered, weights = self.adaptive_filter(
                denoised, processed["context"]
            )
            
            # Step 3: Temporal filtering if enough history
            if self.use_temporal and len(self.history_buffer) >= 5:
                sequence = torch.FloatTensor(
                    np.array(list(self.history_buffer))
                ).unsqueeze(0).to(self.device)
                temporal_filtered = self.temporal_filter(sequence)
                final_output = temporal_filtered[:, -1, :]
            else:
                final_output = adaptive_filtered
        
        return self._postprocess({
            "filtered_data": final_output,
            "denoised": denoised,
            "noise_estimate": noise_estimate,
            "filter_weights": weights,
        })
    
    def _fallback(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Simple exponential moving average fallback."""
        sensor_data = np.asarray(inputs.get("sensor_data", np.zeros(self.input_dim)))
        
        if len(self.history_buffer) > 0:
            # EMA with alpha=0.3
            prev = np.array(self.history_buffer[-1])
            filtered = 0.3 * sensor_data + 0.7 * prev
        else:
            filtered = sensor_data
        
        self.history_buffer.append(sensor_data)
        
        return {
            "filtered_data": filtered,
            "fallback": True,
        }
