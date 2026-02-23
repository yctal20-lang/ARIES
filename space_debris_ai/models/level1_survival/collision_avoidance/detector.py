"""
Collision Detection Neural Network.
Uses CNN to process sensor data and predict collision risk.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CollisionPrediction:
    """Output of collision detection."""
    collision_probability: float      # Probability of collision (0-1)
    time_to_collision: float          # Estimated time to collision (seconds)
    threat_direction: np.ndarray      # Unit vector pointing to threat
    threat_velocity: float            # Relative velocity of threat (m/s)
    confidence: float                 # Prediction confidence (0-1)
    threat_id: Optional[str] = None   # ID of primary threat object


class PointNetEncoder(nn.Module):
    """
    PointNet-style encoder for processing 3D point cloud data (lidar).
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: list = [64, 128, 256],
        output_dim: int = 512,
    ):
        super().__init__()
        
        # Shared MLPs
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(in_dim, h_dim, 1),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = h_dim
        
        layers.append(nn.Conv1d(in_dim, output_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Point cloud [batch, num_points, 3] or [batch, num_points, features]
            
        Returns:
            Global feature vector [batch, output_dim]
        """
        # Transpose to [batch, features, num_points]
        x = x.transpose(1, 2)
        
        # Apply MLP to each point
        x = self.mlp(x)
        
        # Global max pooling
        x = torch.max(x, dim=2)[0]
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for feature processing."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x))


class CollisionDetector(nn.Module):
    """
    Neural network for collision detection and risk assessment.
    
    Inputs:
        - Lidar point cloud (nearby objects)
        - Radar data (long-range detections)
        - IMU data (current state)
        
    Outputs:
        - Collision probability
        - Time to collision
        - Threat direction vector
        - Confidence score
    """
    
    def __init__(
        self,
        lidar_points: int = 1024,
        radar_features: int = 64,
        imu_features: int = 12,
        hidden_dim: int = 256,
        num_residual_blocks: int = 3,
    ):
        super().__init__()
        
        self.lidar_points = lidar_points
        self.radar_features = radar_features
        self.imu_features = imu_features
        
        # Lidar encoder (PointNet-style)
        self.lidar_encoder = PointNetEncoder(
            input_dim=6,  # xyz + intensity/velocity
            hidden_dims=[64, 128, 256],
            output_dim=256,
        )
        
        # Radar encoder
        self.radar_encoder = nn.Sequential(
            nn.Linear(radar_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        
        # IMU encoder
        self.imu_encoder = nn.Sequential(
            nn.Linear(imu_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        
        # Fusion network
        fusion_dim = 256 + 128 + 64  # lidar + radar + imu
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)]
        )
        
        # Output heads
        self.collision_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        self.time_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Softplus(),  # Ensure positive time
        )
        
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),  # 3D direction vector
        )
        
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Softplus(),  # Ensure positive velocity
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        lidar: torch.Tensor,
        radar: torch.Tensor,
        imu: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            lidar: Point cloud [batch, num_points, 6]
            radar: Radar features [batch, radar_features]
            imu: IMU data [batch, imu_features]
            
        Returns:
            Dictionary of predictions
        """
        # Encode each modality
        lidar_features = self.lidar_encoder(lidar)
        radar_features = self.radar_encoder(radar)
        imu_features = self.imu_encoder(imu)
        
        # Fuse features
        fused = torch.cat([lidar_features, radar_features, imu_features], dim=1)
        fused = self.fusion(fused)
        
        # Process through residual blocks
        features = self.residual_blocks(fused)
        
        # Generate outputs
        collision_prob = self.collision_head(features)
        time_to_collision = self.time_head(features)
        direction = self.direction_head(features)
        velocity = self.velocity_head(features)
        confidence = self.confidence_head(features)
        
        # Normalize direction to unit vector
        direction = F.normalize(direction, p=2, dim=1)
        
        return {
            "collision_probability": collision_prob.squeeze(-1),
            "time_to_collision": time_to_collision.squeeze(-1),
            "threat_direction": direction,
            "threat_velocity": velocity.squeeze(-1),
            "confidence": confidence.squeeze(-1),
        }
    
    def predict(
        self,
        lidar: np.ndarray,
        radar: np.ndarray,
        imu: np.ndarray,
        device: str = "cpu",
    ) -> CollisionPrediction:
        """
        Make a single prediction.
        
        Args:
            lidar: Lidar point cloud [num_points, 6]
            radar: Radar features [radar_features]
            imu: IMU data [imu_features]
            device: Compute device
            
        Returns:
            CollisionPrediction object
        """
        self.eval()
        
        with torch.no_grad():
            # Convert to tensors and add batch dimension
            lidar_t = torch.FloatTensor(lidar).unsqueeze(0).to(device)
            radar_t = torch.FloatTensor(radar).unsqueeze(0).to(device)
            imu_t = torch.FloatTensor(imu).unsqueeze(0).to(device)
            
            # Pad lidar if needed
            if lidar_t.shape[1] < self.lidar_points:
                pad_size = self.lidar_points - lidar_t.shape[1]
                padding = torch.zeros(1, pad_size, lidar_t.shape[2], device=device)
                lidar_t = torch.cat([lidar_t, padding], dim=1)
            elif lidar_t.shape[1] > self.lidar_points:
                # Random sampling
                indices = torch.randperm(lidar_t.shape[1])[:self.lidar_points]
                lidar_t = lidar_t[:, indices, :]
            
            outputs = self.forward(lidar_t, radar_t, imu_t)
        
        return CollisionPrediction(
            collision_probability=outputs["collision_probability"].item(),
            time_to_collision=outputs["time_to_collision"].item(),
            threat_direction=outputs["threat_direction"].cpu().numpy().squeeze(),
            threat_velocity=outputs["threat_velocity"].item(),
            confidence=outputs["confidence"].item(),
        )


class CollisionDetectorLoss(nn.Module):
    """
    Loss function for training collision detector.
    """
    
    def __init__(
        self,
        collision_weight: float = 10.0,
        time_weight: float = 1.0,
        direction_weight: float = 5.0,
        velocity_weight: float = 1.0,
    ):
        super().__init__()
        self.collision_weight = collision_weight
        self.time_weight = time_weight
        self.direction_weight = direction_weight
        self.velocity_weight = velocity_weight
        
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineEmbeddingLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            
        Returns:
            Dictionary of loss components
        """
        # Collision probability loss (binary cross-entropy)
        collision_loss = self.bce(
            predictions["collision_probability"],
            targets["collision_label"].float(),
        )
        
        # Time to collision loss (only for positive samples)
        mask = targets["collision_label"] > 0.5
        if mask.sum() > 0:
            time_loss = self.mse(
                predictions["time_to_collision"][mask],
                targets["time_to_collision"][mask],
            )
        else:
            time_loss = torch.tensor(0.0, device=predictions["collision_probability"].device)
        
        # Direction loss (cosine similarity)
        if mask.sum() > 0:
            ones = torch.ones(mask.sum(), device=predictions["threat_direction"].device)
            direction_loss = self.cosine(
                predictions["threat_direction"][mask],
                targets["threat_direction"][mask],
                ones,
            )
        else:
            direction_loss = torch.tensor(0.0, device=predictions["collision_probability"].device)
        
        # Velocity loss
        if mask.sum() > 0:
            velocity_loss = self.mse(
                predictions["threat_velocity"][mask],
                targets["threat_velocity"][mask],
            )
        else:
            velocity_loss = torch.tensor(0.0, device=predictions["collision_probability"].device)
        
        # Total loss
        total_loss = (
            self.collision_weight * collision_loss +
            self.time_weight * time_loss +
            self.direction_weight * direction_loss +
            self.velocity_weight * velocity_loss
        )
        
        return {
            "total": total_loss,
            "collision": collision_loss,
            "time": time_loss,
            "direction": direction_loss,
            "velocity": velocity_loss,
        }
