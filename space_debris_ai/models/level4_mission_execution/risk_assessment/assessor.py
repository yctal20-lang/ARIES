"""
Risk Assessment module for evaluating debris capture risk.
Uses MLP to score object danger and capture difficulty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from ....core.base_module import BaseModule


class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = 0
    MODERATE = 1
    HIGH = 2
    CRITICAL = 3
    EXTREME = 4


@dataclass
class RiskAssessment:
    """Risk assessment result for a debris object."""
    object_id: str
    capture_risk: float          # Risk of capture failure (0-1)
    collision_risk: float        # Risk of collision during approach (0-1)
    damage_potential: float      # Potential damage if collision (0-1)
    capture_difficulty: float    # Difficulty of capture (0-1)
    priority_score: float        # Collection priority (0-1, higher = more urgent)
    risk_level: RiskLevel
    recommended_approach: str
    estimated_fuel_cost: float   # Estimated fuel for capture (kg)
    estimated_time: float        # Estimated time to capture (s)
    confidence: float


class RiskMLP(nn.Module):
    """
    Multi-layer perceptron for risk assessment.
    Takes object features and outputs risk scores.
    """
    
    def __init__(
        self,
        input_dim: int = 24,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Feature extraction layers
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        
        self.features = nn.Sequential(*layers)
        
        # Output heads
        self.capture_risk_head = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        
        self.collision_risk_head = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        
        self.damage_potential_head = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        
        self.difficulty_head = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        
        self.priority_head = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        
        # Risk level classifier
        self.risk_classifier = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 5),  # 5 risk levels
        )
        
        # Regression heads
        self.fuel_estimator = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Softplus(),  # Positive values
        )
        
        self.time_estimator = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Softplus(),
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features [batch, input_dim]
            
        Returns:
            Dictionary of risk scores
        """
        features = self.features(x)
        
        return {
            "capture_risk": self.capture_risk_head(features).squeeze(-1),
            "collision_risk": self.collision_risk_head(features).squeeze(-1),
            "damage_potential": self.damage_potential_head(features).squeeze(-1),
            "capture_difficulty": self.difficulty_head(features).squeeze(-1),
            "priority_score": self.priority_head(features).squeeze(-1),
            "risk_level_logits": self.risk_classifier(features),
            "estimated_fuel": self.fuel_estimator(features).squeeze(-1),
            "estimated_time": self.time_estimator(features).squeeze(-1),
            "confidence": self.confidence_head(features).squeeze(-1),
        }


class RiskAssessor(nn.Module):
    """
    Complete risk assessment system.
    Combines learned MLP with physics-based heuristics.
    """
    
    def __init__(
        self,
        input_dim: int = 24,
        hidden_dims: List[int] = [128, 64, 32],
        use_physics_features: bool = True,
    ):
        super().__init__()
        
        self.use_physics_features = use_physics_features
        
        # Physics feature dimension
        physics_dim = 8 if use_physics_features else 0
        total_input = input_dim + physics_dim
        
        self.mlp = RiskMLP(
            input_dim=total_input,
            hidden_dims=hidden_dims,
        )
        
        # Approach type classifier
        self.approach_classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 4),  # 4 approach types
        )
        
        self.approach_types = [
            "direct_approach",
            "spiral_approach", 
            "tangential_approach",
            "abort_recommended",
        ]
    
    def compute_physics_features(
        self,
        debris_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute physics-based risk features.
        
        Args:
            debris_features: Raw debris features
            
        Returns:
            Physics-derived features
        """
        batch_size = debris_features.shape[0]
        
        # Extract key features (assuming standard ordering)
        # [size, mass, rel_velocity, distance, rotation_rate, ...]
        
        # Momentum risk: mass * velocity
        mass = debris_features[:, 1:2] if debris_features.shape[1] > 1 else torch.ones(batch_size, 1)
        velocity = debris_features[:, 2:3] if debris_features.shape[1] > 2 else torch.zeros(batch_size, 1)
        momentum_risk = mass * torch.abs(velocity) / 100  # Normalized
        
        # Size difficulty: larger objects harder to capture
        size = debris_features[:, 0:1] if debris_features.shape[1] > 0 else torch.ones(batch_size, 1)
        size_factor = torch.tanh(size / 5)  # Saturates at ~5m
        
        # Distance urgency: inverse distance
        distance = debris_features[:, 3:4] if debris_features.shape[1] > 3 else torch.ones(batch_size, 1)
        distance_factor = 1 / (distance + 0.1)  # Avoid division by zero
        
        # Rotation difficulty
        rotation = debris_features[:, 4:5] if debris_features.shape[1] > 4 else torch.zeros(batch_size, 1)
        rotation_factor = torch.tanh(torch.abs(rotation) / 2)
        
        # Approach time estimate (simple kinematic)
        time_estimate = distance / (torch.abs(velocity) + 0.01)
        
        # Kinetic energy (damage potential)
        ke = 0.5 * mass * velocity**2
        ke_normalized = torch.tanh(ke / 1000)
        
        # Combined danger score
        danger = (momentum_risk + ke_normalized) / 2
        
        # Urgency (combination of distance and velocity)
        urgency = distance_factor * (1 + torch.abs(velocity))
        
        physics_features = torch.cat([
            momentum_risk,
            size_factor,
            distance_factor,
            rotation_factor,
            time_estimate,
            ke_normalized,
            danger,
            urgency,
        ], dim=-1)
        
        return physics_features
    
    def forward(
        self,
        debris_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Assess risk for debris objects.
        
        Args:
            debris_features: Object features [batch, input_dim]
            
        Returns:
            Risk assessment outputs
        """
        if self.use_physics_features:
            physics = self.compute_physics_features(debris_features)
            combined = torch.cat([debris_features, physics], dim=-1)
        else:
            combined = debris_features
        
        outputs = self.mlp(combined)
        
        # Get approach recommendation
        features = self.mlp.features(combined)
        approach_logits = self.approach_classifier(features)
        outputs["approach_logits"] = approach_logits
        outputs["approach_probs"] = F.softmax(approach_logits, dim=-1)
        
        return outputs
    
    def assess(
        self,
        debris_features: np.ndarray,
        object_id: str = "unknown",
        device: str = "cpu",
    ) -> RiskAssessment:
        """
        Assess risk for a single debris object.
        
        Args:
            debris_features: Feature vector
            object_id: Object identifier
            device: Compute device
            
        Returns:
            RiskAssessment dataclass
        """
        self.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(debris_features).unsqueeze(0).to(device)
            outputs = self.forward(x)
        
        # Extract predictions
        risk_logits = outputs["risk_level_logits"].cpu().numpy().squeeze()
        risk_level = RiskLevel(np.argmax(risk_logits))
        
        approach_probs = outputs["approach_probs"].cpu().numpy().squeeze()
        approach_idx = np.argmax(approach_probs)
        recommended_approach = self.approach_types[approach_idx]
        
        return RiskAssessment(
            object_id=object_id,
            capture_risk=outputs["capture_risk"].item(),
            collision_risk=outputs["collision_risk"].item(),
            damage_potential=outputs["damage_potential"].item(),
            capture_difficulty=outputs["capture_difficulty"].item(),
            priority_score=outputs["priority_score"].item(),
            risk_level=risk_level,
            recommended_approach=recommended_approach,
            estimated_fuel_cost=outputs["estimated_fuel"].item(),
            estimated_time=outputs["estimated_time"].item(),
            confidence=outputs["confidence"].item(),
        )


class RiskAssessmentModule(BaseModule):
    """Complete risk assessment module."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        super().__init__(
            name="risk_assessment",
            config=config,
            device=device,
        )
        
        self.assessor = None
        self.input_dim = config.get("input_dim", 24)
    
    def _build_model(self) -> nn.Module:
        self.assessor = RiskAssessor(
            input_dim=self.input_dim,
            hidden_dims=self.config.get("hidden_dims", [128, 64, 32]),
            use_physics_features=self.config.get("use_physics_features", True),
        ).to(self.device)
        
        return self.assessor.mlp
    
    def _preprocess(self, inputs: Dict[str, Any]) -> np.ndarray:
        """Build feature vector from inputs."""
        features = []
        
        # Object properties
        features.append(inputs.get("size", 1.0))
        features.append(inputs.get("mass", 10.0))
        
        # Relative motion
        rel_velocity = np.asarray(inputs.get("relative_velocity", [0, 0, 0]))
        features.append(np.linalg.norm(rel_velocity))
        
        rel_position = np.asarray(inputs.get("relative_position", [1, 0, 0]))
        features.append(np.linalg.norm(rel_position))
        
        # Rotation
        angular_velocity = np.asarray(inputs.get("angular_velocity", [0, 0, 0]))
        features.append(np.linalg.norm(angular_velocity))
        
        # Object classification
        features.append(inputs.get("debris_type_code", 0))
        features.append(inputs.get("material_code", 0))
        
        # Current spacecraft state
        fuel_remaining = inputs.get("fuel_remaining", 100.0)
        features.append(fuel_remaining / 100.0)  # Normalized
        
        battery_soc = inputs.get("battery_soc", 1.0)
        features.append(battery_soc)
        
        # Add relative position/velocity components
        features.extend(rel_position.flatten().tolist())
        features.extend(rel_velocity.flatten().tolist())
        
        # Pad to input dimension
        while len(features) < self.input_dim:
            features.append(0.0)
        
        return np.array(features[:self.input_dim], dtype=np.float32)
    
    def _postprocess(self, outputs: RiskAssessment) -> Dict[str, Any]:
        return {
            "object_id": outputs.object_id,
            "capture_risk": outputs.capture_risk,
            "collision_risk": outputs.collision_risk,
            "damage_potential": outputs.damage_potential,
            "capture_difficulty": outputs.capture_difficulty,
            "priority_score": outputs.priority_score,
            "risk_level": outputs.risk_level.name,
            "recommended_approach": outputs.recommended_approach,
            "estimated_fuel_kg": outputs.estimated_fuel_cost,
            "estimated_time_s": outputs.estimated_time,
            "confidence": outputs.confidence,
        }
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        features = self._preprocess(inputs)
        object_id = inputs.get("object_id", "unknown")
        
        assessment = self.assessor.assess(features, object_id, self.device)
        return self._postprocess(assessment)
    
    def assess_multiple(
        self,
        objects: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Assess risk for multiple objects and rank by priority.
        
        Args:
            objects: List of object feature dictionaries
            
        Returns:
            Sorted list of risk assessments (highest priority first)
        """
        assessments = [self.forward(obj) for obj in objects]
        
        # Sort by priority (descending)
        assessments.sort(key=lambda x: x["priority_score"], reverse=True)
        
        return assessments
    
    def _fallback(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Simple heuristic-based fallback."""
        size = inputs.get("size", 1.0)
        mass = inputs.get("mass", 10.0)
        rel_velocity = np.linalg.norm(inputs.get("relative_velocity", [0, 0, 0]))
        distance = np.linalg.norm(inputs.get("relative_position", [1, 0, 0]))
        rotation = np.linalg.norm(inputs.get("angular_velocity", [0, 0, 0]))
        
        # Simple heuristic calculations
        capture_risk = min(1.0, 0.1 + rotation * 0.3 + size * 0.05)
        collision_risk = min(1.0, rel_velocity * 0.5 / max(distance, 0.1))
        damage_potential = min(1.0, mass * rel_velocity**2 / 1000)
        capture_difficulty = min(1.0, 0.2 + size * 0.1 + rotation * 0.4)
        priority_score = min(1.0, size / 5 + damage_potential * 0.5)
        
        # Determine risk level
        avg_risk = (capture_risk + collision_risk + damage_potential) / 3
        if avg_risk < 0.2:
            risk_level = "LOW"
        elif avg_risk < 0.4:
            risk_level = "MODERATE"
        elif avg_risk < 0.6:
            risk_level = "HIGH"
        elif avg_risk < 0.8:
            risk_level = "CRITICAL"
        else:
            risk_level = "EXTREME"
        
        return {
            "object_id": inputs.get("object_id", "unknown"),
            "capture_risk": capture_risk,
            "collision_risk": collision_risk,
            "damage_potential": damage_potential,
            "capture_difficulty": capture_difficulty,
            "priority_score": priority_score,
            "risk_level": risk_level,
            "recommended_approach": "direct_approach" if avg_risk < 0.5 else "abort_recommended",
            "estimated_fuel_kg": distance * 0.5 + mass * 0.01,
            "estimated_time_s": distance * 100,
            "confidence": 0.5,
            "fallback": True,
        }

