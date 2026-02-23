"""
Collision Avoidance Agent using Soft Actor-Critic (SAC).
Provides reactive avoidance control based on collision predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ....core.base_module import BaseModule


class GaussianPolicy(nn.Module):
    """
    Gaussian policy network for continuous action spaces.
    Outputs mean and log_std of action distribution.
    """
    
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
    ):
        super().__init__()
        
        # Build network
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = h_dim
        
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)
    
    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch, state_dim]
            
        Returns:
            mean, log_std tensors
        """
        features = self.backbone(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std
    
    def sample(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: State tensor
            
        Returns:
            action, log_prob tensors
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from Gaussian
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Apply tanh squashing
        action = torch.tanh(x_t)
        
        # Compute log probability with correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        """Get deterministic action (mean)."""
        mean, _ = self.forward(state)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    """
    Q-value network for SAC.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
    ):
        super().__init__()
        
        layers = []
        in_dim = state_dim + action_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q-value."""
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class TwinQNetwork(nn.Module):
    """
    Twin Q-networks for reducing overestimation bias.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
    ):
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute both Q-values."""
        return self.q1(state, action), self.q2(state, action)
    
    def min_q(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute minimum of both Q-values."""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class SACAgent:
    """
    Soft Actor-Critic agent for collision avoidance.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize SAC agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            lr: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Entropy coefficient
            auto_alpha: Auto-tune alpha
            device: Compute device
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        
        # Networks
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dims).to(device)
        self.critic = TwinQNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic_target = TwinQNetwork(state_dim, action_dim, hidden_dims).to(device)
        
        # Copy weights to target
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Auto alpha tuning
        if auto_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Select action given state.
        
        Args:
            state: State array
            deterministic: Use deterministic policy
            
        Returns:
            Action array
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if deterministic:
                action = self.policy.deterministic(state_t)
            else:
                action, _ = self.policy.sample(state_t)
            
            return action.cpu().numpy().squeeze(0)
    
    def update(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Update networks with batch of transitions.
        
        Args:
            batch: Dictionary with states, actions, rewards, next_states, dones
            
        Returns:
            Dictionary of losses
        """
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            next_q = self.critic_target.min_q(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * (next_q - self.alpha * next_log_probs)
        
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update policy
        new_actions, log_probs = self.policy.sample(states)
        q_new = self.critic.min_q(states, new_actions)
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update alpha
        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target networks
        self._soft_update()
        
        return {
            "critic_loss": critic_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss if isinstance(alpha_loss, float) else alpha_loss.item(),
            "alpha": self.alpha,
        }
    
    def _soft_update(self):
        """Soft update target networks."""
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha": self.alpha,
        }, path)
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.alpha = checkpoint["alpha"]


class CollisionAvoidanceModule(BaseModule):
    """
    Complete collision avoidance module integrating detector and agent.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        super().__init__(
            name="collision_avoidance",
            config=config,
            device=device,
        )
        
        self.detector = None
        self.agent = None
        
        # Extract config
        self.state_dim = config.get("state_dim", 44)
        self.action_dim = config.get("action_dim", 6)  # 3D thrust + 3D torque
        self.hidden_dims = config.get("hidden_dims", [256, 256])
        self.collision_threshold = config.get("collision_threshold", 0.7)
        self.min_reaction_time = config.get("min_reaction_time", 0.5)
    
    def _build_model(self) -> nn.Module:
        """Build the neural network models."""
        from .detector import CollisionDetector
        
        # Build detector
        self.detector = CollisionDetector(
            lidar_points=self.config.get("lidar_points", 1024),
            radar_features=self.config.get("radar_features", 64),
            imu_features=self.config.get("imu_features", 12),
            hidden_dim=self.config.get("detector_hidden_dim", 256),
        ).to(self.device)
        
        # Build SAC agent
        self.agent = SACAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            device=self.device,
        )
        
        # Return policy as main model for base class
        return self.agent.policy
    
    def _preprocess(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Preprocess inputs for the model."""
        processed = {}
        
        # Convert numpy to tensors
        if "state" in inputs:
            processed["state"] = torch.FloatTensor(inputs["state"]).to(self.device)
            if processed["state"].dim() == 1:
                processed["state"] = processed["state"].unsqueeze(0)
        
        if "lidar" in inputs:
            processed["lidar"] = torch.FloatTensor(inputs["lidar"]).to(self.device)
            if processed["lidar"].dim() == 2:
                processed["lidar"] = processed["lidar"].unsqueeze(0)
        
        if "radar" in inputs:
            processed["radar"] = torch.FloatTensor(inputs["radar"]).to(self.device)
            if processed["radar"].dim() == 1:
                processed["radar"] = processed["radar"].unsqueeze(0)
        
        if "imu" in inputs:
            processed["imu"] = torch.FloatTensor(inputs["imu"]).to(self.device)
            if processed["imu"].dim() == 1:
                processed["imu"] = processed["imu"].unsqueeze(0)
        
        return processed
    
    def _postprocess(self, outputs: Any) -> Dict[str, Any]:
        """Postprocess model outputs."""
        if isinstance(outputs, dict):
            return {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                    for k, v in outputs.items()}
        return {"action": outputs.cpu().numpy()}
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute collision avoidance.
        
        Args:
            inputs: Dictionary with state, lidar, radar, imu data
            
        Returns:
            Dictionary with action and collision info
        """
        processed = self._preprocess(inputs)
        
        results = {
            "action": np.zeros(self.action_dim),
            "collision_detected": False,
            "avoidance_active": False,
        }
        
        # Run collision detection if sensor data available
        if all(k in processed for k in ["lidar", "radar", "imu"]):
            with torch.no_grad():
                detection = self.detector(
                    processed["lidar"],
                    processed["radar"],
                    processed["imu"],
                )
            
            collision_prob = detection["collision_probability"].item()
            time_to_collision = detection["time_to_collision"].item()
            
            results["collision_probability"] = collision_prob
            results["time_to_collision"] = time_to_collision
            results["threat_direction"] = detection["threat_direction"].cpu().numpy().squeeze()
            
            # Check if avoidance needed
            if collision_prob > self.collision_threshold or time_to_collision < self.min_reaction_time:
                results["collision_detected"] = True
                results["avoidance_active"] = True
        
        # Get avoidance action from agent
        if "state" in processed and results["avoidance_active"]:
            action = self.agent.select_action(
                processed["state"].cpu().numpy().squeeze(),
                deterministic=True,  # Use deterministic in deployment
            )
            results["action"] = action
        
        return results
    
    def _fallback(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classical fallback for collision avoidance.
        Simple repulsive potential field approach.
        """
        results = {
            "action": np.zeros(self.action_dim),
            "collision_detected": False,
            "avoidance_active": False,
            "fallback": True,
        }
        
        # Simple heuristic: if threat direction is known, thrust away from it
        if "threat_direction" in inputs:
            threat_dir = np.array(inputs["threat_direction"])
            # Thrust in opposite direction
            avoidance_thrust = -threat_dir * 0.5  # 50% thrust
            results["action"][:3] = avoidance_thrust
            results["avoidance_active"] = True
        
        return results
    
    def save(self, path: str):
        """Save the complete module."""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save detector
        torch.save(self.detector.state_dict(), f"{path}/detector.pt")
        
        # Save agent
        self.agent.save(f"{path}/agent.pt")
    
    def load(self, path: str):
        """Load the complete module."""
        # Load detector
        self.detector.load_state_dict(
            torch.load(f"{path}/detector.pt", map_location=self.device)
        )
        
        # Load agent
        self.agent.load(f"{path}/agent.pt")


class CollisionAvoidanceAgent(CollisionAvoidanceModule):
    """Alias for backward compatibility."""
    pass
