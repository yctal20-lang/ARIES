"""
Energy Management Agent using PPO.
Optimizes power distribution across subsystems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from ....core.base_module import BaseModule
from .power_model import PowerModel, PowerState, EnergyMode


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    Actor outputs power allocation for each subsystem.
    Critic estimates state value.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()
        
        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = h_dim
        self.shared = nn.Sequential(*layers)
        
        # Actor head (outputs mean of action distribution)
        self.actor_mean = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[-1], action_dim),
            nn.Sigmoid(),  # Actions bounded [0, 1]
        )
        
        # Learnable log std
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (outputs state value)
        self.critic = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[-1], 1),
        )
    
    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor [batch, obs_dim]
            
        Returns:
            action_mean, action_log_std, value
        """
        features = self.shared(obs)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        
        return action_mean, self.actor_log_std.expand_as(action_mean), value
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from policy.
        
        Args:
            obs: Observation
            deterministic: If True, return mean action
            
        Returns:
            action, log_prob, value
        """
        action_mean, action_log_std, value = self.forward(obs)
        
        if deterministic:
            return action_mean, torch.zeros_like(action_mean[:, 0]), value
        
        std = action_log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        action = torch.clamp(action, 0, 1)  # Ensure valid range
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.
        
        Args:
            obs: Observations
            actions: Actions taken
            
        Returns:
            log_probs, values, entropy
        """
        action_mean, action_log_std, values = self.forward(obs)
        std = action_log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, values.squeeze(-1), entropy


class PPOAgent:
    """
    Proximal Policy Optimization agent for energy management.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        """
        Initialize PPO agent.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer sizes
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_ratio: PPO clip ratio
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm for clipping
            device: Compute device
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.network = ActorCritic(obs_dim, action_dim, hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
    
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action given observation.
        
        Args:
            obs: Observation array
            deterministic: Use deterministic policy
            
        Returns:
            action, log_prob, value
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob, value = self.network.get_action(obs_t, deterministic)
        
        return (
            action.cpu().numpy().squeeze(0),
            log_prob.item(),
            value.item(),
        )
    
    def update(
        self,
        batch: Dict[str, torch.Tensor],
        n_epochs: int = 10,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """
        Update policy with collected experience.
        
        Args:
            batch: Dictionary with obs, actions, returns, advantages, old_log_probs
            n_epochs: Number of update epochs
            batch_size: Mini-batch size
            
        Returns:
            Dictionary of loss metrics
        """
        obs = batch["obs"]
        actions = batch["actions"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        old_log_probs = batch["old_log_probs"]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        n_samples = obs.shape[0]
        metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0}
        n_updates = 0
        
        for _ in range(n_epochs):
            # Random permutation
            indices = torch.randperm(n_samples)
            
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                obs_batch = obs[batch_indices]
                action_batch = actions[batch_indices]
                return_batch = returns[batch_indices]
                adv_batch = advantages[batch_indices]
                old_log_prob_batch = old_log_probs[batch_indices]
                
                # Evaluate current policy
                log_probs, values, entropy = self.network.evaluate_actions(
                    obs_batch, action_batch
                )
                
                # PPO loss
                ratio = torch.exp(log_probs - old_log_prob_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, return_batch)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += -entropy_loss.item()
                n_updates += 1
        
        # Average metrics
        for key in metrics:
            metrics[key] /= n_updates
        
        return metrics
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            last_value: Value estimate for last state
            
        Returns:
            returns, advantages
        """
        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)
        
        gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return returns, advantages
    
    def save(self, path: str):
        """Save agent."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class EnergyAgent(PPOAgent):
    """Alias for EnergyManagement PPO agent."""
    pass


class EnergyManagementModule(BaseModule):
    """
    Complete energy management module.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        super().__init__(
            name="energy_management",
            config=config,
            device=device,
        )
        
        self.power_model = None
        self.agent = None
        
        # Get config
        self.solar_panel_area = config.get("solar_panel_area", 20.0)
        self.battery_capacity = config.get("battery_capacity", 10000.0)
    
    def _build_model(self) -> nn.Module:
        """Build energy management models."""
        # Initialize power model
        self.power_model = PowerModel(
            solar_panel_area=self.solar_panel_area,
            battery_capacity=self.battery_capacity,
        )
        
        # Calculate dimensions
        obs_dim = len(self.power_model.get_observation())
        action_dim = len(self.power_model.subsystems)
        
        # Initialize PPO agent
        self.agent = EnergyAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=self.config.get("hidden_dims", [256, 256]),
            device=self.device,
        )
        
        return self.agent.network
    
    def _preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess inputs."""
        return inputs
    
    def _postprocess(self, outputs: PowerState) -> Dict[str, Any]:
        """Convert power state to output dict."""
        return {
            "battery_soc": outputs.battery_soc,
            "battery_charge": outputs.battery_charge,
            "solar_generation": outputs.solar_generation,
            "total_consumption": outputs.total_consumption,
            "net_power": outputs.net_power,
            "time_to_empty": outputs.time_to_empty,
            "mode": outputs.mode.value,
            "subsystem_allocation": outputs.subsystem_allocation,
            "in_eclipse": outputs.in_eclipse,
            "is_critical": outputs.is_critical,
        }
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute energy management step.
        
        Args:
            inputs: Dictionary with:
                - dt: Time step (seconds)
                - sun_angle: Angle to sun (radians)
                - spacecraft_position: Position for eclipse calculation (optional)
                - sun_direction: Direction to sun (optional)
                - mode_override: Force specific mode (optional)
                
        Returns:
            Power state and allocations
        """
        if self.power_model is None:
            raise RuntimeError("Module not initialized")
        
        dt = inputs.get("dt", 1.0)
        sun_angle = inputs.get("sun_angle", 0.0)
        
        # Update eclipse state if position provided
        if "spacecraft_position" in inputs and "sun_direction" in inputs:
            self.power_model.update_eclipse(
                inputs["spacecraft_position"],
                inputs["sun_direction"],
            )
        
        # Mode override
        if "mode_override" in inputs:
            self.power_model.mode = EnergyMode(inputs["mode_override"])
        
        # Get observation
        obs = self.power_model.get_observation()
        
        # Get action from agent
        action, _, _ = self.agent.select_action(obs, deterministic=True)
        
        # Convert action to allocations
        allocations = {}
        for i, name in enumerate(self.power_model.subsystems.keys()):
            allocations[name] = float(action[i])
        
        # Step power model
        power_state = self.power_model.step(dt, sun_angle, allocations)
        
        return self._postprocess(power_state)
    
    def _fallback(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to mode-based power management."""
        if self.power_model is None:
            self._build_model()
        
        dt = inputs.get("dt", 1.0)
        sun_angle = inputs.get("sun_angle", 0.0)
        
        # Use mode-based allocations
        power_state = self.power_model.step(dt, sun_angle, allocations=None)
        
        result = self._postprocess(power_state)
        result["fallback"] = True
        return result
