"""
Robotic manipulator control using SAC and imitation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from ....core.base_module import BaseModule


@dataclass
class ManipulatorState:
    """State of robotic manipulator."""
    joint_positions: np.ndarray    # Joint angles (rad)
    joint_velocities: np.ndarray   # Joint angular velocities (rad/s)
    end_effector_pos: np.ndarray   # End effector position (m)
    end_effector_vel: np.ndarray   # End effector velocity (m/s)
    gripper_state: float           # 0=open, 1=closed
    force_torque: np.ndarray       # Force/torque at end effector


class ManipulatorPolicy(nn.Module):
    """Policy network for manipulator control."""
    
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()
        
        # Shared encoder
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = h_dim
        self.encoder = nn.Sequential(*layers)
        
        # Action heads
        self.mean = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Linear(in_dim, action_dim)
    
    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(state)
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std
    
    def sample(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class ManipulatorCritic(nn.Module):
    """Critic network for SAC."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()
        
        # Q1
        q1_layers = []
        in_dim = state_dim + action_dim
        for h_dim in hidden_dims:
            q1_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = h_dim
        q1_layers.append(nn.Linear(in_dim, 1))
        self.q1 = nn.Sequential(*q1_layers)
        
        # Q2
        q2_layers = []
        in_dim = state_dim + action_dim
        for h_dim in hidden_dims:
            q2_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = h_dim
        q2_layers.append(nn.Linear(in_dim, 1))
        self.q2 = nn.Sequential(*q2_layers)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)
    
    def q_min(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class InverseKinematics(nn.Module):
    """Neural network for inverse kinematics."""
    
    def __init__(
        self,
        num_joints: int = 6,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # Input: target position (3) + orientation (4) + current joints (num_joints)
        input_dim = 7 + num_joints
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_joints),
            nn.Tanh(),  # Output in [-1, 1], scale to joint limits
        )
        
        self.joint_limits = nn.Parameter(
            torch.tensor([3.14] * num_joints),  # ±π for each joint
            requires_grad=False,
        )
    
    def forward(
        self,
        target_pose: torch.Tensor,
        current_joints: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute joint angles for target pose.
        
        Args:
            target_pose: [batch, 7] - position (3) + quaternion (4)
            current_joints: [batch, num_joints]
            
        Returns:
            Joint angles [batch, num_joints]
        """
        x = torch.cat([target_pose, current_joints], dim=-1)
        normalized = self.network(x)
        return normalized * self.joint_limits


class ManipulatorController:
    """
    SAC-based controller for robotic manipulator.
    """
    
    def __init__(
        self,
        state_dim: int = 24,
        action_dim: int = 7,
        num_joints: int = 6,
        hidden_dims: List[int] = [256, 256],
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        device: str = "cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.num_joints = num_joints
        
        # Networks
        self.policy = ManipulatorPolicy(state_dim, action_dim, hidden_dims).to(device)
        self.critic = ManipulatorCritic(state_dim, action_dim, hidden_dims).to(device)
        self.critic_target = ManipulatorCritic(state_dim, action_dim, hidden_dims).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # IK helper
        self.ik_network = InverseKinematics(num_joints).to(device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.ik_optimizer = torch.optim.Adam(self.ik_network.parameters(), lr=lr)
        
        # Auto alpha tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select action from policy."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if deterministic:
                mean, _ = self.policy(state_t)
                action = torch.tanh(mean)
            else:
                action, _ = self.policy.sample(state_t)
            
            return action.cpu().numpy().squeeze(0)
    
    def compute_ik(
        self,
        target_pose: np.ndarray,
        current_joints: np.ndarray,
    ) -> np.ndarray:
        """Compute inverse kinematics."""
        with torch.no_grad():
            target = torch.FloatTensor(target_pose).unsqueeze(0).to(self.device)
            current = torch.FloatTensor(current_joints).unsqueeze(0).to(self.device)
            joints = self.ik_network(target, current)
            return joints.cpu().numpy().squeeze(0)
    
    def update(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Update networks with batch."""
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        
        # Critic update
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            next_q = self.critic_target.q_min(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * (next_q - self.alpha * next_log_probs)
        
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Policy update
        new_actions, log_probs = self.policy.sample(states)
        q_new = self.critic.q_min(states, new_actions)
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Alpha update
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()
        
        # Soft update target
        for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_targ.data.copy_(self.tau * p.data + (1 - self.tau) * p_targ.data)
        
        return {
            "critic_loss": critic_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": self.alpha,
        }
    
    def save(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "ik_network": self.ik_network.state_dict(),
            "alpha": self.alpha,
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.ik_network.load_state_dict(checkpoint["ik_network"])
        self.alpha = checkpoint["alpha"]


class ManipulatorModule(BaseModule):
    """Complete manipulator control module."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        super().__init__(
            name="manipulator_control",
            config=config,
            device=device,
        )
        
        self.controller = None
        self.num_joints = config.get("num_joints", 6)
    
    def _build_model(self) -> nn.Module:
        self.controller = ManipulatorController(
            state_dim=self.config.get("state_dim", 24),
            action_dim=self.config.get("action_dim", 7),
            num_joints=self.num_joints,
            device=self.device,
        )
        return self.controller.policy
    
    def _preprocess(self, inputs: Dict[str, Any]) -> np.ndarray:
        state = []
        
        # Joint states
        if "joint_positions" in inputs:
            state.extend(np.asarray(inputs["joint_positions"]).flatten().tolist())
        else:
            state.extend([0.0] * self.num_joints)
        
        if "joint_velocities" in inputs:
            state.extend(np.asarray(inputs["joint_velocities"]).flatten().tolist())
        else:
            state.extend([0.0] * self.num_joints)
        
        # Target pose
        if "target_position" in inputs:
            state.extend(np.asarray(inputs["target_position"]).flatten().tolist())
        else:
            state.extend([0.0] * 3)
        
        # Relative position to target
        if "relative_position" in inputs:
            state.extend(np.asarray(inputs["relative_position"]).flatten().tolist())
        else:
            state.extend([0.0] * 3)
        
        # Gripper state
        state.append(inputs.get("gripper_state", 0.0))
        
        # Pad to expected dimension
        expected_dim = self.config.get("state_dim", 24)
        while len(state) < expected_dim:
            state.append(0.0)
        
        return np.array(state[:expected_dim], dtype=np.float32)
    
    def _postprocess(self, outputs: np.ndarray) -> Dict[str, Any]:
        return {
            "joint_velocities": outputs[:self.num_joints],
            "gripper_command": outputs[self.num_joints] if len(outputs) > self.num_joints else 0.0,
        }
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        state = self._preprocess(inputs)
        action = self.controller.select_action(state, deterministic=True)
        return self._postprocess(action)
    
    def _fallback(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Simple proportional control fallback."""
        if "target_position" in inputs and "end_effector_pos" in inputs:
            target = np.asarray(inputs["target_position"])
            current = np.asarray(inputs["end_effector_pos"])
            error = target - current
            
            # Simple P-control
            velocity = np.clip(error * 0.5, -0.1, 0.1)
            
            return {
                "joint_velocities": np.zeros(self.num_joints),
                "end_effector_velocity": velocity,
                "gripper_command": 0.0,
                "fallback": True,
            }
        
        return {
            "joint_velocities": np.zeros(self.num_joints),
            "gripper_command": 0.0,
            "fallback": True,
        }
