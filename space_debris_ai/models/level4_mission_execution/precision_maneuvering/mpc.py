"""
Neural Model Predictive Control for precision orbital maneuvering.
Uses learned dynamics model for trajectory optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from ....core.base_module import BaseModule


@dataclass
class TrajectoryPlan:
    """Planned trajectory from MPC."""
    positions: np.ndarray           # Planned positions [horizon, 3]
    velocities: np.ndarray          # Planned velocities [horizon, 3]
    thrust_commands: np.ndarray     # Optimal thrust commands [horizon, 3]
    torque_commands: np.ndarray     # Optimal torque commands [horizon, 3]
    predicted_cost: float           # Total predicted cost
    feasible: bool                  # Whether plan is feasible
    horizon_time: float             # Time span of plan


class SpacecraftDynamicsModel(nn.Module):
    """
    Neural network model of spacecraft dynamics.
    Learns residual dynamics on top of physics-based model.
    """
    
    def __init__(
        self,
        state_dim: int = 13,      # pos(3) + vel(3) + att(4) + omega(3)
        action_dim: int = 6,       # thrust(3) + torque(3)
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Physics parameters (learnable refinements)
        self.mu = nn.Parameter(torch.tensor(398600.4418))  # GM Earth
        self.drag_coef = nn.Parameter(torch.tensor(0.0001))
        
        # Residual dynamics network
        layers = []
        in_dim = state_dim + action_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),  # Bounded activation for stability
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, state_dim))
        
        self.residual_net = nn.Sequential(*layers)
        
        # Scale factors for residual (start small)
        self.residual_scale = nn.Parameter(torch.ones(state_dim) * 0.01)
    
    def physics_dynamics(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        dt: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute physics-based state derivative.
        
        Args:
            state: Current state [batch, state_dim]
            action: Control action [batch, action_dim]
            dt: Time step
            
        Returns:
            Next state from physics model
        """
        pos = state[:, :3]
        vel = state[:, 3:6]
        att = state[:, 6:10]
        omega = state[:, 10:13]
        
        thrust = action[:, :3]
        torque = action[:, 3:6]
        
        # Gravitational acceleration
        r = torch.norm(pos, dim=-1, keepdim=True)
        a_grav = -self.mu * pos / (r ** 3 + 1e-6)
        
        # Thrust acceleration (simplified: assuming body frame aligned)
        a_thrust = thrust / 500.0  # Assuming 500kg mass
        
        # Simple atmospheric drag (exponential model)
        altitude = r - 6371.0
        rho = 1e-12 * torch.exp(-altitude / 50.0)  # Simplified density
        a_drag = -self.drag_coef * rho * vel * torch.norm(vel, dim=-1, keepdim=True)
        
        # Total acceleration
        a_total = a_grav + a_thrust + a_drag
        
        # Integrate position and velocity (Euler)
        new_vel = vel + a_total * dt
        new_pos = pos + vel * dt + 0.5 * a_total * dt**2
        
        # Attitude dynamics (simplified)
        # Quaternion rate from angular velocity
        omega_quat = torch.zeros_like(att)
        omega_quat[:, 1:4] = omega * dt * 0.5
        
        # Quaternion update (approximate)
        new_att = att + omega_quat
        new_att = new_att / (torch.norm(new_att, dim=-1, keepdim=True) + 1e-6)
        
        # Angular velocity update (simplified Euler equation)
        new_omega = omega + torque * dt / 100.0  # Assuming inertia ~100
        
        return torch.cat([new_pos, new_vel, new_att, new_omega], dim=-1)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        dt: float = 0.1,
    ) -> torch.Tensor:
        """
        Predict next state (physics + learned residual).
        
        Args:
            state: Current state [batch, state_dim]
            action: Control action [batch, action_dim]
            dt: Time step
            
        Returns:
            Predicted next state
        """
        # Physics prediction
        physics_next = self.physics_dynamics(state, action, dt)
        
        # Learned residual
        x = torch.cat([state, action], dim=-1)
        residual = self.residual_net(x) * self.residual_scale
        
        return physics_next + residual * dt
    
    def rollout(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
        dt: float = 0.1,
    ) -> torch.Tensor:
        """
        Roll out trajectory given action sequence.
        
        Args:
            initial_state: Starting state [batch, state_dim]
            actions: Action sequence [batch, horizon, action_dim]
            dt: Time step
            
        Returns:
            State trajectory [batch, horizon+1, state_dim]
        """
        batch_size = initial_state.shape[0]
        horizon = actions.shape[1]
        
        states = [initial_state]
        state = initial_state
        
        for t in range(horizon):
            state = self.forward(state, actions[:, t], dt)
            states.append(state)
        
        return torch.stack(states, dim=1)


class TrajectoryOptimizer(nn.Module):
    """
    Neural trajectory optimizer using learned dynamics.
    Optimizes control sequence to minimize cost.
    """
    
    def __init__(
        self,
        state_dim: int = 13,
        action_dim: int = 6,
        horizon: int = 20,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),  # Current + target
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Action sequence generator (outputs mean actions)
        self.action_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, horizon * action_dim),
            nn.Tanh(),  # Actions in [-1, 1]
        )
        
        # Cost predictor
        self.cost_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        
        # Action scaling
        self.action_scale = nn.Parameter(torch.tensor([100.0, 100.0, 100.0, 10.0, 10.0, 10.0]))
    
    def forward(
        self,
        current_state: torch.Tensor,
        target_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate optimal action sequence.
        
        Args:
            current_state: Current state [batch, state_dim]
            target_state: Target state [batch, state_dim]
            
        Returns:
            actions: Optimal actions [batch, horizon, action_dim]
            predicted_cost: Predicted trajectory cost [batch]
        """
        batch_size = current_state.shape[0]
        
        # Encode state
        x = torch.cat([current_state, target_state], dim=-1)
        features = self.state_encoder(x)
        
        # Generate action sequence
        actions_flat = self.action_generator(features)
        actions = actions_flat.view(batch_size, self.horizon, self.action_dim)
        
        # Scale actions
        actions = actions * self.action_scale
        
        # Predict cost
        cost = self.cost_predictor(features).squeeze(-1)
        
        return actions, cost


class NeuralMPC(nn.Module):
    """
    Neural Model Predictive Controller for spacecraft.
    Combines learned dynamics model with trajectory optimization.
    """
    
    def __init__(
        self,
        state_dim: int = 13,
        action_dim: int = 6,
        horizon: int = 20,
        hidden_dim: int = 128,
        dt: float = 0.1,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.dt = dt
        
        # Dynamics model
        self.dynamics = SpacecraftDynamicsModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )
        
        # Trajectory optimizer
        self.optimizer = TrajectoryOptimizer(
            state_dim=state_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
        )
        
        # Cost weights
        self.position_weight = nn.Parameter(torch.tensor(1.0))
        self.velocity_weight = nn.Parameter(torch.tensor(0.5))
        self.attitude_weight = nn.Parameter(torch.tensor(0.1))
        self.control_weight = nn.Parameter(torch.tensor(0.01))
        self.terminal_weight = nn.Parameter(torch.tensor(10.0))
    
    def compute_cost(
        self,
        trajectory: torch.Tensor,
        actions: torch.Tensor,
        target_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute trajectory cost.
        
        Args:
            trajectory: State trajectory [batch, horizon+1, state_dim]
            actions: Action sequence [batch, horizon, action_dim]
            target_state: Target state [batch, state_dim]
            
        Returns:
            Total cost [batch]
        """
        # Position error
        pos_error = trajectory[:, :, :3] - target_state[:, None, :3]
        pos_cost = (pos_error ** 2).sum(dim=-1).mean(dim=-1)
        
        # Velocity error
        vel_error = trajectory[:, :, 3:6] - target_state[:, None, 3:6]
        vel_cost = (vel_error ** 2).sum(dim=-1).mean(dim=-1)
        
        # Attitude error (simplified)
        att_error = trajectory[:, :, 6:10] - target_state[:, None, 6:10]
        att_cost = (att_error ** 2).sum(dim=-1).mean(dim=-1)
        
        # Control effort
        control_cost = (actions ** 2).sum(dim=-1).mean(dim=-1).sum(dim=-1)
        
        # Terminal cost (final state error)
        final_error = trajectory[:, -1] - target_state
        terminal_cost = (final_error ** 2).sum(dim=-1)
        
        total_cost = (
            self.position_weight * pos_cost +
            self.velocity_weight * vel_cost +
            self.attitude_weight * att_cost +
            self.control_weight * control_cost +
            self.terminal_weight * terminal_cost
        )
        
        return total_cost
    
    def forward(
        self,
        current_state: torch.Tensor,
        target_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Plan optimal trajectory.
        
        Args:
            current_state: Current state [batch, state_dim]
            target_state: Target state [batch, state_dim]
            
        Returns:
            trajectory: Planned states [batch, horizon+1, state_dim]
            actions: Optimal actions [batch, horizon, action_dim]
            cost: Trajectory cost [batch]
        """
        # Get initial action sequence from optimizer
        actions, _ = self.optimizer(current_state, target_state)
        
        # Roll out trajectory using dynamics model
        trajectory = self.dynamics.rollout(current_state, actions, self.dt)
        
        # Compute actual cost
        cost = self.compute_cost(trajectory, actions, target_state)
        
        return trajectory, actions, cost
    
    def plan(
        self,
        current_state: np.ndarray,
        target_state: np.ndarray,
        device: str = "cpu",
    ) -> TrajectoryPlan:
        """
        Generate trajectory plan.
        
        Args:
            current_state: Current state vector
            target_state: Target state vector
            device: Compute device
            
        Returns:
            TrajectoryPlan dataclass
        """
        self.eval()
        
        with torch.no_grad():
            current = torch.FloatTensor(current_state).unsqueeze(0).to(device)
            target = torch.FloatTensor(target_state).unsqueeze(0).to(device)
            
            trajectory, actions, cost = self.forward(current, target)
        
        traj_np = trajectory.cpu().numpy().squeeze()
        actions_np = actions.cpu().numpy().squeeze()
        
        # Check feasibility (simple constraint checks)
        max_thrust = np.abs(actions_np[:, :3]).max()
        max_torque = np.abs(actions_np[:, 3:]).max()
        feasible = bool(max_thrust < 200 and max_torque < 20)
        
        return TrajectoryPlan(
            positions=traj_np[:, :3],
            velocities=traj_np[:, 3:6],
            thrust_commands=actions_np[:, :3],
            torque_commands=actions_np[:, 3:6],
            predicted_cost=cost.item(),
            feasible=feasible,
            horizon_time=self.horizon * self.dt,
        )
    
    def get_control(
        self,
        current_state: np.ndarray,
        target_state: np.ndarray,
        device: str = "cpu",
    ) -> np.ndarray:
        """
        Get first control action (for receding horizon).
        
        Args:
            current_state: Current state
            target_state: Target state
            device: Compute device
            
        Returns:
            Control action [action_dim]
        """
        plan = self.plan(current_state, target_state, device)
        return np.concatenate([plan.thrust_commands[0], plan.torque_commands[0]])


class PrecisionManeuveringModule(BaseModule):
    """Complete precision maneuvering module using Neural MPC."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        super().__init__(
            name="precision_maneuvering",
            config=config,
            device=device,
        )
        
        self.mpc = None
        self.state_dim = config.get("state_dim", 13)
        self.action_dim = config.get("action_dim", 6)
        self.horizon = config.get("horizon", 20)
        self.dt = config.get("dt", 0.1)
        
        # Current plan (for warm starting)
        self.current_plan: Optional[TrajectoryPlan] = None
    
    def _build_model(self) -> nn.Module:
        self.mpc = NeuralMPC(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            horizon=self.horizon,
            hidden_dim=self.config.get("hidden_dim", 128),
            dt=self.dt,
        ).to(self.device)
        
        return self.mpc.dynamics
    
    def _preprocess(self, inputs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Build state vectors from inputs."""
        # Current state
        position = np.asarray(inputs.get("position", [7000, 0, 0]))
        velocity = np.asarray(inputs.get("velocity", [0, 7.5, 0]))
        attitude = np.asarray(inputs.get("attitude", [1, 0, 0, 0]))
        angular_velocity = np.asarray(inputs.get("angular_velocity", [0, 0, 0]))
        
        current_state = np.concatenate([position, velocity, attitude, angular_velocity])
        
        # Target state
        target_position = np.asarray(inputs.get("target_position", position))
        target_velocity = np.asarray(inputs.get("target_velocity", velocity))
        target_attitude = np.asarray(inputs.get("target_attitude", [1, 0, 0, 0]))
        target_angular_velocity = np.asarray(inputs.get("target_angular_velocity", [0, 0, 0]))
        
        target_state = np.concatenate([
            target_position, target_velocity, target_attitude, target_angular_velocity
        ])
        
        return current_state, target_state
    
    def _postprocess(self, outputs: TrajectoryPlan) -> Dict[str, Any]:
        return {
            "thrust_command": outputs.thrust_commands[0].tolist(),
            "torque_command": outputs.torque_commands[0].tolist(),
            "planned_trajectory": outputs.positions.tolist(),
            "planned_velocities": outputs.velocities.tolist(),
            "predicted_cost": outputs.predicted_cost,
            "feasible": outputs.feasible,
            "horizon_time_s": outputs.horizon_time,
        }
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute optimal control for precision maneuvering.
        
        Args:
            inputs: Dictionary with current and target states
            
        Returns:
            Control commands and trajectory plan
        """
        current_state, target_state = self._preprocess(inputs)
        
        plan = self.mpc.plan(current_state, target_state, self.device)
        self.current_plan = plan
        
        return self._postprocess(plan)
    
    def _fallback(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Simple proportional controller fallback."""
        current_state, target_state = self._preprocess(inputs)
        
        # Simple P-control
        position_error = target_state[:3] - current_state[:3]
        velocity_error = target_state[3:6] - current_state[3:6]
        
        # Proportional gains
        kp_pos = 0.1
        kp_vel = 1.0
        
        # Desired velocity towards target
        desired_vel = kp_pos * position_error
        
        # Thrust proportional to velocity error
        thrust = kp_vel * (desired_vel - current_state[3:6])
        
        # Clip to max thrust
        thrust = np.clip(thrust, -100, 100)
        
        # Simple attitude control (point towards target)
        torque = -0.5 * current_state[10:13]  # Damping only
        
        return {
            "thrust_command": thrust.tolist(),
            "torque_command": torque.tolist(),
            "planned_trajectory": None,
            "feasible": True,
            "fallback": True,
        }

