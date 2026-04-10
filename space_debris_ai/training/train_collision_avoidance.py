"""
Training script for Collision Avoidance module (SAC agent).
"""

import argparse
import os
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from space_debris_ai.simulation import OrbitalEnv, EnvConfig
from space_debris_ai.core.config import SystemConfig, TrainingConfig


def make_env(config: EnvConfig, rank: int = 0, seed: int = 0):
    """Create a single environment."""
    def _init():
        env = OrbitalEnv(config)
        env = Monitor(env, filename=None, allow_early_resets=True)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_collision_avoidance(
    total_timesteps: int = 1_000_000,
    num_envs: int = 4,
    learning_rate: float = 3e-4,
    buffer_size: int = 300_000,
    batch_size: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    device: str = "auto",
    checkpoint_dir: str = "checkpoints/collision_avoidance",
    log_dir: str = "logs/collision_avoidance",
    eval_freq: int = 50_000,
    save_freq: int = 100_000,
    seed: int = 42,
    use_virtual_sensors: bool = True,
):
    """
    Train collision avoidance agent using SAC.
    
    Args:
        total_timesteps: Total training timesteps
        num_envs: Number of parallel environments
        learning_rate: Learning rate
        buffer_size: Replay buffer size
        batch_size: Batch size
        gamma: Discount factor
        tau: Soft update coefficient
        device: Compute device
        checkpoint_dir: Checkpoint directory
        log_dir: Log directory
        eval_freq: Evaluation frequency
        save_freq: Save frequency
        seed: Random seed
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Environment config
    env_config = EnvConfig(
        num_debris=50,
        collision_distance=0.05,
        safe_distance=0.5,
        collision_penalty=-1000.0,
        safety_bonus=0.1,
        fuel_penalty=-0.01,
        use_virtual_sensors=use_virtual_sensors,
        virtual_sensor_seed=seed,
    )
    
    # Create environments
    if num_envs == 1:
        env = DummyVecEnv([make_env(env_config, seed=seed)])
    else:
        env = SubprocVecEnv([make_env(env_config, rank=i, seed=seed) for i in range(num_envs)])
    
    # Evaluation environment
    eval_env = DummyVecEnv([make_env(env_config, seed=seed + 1000)])
    
    # Create SAC agent
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        device=device,
        verbose=1,
        tensorboard_log=log_dir,
        seed=seed,
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix="sac_collision",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )
    
    # Train
    print(f"Training collision avoidance agent for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
    )
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model")
    model.save(final_path)
    print(f"Training complete! Model saved to {final_path}")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Collision Avoidance Agent")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/collision_avoidance")
    parser.add_argument("--log-dir", type=str, default="logs/collision_avoidance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-virtual-sensors", action="store_true",
                        help="Disable virtual sensors (use legacy 44-dim obs)")
    
    args = parser.parse_args()
    
    train_collision_avoidance(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        use_virtual_sensors=not args.no_virtual_sensors,
    )

