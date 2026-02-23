"""
Training script for Energy Management module (PPO agent).
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from space_debris_ai.simulation import OrbitalEnv, EnvConfig


def make_env(config: EnvConfig, rank: int = 0, seed: int = 0):
    """Create a single environment."""
    def _init():
        env = OrbitalEnv(config)
        env = Monitor(env, filename=None, allow_early_resets=True)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_energy_management(
    total_timesteps: int = 1_000_000,
    num_envs: int = 8,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    device: str = "auto",
    checkpoint_dir: str = "checkpoints/energy_management",
    log_dir: str = "logs/energy_management",
    eval_freq: int = 50_000,
    save_freq: int = 100_000,
    seed: int = 42,
):
    """
    Train energy management agent using PPO.
    
    Args:
        total_timesteps: Total training timesteps
        num_envs: Number of parallel environments
        learning_rate: Learning rate
        n_steps: Steps per update
        batch_size: Batch size
        n_epochs: Number of optimization epochs
        gamma: Discount factor
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
        num_debris=30,
        max_episode_steps=5000,
    )
    
    # Create environments
    if num_envs == 1:
        env = DummyVecEnv([make_env(env_config, seed=seed)])
    else:
        env = SubprocVecEnv([make_env(env_config, rank=i, seed=seed) for i in range(num_envs)])
    
    # Evaluation environment
    eval_env = DummyVecEnv([make_env(env_config, seed=seed + 1000)])
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        device=device,
        verbose=1,
        tensorboard_log=log_dir,
        seed=seed,
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix="ppo_energy",
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
    print(f"Training energy management agent for {total_timesteps} timesteps...")
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
    parser = argparse.ArgumentParser(description="Train Energy Management Agent")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/energy_management")
    parser.add_argument("--log-dir", type=str, default="logs/energy_management")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    train_energy_management(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        seed=args.seed,
    )

