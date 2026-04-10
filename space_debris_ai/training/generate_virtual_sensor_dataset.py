"""
Generate training datasets from OrbitalEnv with full virtual sensor suite.

Outputs ``.npz`` files containing:
- ``telemetry``: (num_steps, TELEMETRY_DIM) float32 array
- ``observations``: (num_steps, obs_dim) raw env observations
- ``actions``: (num_steps, 7) actions taken
- ``rewards``: (num_steps,) rewards received
- ``metadata``: dict with episode info (seed, num_debris, steps, etc.)

Usage::

    python generate_virtual_sensor_dataset.py --num-episodes 200 --output-dir data/virtual_sensor_dataset
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from space_debris_ai.simulation.environment import OrbitalEnv, EnvConfig
from space_debris_ai.sensors.virtual.hub import TELEMETRY_DIM


def generate_dataset(
    num_episodes: int = 100,
    max_steps_per_episode: int = 1000,
    num_debris: int = 30,
    output_dir: str = "data/virtual_sensor_dataset",
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)

    env_config = EnvConfig(
        use_virtual_sensors=True,
        virtual_sensor_seed=seed,
        num_debris=num_debris,
        max_episode_steps=max_steps_per_episode + 10,
    )
    env = OrbitalEnv(config=env_config)

    all_telemetry = []
    all_observations = []
    all_actions = []
    all_rewards = []

    print(f"Generating {num_episodes} episodes with virtual sensors (dim={TELEMETRY_DIM})...")
    for ep in tqdm(range(num_episodes)):
        ep_seed = seed + ep
        obs, info = env.reset(seed=ep_seed)

        ep_obs = [obs]
        ep_act = []
        ep_rew = []
        ep_telem = []

        # First step telemetry comes from reset
        sd = info.get("sensor_data", {})
        tv = sd.get("telemetry_vector", obs)
        ep_telem.append(tv)

        for step in range(max_steps_per_episode):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            ep_obs.append(obs)
            ep_act.append(action)
            ep_rew.append(reward)

            sd = info.get("sensor_data", {})
            tv = sd.get("telemetry_vector", obs)
            ep_telem.append(tv)

            if terminated or truncated:
                break

        # Save per-episode file
        ep_file = os.path.join(output_dir, f"episode_{ep:04d}.npz")
        np.savez_compressed(
            ep_file,
            telemetry=np.array(ep_telem, dtype=np.float32),
            observations=np.array(ep_obs, dtype=np.float32),
            actions=np.array(ep_act, dtype=np.float32),
            rewards=np.array(ep_rew, dtype=np.float32),
            seed=ep_seed,
            num_debris=num_debris,
            steps=len(ep_act),
        )

        all_telemetry.append(np.array(ep_telem, dtype=np.float32))
        all_observations.append(np.array(ep_obs, dtype=np.float32))
        all_actions.append(np.array(ep_act, dtype=np.float32))
        all_rewards.append(np.array(ep_rew, dtype=np.float32))

    env.close()

    # Save a combined summary file with concatenated data
    combined = os.path.join(output_dir, "combined.npz")
    np.savez_compressed(
        combined,
        telemetry_dim=TELEMETRY_DIM,
        num_episodes=num_episodes,
        episode_lengths=np.array([len(r) for r in all_rewards]),
    )

    total_steps = sum(len(r) for r in all_rewards)
    print(f"Done. {num_episodes} episodes, {total_steps} total steps.")
    print(f"Saved to: {output_dir}/")
    print(f"  Per-episode: episode_0000.npz .. episode_{num_episodes-1:04d}.npz")
    print(f"  Summary:     combined.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Virtual Sensor Dataset")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-debris", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="data/virtual_sensor_dataset")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    generate_dataset(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        num_debris=args.num_debris,
        output_dir=args.output_dir,
        seed=args.seed,
    )
