# Training utilities (`space_debris_ai.training`)

Scripts in this package **train or evaluate** models used by the Space Debris Collector stack. They assume the **Gymnasium** environment `OrbitalEnv` and (optionally) the **virtual sensor hub** for rich telemetry.

## Quick reference

| Script | What it trains | Algorithm / model | Default checkpoints |
| ------ | -------------- | ----------------- | --------------------- |
| `train_collision_avoidance.py` | Avoid debris, stay safe | **SAC** (stable-baselines3) | `checkpoints/collision_avoidance/` |
| `train_energy_management.py` | Power / resource policy | **PPO** | `checkpoints/energy_management/` |
| `train_manipulator_control.py` | Arm / gripper control | **SAC** | `checkpoints/manipulator_control/` |
| `train_anomaly_detection.py` | Normal telemetry manifold | **LSTM autoencoder** (PyTorch) | `checkpoints/anomaly_detection/` |
| `train_state_prediction.py` | Future states from history | **TCN** + physics-informed loss | `checkpoints/state_prediction/` |
| `generate_virtual_sensor_dataset.py` | Offline `.npz` datasets | No training—data generation | Your `--output-dir` |
| `benchmark.py` | — | Profiles inference latency | JSON output path |

## Reinforcement learning (SAC / PPO)

- Environments are wrapped with `Monitor`; you can run **multiple parallel envs** (`--num-envs`) for throughput.
- Logs go under `logs/<module>/` (TensorBoard-compatible where SB3 writes events).
- Checkpoints are saved on a schedule (`--save-freq`, `--eval-freq` where applicable).

Example (collision avoidance):

```bash
python -m space_debris_ai.training.train_collision_avoidance \
  --total-timesteps 1000000 \
  --num-envs 4 \
  --checkpoint-dir checkpoints/collision_avoidance
```

## Supervised / self-supervised (PyTorch)

- **Anomaly detection:** sequences of telemetry; the model **reconstructs** the input. Large reconstruction error on running data suggests a fault or regime shift.
- **State prediction:** sequences are sliced into input windows and future targets; the **TCN** is trained with additional physics-aware terms where implemented.

Virtual sensors: when enabled, feature dimension matches `TELEMETRY_DIM` from `space_debris_ai.sensors.virtual.hub`.

## Virtual sensor dataset generation

Produces `.npz` files with arrays such as `telemetry`, `observations`, `actions`, `rewards`, plus `metadata` for reproducibility.

```bash
python -m space_debris_ai.training.generate_virtual_sensor_dataset \
  --num-episodes 200 \
  --output-dir data/virtual_sensor_dataset
```

Use this when you want **fixed offline data** for custom experiments or faster iteration without rolling the full RL loop.

## Benchmarking

Measures **latency percentiles** and **throughput** for registered modules (see `benchmark.py` CLI and `benchmark_all_modules`).

```bash
python -m space_debris_ai.training.benchmark --num-runs 1000 --output benchmark_results.json
```

## Lazy imports

`from space_debris_ai.training import train_collision_avoidance` loads the corresponding submodule on demand (see `training/__init__.py`).

## Further reading

- Main project overview and **ML vs LLM** clarification: **[`../README.md`](../README.md)** — section *Machine learning and neural networks in this project*.
- Model implementations live under `space_debris_ai/models/` by priority level (survival → safety → mission → execution).
