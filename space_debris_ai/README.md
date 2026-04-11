# Space Debris Collector AI System

Autonomous AI system for orbital debris collection using deep learning, reinforcement learning, and advanced sensor fusion.

## Overview

This system provides a complete neural network architecture for controlling a space debris collection spacecraft. It includes modules for:

- **Collision Avoidance** - CNN + SAC-based reactive avoidance
- **Navigation** - EKF sensor fusion with neural network correction
- **Energy Management** - PPO-based power distribution
- **Anomaly Detection** - LSTM autoencoder for telemetry analysis
- **State Prediction** - TCN with physics-informed constraints
- **Early Warning** - Attention-based alert system
- **Debris Recognition** - Multi-modal classification (visual, radar, lidar)
- **Manipulator Control** - SAC-based robotic arm control
- **Object Tracking** - DETR-like multi-object tracker

## Machine learning and neural networks in this project

This section clarifies **what kind of AI** the repository implements and how the pieces fit together.

### Terminology

- **Machine learning (ML)** is the broad idea: systems that improve from **data** or from **interaction** with an environment (here, mostly the orbital simulator `OrbitalEnv`).
- **Neural networks** are the main **learnable models** used inside that ML stack (deep learning): layers of parameters trained by gradient-based optimization.
- **Reinforcement learning (RL)** is a family of ML methods where an **agent** learns a **policy** (what action to take) from **rewards** collected in the environment. In this project, **SAC** and **PPO** are RL algorithms; they typically use **neural networks** as function approximators for the policy and value/critic.
- **This is not an LLM (large language model).** There is no text corpus, tokenizer, or vocabulary. Nothing here predicts the next **word token** or generates natural-language answers. If you see **attention** or **Transformer** in the code, it is applied to **numeric sequences** (telemetry), **detection features** (tracking), or similar—not to language.

So: the project uses **ML**, and most trainable parts are **neural networks**, plus **classical** fusion and physics where noted below.

### Map of subsystems to methods

| Subsystem | Role | ML paradigm | Typical learnable part |
| --------- | ---- | ----------- | ---------------------- |
| Collision avoidance | Reactive safe motion | **RL** | **SAC** + policy/critic networks; CNN-style perception where used |
| Energy management | Power allocation | **RL** | **PPO** + policy/value networks |
| Manipulator control | Arm / gripper commands | **RL** | **SAC** + policy/critic networks |
| Anomaly detection | Telemetry health | **Self-supervised** | **LSTM autoencoder** (reconstruct normal sequences; high error ⇒ anomaly) |
| State prediction | Short-horizon dynamics | **Supervised** (+ constraints) | **TCN** with **physics-informed** loss |
| Early warning | Critical-event risk from sequences | **Supervised** | **Multi-head self-attention** over sequence embeddings |
| Failure prediction (TFT) | Remaining useful life / degradation | **Supervised** (time series) | **Temporal Fusion Transformer** on tabular/series features—not language |
| Sensor filter | Denoise / reconstruct signals | **Self-supervised** | **Denoising autoencoder** |
| Debris recognition | Classify debris type / attributes | **Supervised** | **CNN / multimodal encoders** (vision + radar-style features) |
| Object tracking | Associate detections over time | **Supervised** (conceptually) | **Transformer encoder–decoder** (DETR-style) on detection features |
| Navigation correction | Refine fused state | **Supervised** | **MLP + attention** over measurement channels (numeric, not text) |

### Classical (non-neural) pieces

These are still central to the system but are **not** “trained neural nets” in the usual sense:

- **Extended Kalman Filter (EKF)** and related fusion in navigation.
- **Orbital mechanics** and environment dynamics in `simulation/`.
- **Safety** layers: watchdogs, fail-safes, timeouts.

They complement learning: physics and filters provide structure; neural modules handle patterns that are hard to hand-code.

### Data flow for training (high level)

1. **RL scripts** roll out episodes in `OrbitalEnv`, optimize reward, save checkpoints under `checkpoints/…`.
2. **Sequence models** (LSTM-AE, TCN) consume **telemetry or state trajectories**, often generated in the same simulator; optional **virtual sensor** streams increase feature dimension and realism for training.
3. **Benchmark** utilities measure **inference latency and throughput** of modules, not language quality.

For concrete commands and paths, see **[`training/README.md`](training/README.md)** and the [Training](#training) section below.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from space_debris_ai import Config, SystemConfig
from space_debris_ai.simulation import OrbitalEnv
from space_debris_ai.inference import MissionController

# Create configuration
config = SystemConfig(mission_name="debris_collection_001")

# Create simulation environment
env = OrbitalEnv()
obs, info = env.reset()

# Create mission controller
controller = MissionController(config)
controller.start()

# Run mission
for step in range(1000):
    sensor_data = {
        "position": obs[:3],
        "velocity": obs[3:6],
        "attitude": obs[6:10],
    }
    
    result = controller.step(sensor_data, dt=0.1)
    action = result["commands"]
    
    obs, reward, done, truncated, info = env.step(action)
    
    if done or truncated:
        break

controller.stop()
env.close()
```

## Visualization

Mission dashboard (3D orbit, trajectory, sensor fusion, resources):

```bash
cd space_debris_ai
python -m space_debris_ai.visualization.dashboard
```

Optionally save the figure instead of showing a window:

```python
from space_debris_ai.visualization import run_dashboard
run_dashboard(num_steps=120, seed=42, show=False, save_path="mission_dashboard.png")
```

## Project Structure

```text
space_debris_ai/
├── core/                              # 🔧 Base Infrastructure
│   ├── config.py                      # Configuration management
│   ├── base_module.py                 # Abstract module class
│   └── message_bus.py                 # Inter-module communication
│
├── models/                            # 🧠 Neural Network Modules (by priority)
│   │
│   ├── level1_survival/               # ⚠️ CRITICAL: Survival Systems
│   │   ├── collision_avoidance/       # CNN + SAC collision avoidance
│   │   └── navigation/                # EKF + NN navigation
│   │
│   ├── level2_safety/                 # 🛡️ HIGH: Safety Systems  
│   │   ├── anomaly_detection/         # LSTM autoencoder
│   │   └── energy_management/         # PPO power management
│   │
│   ├── level3_mission_critical/       # 📊 MEDIUM-HIGH: Mission Critical
│   │   ├── state_prediction/          # TCN + Physics-Informed
│   │   ├── early_warning/             # Attention warning system
│   │   ├── sensor_filter/             # Denoising autoencoder
│   │   └── failure_prediction/        # TFT RUL estimation
│   │
│   └── level4_mission_execution/      # 🎯 MEDIUM: Mission Execution
│       ├── debris_recognition/        # EfficientNet multimodal
│       ├── manipulator_control/       # SAC robotic arm
│       └── object_tracking/           # DETR-like tracker
│
├── sensors/                           # 📡 Sensor Interfaces
│   ├── imu.py, lidar.py, camera.py
│   └── fusion.py                      # Multi-sensor fusion
│
├── simulation/                        # 🌍 Simulation Environment
│   ├── physics.py                     # Orbital mechanics
│   ├── environment.py                 # Gymnasium environment
│   └── scenarios.py                   # Training scenarios
│
├── safety/                            # 🛡️ Safety Systems
│   ├── failsafe.py                    # Fail-safe mechanisms
│   └── watchdog.py                    # Health monitoring
│
├── training/                          # 🏋️ Training Utilities
├── inference/                         # ⚡ Real-time Inference
│   └── mission_controller.py          # Central controller
├── visualization/                     # 📊 Mission dashboard
│   └── dashboard.py                   # 3D orbit, trajectory, fusion metrics
└── tests/                             # ✅ Test Suite
```

### Priority Levels Explained

| Level | Name | Response Time | Reliability | Description |
| ----- | ---- | ------------- | ----------- | ----------- |
| 1 | Survival | < 100ms | 99.999% | Критичные для выживания аппарата |
| 2 | Safety | < 500ms | 99.99% | Обеспечение безопасности |
| 3 | Mission Critical | < 1s | 99.9% | Критичные для миссии |
| 4 | Mission Execution | < 2s | 99% | Выполнение миссии |

## Key Features

### Safety First

- Multi-level fail-safe mechanisms
- Classical algorithm fallbacks
- Watchdog timers for all modules
- Automatic mode degradation

### Physics-Informed Learning

- Orbital mechanics constraints in loss functions
- Physically plausible predictions
- Energy conservation checks

### Multi-Modal Sensing

- Visual (camera), radar, and lidar fusion
- Cross-modal attention
- Adaptive sensor filtering

### Real-Time Performance

- Target latency < 100ms for critical systems
- Optimized inference with ONNX support
- Efficient attention mechanisms

## Training

Train individual modules using the provided training scripts:

### Reinforcement Learning Agents

**Collision Avoidance (SAC):**

```bash
python -m space_debris_ai.training.train_collision_avoidance \
    --total-timesteps 1000000 \
    --num-envs 8 \
    --checkpoint-dir checkpoints/collision_avoidance
```

**Energy Management (PPO):**

```bash
python -m space_debris_ai.training.train_energy_management \
    --total-timesteps 1000000 \
    --num-envs 8 \
    --checkpoint-dir checkpoints/energy_management
```

**Manipulator Control (SAC):**

```bash
python -m space_debris_ai.training.train_manipulator_control \
    --total-timesteps 1000000 \
    --num-envs 8 \
    --checkpoint-dir checkpoints/manipulator_control
```

### Supervised Learning Models

**Anomaly Detection (LSTM Autoencoder):**

```bash
python -m space_debris_ai.training.train_anomaly_detection \
    --num-epochs 100 \
    --batch-size 32 \
    --checkpoint-dir checkpoints/anomaly_detection
```

**State Prediction (TCN):**

```bash
python -m space_debris_ai.training.train_state_prediction \
    --num-epochs 100 \
    --batch-size 32 \
    --checkpoint-dir checkpoints/state_prediction
```

### Programmatic Training

You can also train modules programmatically:

```python
from space_debris_ai.training import (
    train_collision_avoidance,
    train_energy_management,
    train_anomaly_detection,
)

# Train collision avoidance
train_collision_avoidance(
    total_timesteps=1_000_000,
    num_envs=8,
    checkpoint_dir="checkpoints/collision_avoidance",
)
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

### Test Coverage

- **Unit Tests** (`test_basic.py`): Individual module functionality
- **Integration Tests** (`test_integration.py`): Module integration and end-to-end workflows
- **Verification** (`test_verify_integration.py`): Comprehensive integration verification

Run integration verification:

```bash
python -m space_debris_ai.tests.test_verify_integration
```

## Configuration

Edit configuration via Python:

```python
from space_debris_ai.core.config import SystemConfig, MissionMode

config = SystemConfig(
    mission_mode=MissionMode.AGGRESSIVE,
    spacecraft=SpacecraftParameters(mass=600.0),
    safety=SafetyConfig(collision_warning_time=60.0),
)
config.save("config.json")
```

## Benchmarking

Benchmark system performance and latency:

```bash
python -m space_debris_ai.training.benchmark \
    --num-runs 1000 \
    --output benchmark_results.json
```

Or programmatically:

```python
from space_debris_ai.training.benchmark import benchmark_all_modules

results = benchmark_all_modules(
    output_path="benchmark_results.json",
    num_runs=1000,
    device="auto",
)
```

The benchmark measures:

- Average, P50, P95, P99 latency
- Throughput (Hz)
- Memory usage (CUDA)
- Error rates

## Metrics

Target performance:

- Debris collection accuracy: >95%
- Collision avoidance: 100%
- Fuel efficiency: +30% vs classical
- Autonomous operation: >90 days
- Capture success rate: >85% first attempt
- Failure prediction: >48h lead time, >80% accuracy
- Latency: <100ms for Level 1 (survival) modules

## Кратко по-русски: методы и термины

Этот раздел дублирует смысл секции *Machine learning and neural networks in this project* для русскоязычных читателей.

- **Машинное обучение (ML)** здесь — модели, которые улучшаются по **данным** или по **опыту в симуляторе** (`OrbitalEnv`).
- **Нейронные сети** — основной тип обучаемых моделей (глубокое обучение): политики, критики, автоэнкодеры, TCN и т.д.
- **Обучение с подкреплением (RL):** **SAC** и **PPO** учат **политику** (какие действия предпринимать) из **награды** в среде; внутри почти всегда стоят **нейросети** для политики и оценки ценности.
- **Это не LLM:** нет работы с текстом, токенизатора и «следующего слова». **Attention** и **Transformer** в репозитории относятся к **числовым последовательностям** (телеметрия) или **признакам детекций** (трекинг), а не к естественному языку.
- **Классика рядом с ML:** EKF и слияние измерений, орбитальная механика, watchdog и fail-safe — **не** глубокие сети, они задают физику и надёжность.

**Скрипты обучения** и генерации датасетов описаны в [`training/README.md`](training/README.md).

## License

MIT License

## Citation

If you use this system in your research, please cite:

```bibtex
@software{space_debris_ai,
  title={Space Debris Collector AI System},
  year={2026},
  url={https://github.com/space-debris-ai}
}
```
