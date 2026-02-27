# A.R.I.E.S — Advanced Retrieval & In-Orbit Elimination System

<p align="center">
  <strong>Autonomous AI system for orbital debris collection</strong><br>
  Deep learning · Reinforcement learning · Sensor fusion · Orbital mechanics
</p>

<p align="center">
  <a href="README.en.md">English</a> · <a href="README.ru.md">Русский</a>
</p>

---

## About

A.R.I.E.S is an autonomous spacecraft control system designed for active space debris removal.
The system combines reinforcement learning agents, neural network modules, physics-informed simulation,
and a real-time web dashboard — built from the ground up in Python.

**Live demo:** deployed on [Render](https://render.com) via Flask + Gunicorn.

> **Note:** the demo runs on Render's free tier. After ~15 minutes of inactivity the service goes to sleep. The first request after that may take 30–60 seconds to wake it up.

---

## Key Features

| Feature | Description |
|---|---|
| **4-level priority architecture** | Survival → Safety → Mission Critical → Mission Execution |
| **Physics-informed simulation** | Keplerian orbital mechanics with J2 perturbation, solar pressure, atmospheric drag |
| **Multi-sensor fusion** | GPS, IMU, Star Tracker with weighted fusion |
| **RL agents** | SAC (collision avoidance, manipulator), PPO (energy management) |
| **Neural modules** | CNN collision detector, LSTM autoencoder (anomaly), TCN (state prediction), TFT (failure prediction), EfficientNet (debris recognition), DETR tracker |
| **Fail-safe system** | Classical algorithm fallbacks, watchdog timers, automatic mode degradation |
| **Web dashboard** | AetherOS-style dark UI: 3D orbit view, telemetry, debris tracking, danger alerts |
| **Gymnasium environment** | Full RL-compatible orbital environment for training |

---

## Architecture

```
space_debris_ai/
├── core/                              # Base infrastructure
│   ├── config.py                      # Pydantic configuration & validation
│   ├── base_module.py                 # Abstract AI module class (PyTorch)
│   └── message_bus.py                 # Inter-module pub/sub messaging
│
├── models/                            # Neural network modules (by priority)
│   ├── level1_survival/               # < 100ms · 99.999% reliability
│   │   ├── collision_avoidance/       # PointNet + CNN + SAC agent
│   │   └── navigation/               # EKF + neural correction
│   │
│   ├── level2_safety/                 # < 500ms · 99.99% reliability
│   │   ├── anomaly_detection/         # LSTM autoencoder + classifier
│   │   └── energy_management/         # PPO power distribution
│   │
│   ├── level3_mission_critical/       # < 1s · 99.9% reliability
│   │   ├── state_prediction/          # TCN + physics-informed loss
│   │   ├── early_warning/             # Attention-based alert system
│   │   ├── sensor_filter/             # Denoising autoencoder
│   │   └── failure_prediction/        # Temporal Fusion Transformer (RUL)
│   │
│   └── level4_mission_execution/      # < 2s · 99% reliability
│       ├── debris_recognition/        # EfficientNet multimodal classifier
│       ├── manipulator_control/       # SAC robotic arm control
│       ├── object_tracking/           # DETR-like multi-object tracker
│       ├── precision_maneuvering/     # MPC trajectory controller
│       └── risk_assessment/           # Mission risk assessor
│
├── sensors/                           # Sensor interfaces
│   ├── imu.py, lidar.py, camera.py
│   └── fusion.py                      # Multi-sensor weighted fusion
│
├── simulation/                        # Gymnasium orbital environment
│   ├── physics.py                     # Keplerian mechanics + perturbations
│   ├── environment.py                 # OrbitalEnv (Gymnasium API)
│   └── scenarios.py                   # Procedural scenario generator
│
├── safety/                            # Fail-safe mechanisms
│   ├── failsafe.py                    # Fallback modes (Normal → Emergency)
│   └── watchdog.py                    # Health monitoring & timeouts
│
├── inference/                         # Real-time inference engine
│   └── mission_controller.py          # Central orchestrator for all modules
│
├── training/                          # Training scripts & benchmarks
│   ├── train_collision_avoidance.py
│   ├── train_energy_management.py
│   ├── train_anomaly_detection.py
│   ├── train_state_prediction.py
│   ├── train_manipulator_control.py
│   └── benchmark.py                   # Latency & throughput benchmarks
│
├── visualization/                     # Dashboards
│   ├── dashboard.py                   # Matplotlib mission dashboard
│   ├── web_server.py                  # Flask web dashboard (production)
│   ├── templates/index.html           # AetherOS-style dark UI
│   └── static/                        # CSS + JS (Plotly.js 3D charts)
│
└── tests/                             # Test suite
    ├── test_basic.py
    ├── test_integration.py
    └── test_verify_integration.py
```

---

## Priority Levels

| Level | Name | Latency | Reliability | Purpose |
|:---:|---|---|---|---|
| 1 | Survival | < 100 ms | 99.999% | Collision avoidance, navigation |
| 2 | Safety | < 500 ms | 99.99% | Anomaly detection, power management |
| 3 | Mission Critical | < 1 s | 99.9% | State prediction, early warning, failure prediction |
| 4 | Mission Execution | < 2 s | 99% | Debris recognition, capture, tracking |

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/yctal20-lang/ARIES.git
cd ARIES
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

**Web dashboard only** (lightweight, no GPU):

```bash
pip install -r requirements.txt
```

**Full system** (PyTorch, training, all models):

```bash
pip install -r space_debris_ai/requirements.txt
```

### 4. Run

**Web dashboard:**

```bash
python run_web_dashboard.py
```

Open `http://127.0.0.1:5000` in your browser.

Options:

```bash
python run_web_dashboard.py --port 8080          # custom port
python run_web_dashboard.py --host 127.0.0.1     # localhost only
python run_web_dashboard.py --no-debug            # production mode
```

**Matplotlib dashboard** (saves PNG):

```bash
python run_dashboard.py --num-steps 120 --seed 42 --save-path mission.png
```

**Run tests:**

```bash
pytest space_debris_ai/tests/ -v
```

---

## Quick Start (Python API)

### Web Dashboard (no GPU required)

```bash
pip install flask numpy gymnasium gunicorn
python run_web_dashboard.py
```

Open `http://127.0.0.1:5000` — 3D orbit view, telemetry, debris tracking, danger alerts.

### Full System (requires PyTorch)

```python
from space_debris_ai import SystemConfig
from space_debris_ai.simulation import OrbitalEnv
from space_debris_ai.inference import MissionController

config = SystemConfig(mission_name="debris_collection_001")
env = OrbitalEnv()
obs, info = env.reset()

controller = MissionController(config)
controller.start()

for step in range(1000):
    result = controller.step(
        {"position": obs[:3], "velocity": obs[3:6], "attitude": obs[6:10]},
        dt=0.1,
    )
    obs, reward, done, truncated, info = env.step(result["commands"])
    if done or truncated:
        break

controller.stop()
env.close()
```

---

## Training

```bash
# Collision Avoidance (SAC)
python -m space_debris_ai.training.train_collision_avoidance --total-timesteps 1000000

# Energy Management (PPO)
python -m space_debris_ai.training.train_energy_management --total-timesteps 1000000

# Anomaly Detection (LSTM Autoencoder)
python -m space_debris_ai.training.train_anomaly_detection --num-epochs 100

# State Prediction (TCN)
python -m space_debris_ai.training.train_state_prediction --num-epochs 100

# Manipulator Control (SAC)
python -m space_debris_ai.training.train_manipulator_control --total-timesteps 1000000
```

---

## Simulation

Gymnasium-compatible orbital environment with:

- Keplerian orbital mechanics (J2 perturbation, solar radiation pressure, atmospheric drag)
- Configurable debris fields (type, size, mass, material)
- Spacecraft dynamics: thrust, fuel consumption, attitude control
- Procedural scenario generation with 8 difficulty levels

```python
from space_debris_ai.simulation import OrbitalEnv
from space_debris_ai.simulation.environment import EnvConfig

env = OrbitalEnv(config=EnvConfig(dt=0.5, num_debris=20, max_episode_steps=10000))
obs, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break
```

---

## Web Dashboard

AetherOS-style dark theme interface with real-time mission data:

- **Orbit Track** — 3D Earth + spacecraft trajectory + debris field (Plotly.js)
- **Telemetry** — position/velocity components over time
- **Fusion** — sensor confidence and spacecraft speed
- **Resources** — fuel level and debris count
- **Danger Alerts** — collision warnings, anomalies, low fuel events
- **Debris Table** — each object with size, mass, material, distance, disposal recommendation

Disposal suggestions follow ESA/NASA approaches: laser ablation, net capture, harpoon/robotic arm, rendezvous & deorbit.

---

## Deployment (Render)

The project includes `render.yaml` for one-click deployment:

| Setting | Value |
|---|---|
| **Runtime** | Python 3.11 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn space_debris_ai.visualization.web_server:app --bind 0.0.0.0:$PORT` |

---

## Testing

```bash
pytest space_debris_ai/tests/ -v
```

---

## Target Metrics

| Metric | Target |
|---|---|
| Collision avoidance | 100% |
| Debris collection accuracy | > 95% |
| Fuel efficiency vs classical | +30% |
| Autonomous operation | > 90 days |
| Capture success (first attempt) | > 85% |
| Failure prediction lead time | > 48 hours |
| Level 1 latency | < 100 ms |

---

## Tech Stack

**Core:** Python 3.11 · PyTorch 2.x · Gymnasium · Stable-Baselines3

**Models:** CNN · LSTM · TCN · Transformer (TFT) · EfficientNet · DETR · PointNet · SAC · PPO · MPC

**Simulation:** Keplerian mechanics · J2 perturbation · EKF sensor fusion

**Web:** Flask · Gunicorn · Plotly.js · AetherOS CSS

**Infra:** Render · ONNX Runtime · TensorBoard · W&B

---

## License

MIT
