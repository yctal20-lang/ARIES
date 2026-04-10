"""
Background OrbitalEnv + VirtualSensorHub loop for live API/SSE (no per-request sim restart).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from space_debris_ai.simulation.environment import EnvConfig, OrbitalEnv
from space_debris_ai.sensors.virtual.hub import VirtualSensorHub

from .virtual_sensor_serialization import serialize_virtual_sensor_data

# Match dashboard run_simulation() dynamics
_DEFAULT_DT = 0.5
_DEFAULT_NUM_DEBRIS = 20
_DEFAULT_MAX_STEPS = 500_000
# Wall-clock pacing between env steps (~10 Hz)
_DEFAULT_STEP_INTERVAL_S = 0.1

_runner: Optional["VirtualSensorRunner"] = None
_runner_lock = threading.Lock()


class VirtualSensorRunner:
    def __init__(
        self,
        seed: int = 42,
        step_interval_s: float = _DEFAULT_STEP_INTERVAL_S,
    ):
        self._seed = int(seed)
        self._step_interval_s = float(step_interval_s)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False

        self._data_lock = threading.Lock()
        self._latest: Dict[str, Any] = {}
        self._seq = 0

        self._pending_seed_lock = threading.Lock()
        self._pending_seed: Optional[int] = None

    def request_reset(self, seed: Optional[int] = None) -> None:
        """Next loop iteration will reset env (optionally with a new seed)."""
        with self._pending_seed_lock:
            self._pending_seed = int(seed) if seed is not None else self._seed

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._run_loop, name="VirtualSensorRunner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._started = False

    def get_latest(self) -> Tuple[int, Dict[str, Any]]:
        with self._data_lock:
            return self._seq, dict(self._latest)

    def _push(
        self,
        payload: Dict[str, Any],
        sim_time: float,
        seed: int,
        fuel: float,
        env: OrbitalEnv,
    ) -> None:
        self._seq += 1
        out = dict(payload)
        sc = env.spacecraft
        out["_motion"] = {
            "speed_kmh": float(sc.speed) * 3600.0,
            "altitude_km": float(sc.altitude),
        }
        out["_meta"] = {
            "seq": self._seq,
            "sim_time": float(sim_time),
            "seed": int(seed),
            "fuel": float(fuel),
        }
        with self._data_lock:
            self._latest = out

    @staticmethod
    def _read_sensors(env: OrbitalEnv, hub: VirtualSensorHub, config: EnvConfig, thrust_mag: float) -> Dict[str, Any]:
        debris_pos = [d.position for d in env.debris_objects]
        debris_vel = [d.velocity for d in env.debris_objects]
        debris_sizes = [d.size for d in env.debris_objects]
        debris_types = [d.debris_type for d in env.debris_objects]
        t = float(env.spacecraft.time)
        return hub.read_all(
            spacecraft_position=env.spacecraft.position,
            spacecraft_velocity=env.spacecraft.velocity,
            spacecraft_attitude=env.spacecraft.attitude,
            spacecraft_angular_velocity=env.spacecraft.angular_velocity,
            debris_positions=debris_pos,
            debris_velocities=debris_vel,
            debris_sizes=debris_sizes,
            debris_types=debris_types,
            thrust_magnitude=thrust_mag,
            timestamp=t,
        )

    def _run_loop(self) -> None:
        seed = self._seed
        while not self._stop.is_set():
            with self._pending_seed_lock:
                if self._pending_seed is not None:
                    seed = self._pending_seed
                    self._seed = seed
                    self._pending_seed = None

            np.random.seed(seed)
            config = EnvConfig(
                dt=_DEFAULT_DT,
                max_episode_steps=_DEFAULT_MAX_STEPS,
                num_debris=_DEFAULT_NUM_DEBRIS,
            )
            env = OrbitalEnv(config=config)
            hub = VirtualSensorHub(seed=seed)
            env.reset(seed=seed)
            sdata0 = self._read_sensors(env, hub, config, 0.0)
            serialized0 = serialize_virtual_sensor_data(sdata0)
            self._push(
                serialized0,
                sim_time=float(env.spacecraft.time),
                seed=seed,
                fuel=float(env.spacecraft.fuel_mass),
                env=env,
            )

            while not self._stop.is_set():
                with self._pending_seed_lock:
                    if self._pending_seed is not None:
                        break

                action = np.zeros(7, dtype=np.float64)
                action[:3] = np.random.randn(3) * 0.02
                thrust_mag = float(np.linalg.norm(action[:3]) * config.max_thrust)

                _obs, _reward, term, trunc, _info = env.step(action)

                if term or trunc:
                    env.reset(seed=seed)
                    sdata_r = self._read_sensors(env, hub, config, 0.0)
                    serialized_r = serialize_virtual_sensor_data(sdata_r)
                    self._push(
                        serialized_r,
                        sim_time=float(env.spacecraft.time),
                        seed=seed,
                        fuel=float(env.spacecraft.fuel_mass),
                        env=env,
                    )
                    continue

                sdata = self._read_sensors(env, hub, config, thrust_mag)
                serialized = serialize_virtual_sensor_data(sdata)
                self._push(
                    serialized,
                    sim_time=float(env.spacecraft.time),
                    seed=seed,
                    fuel=float(env.spacecraft.fuel_mass),
                    env=env,
                )

                time.sleep(self._step_interval_s)


def ensure_virtual_sensor_runner(seed: int = 42) -> VirtualSensorRunner:
    """Singleton runner (one background sim per process)."""
    global _runner
    with _runner_lock:
        if _runner is None:
            _runner = VirtualSensorRunner(seed=seed)
            _runner.start()
        return _runner
