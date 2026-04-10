"""
A.R.I.E.S Web Dashboard Server.
Flask server for AetherOS-style web interface.
"""

from flask import Flask, render_template, jsonify, request, Response, stream_with_context
from flask.json.provider import DefaultJSONProvider
import numpy as np
import sys
from pathlib import Path
import json
import time

# Add parent to path
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from space_debris_ai.simulation.environment import OrbitalEnv, EnvConfig
from space_debris_ai.sensors.fusion import SensorFusion
from space_debris_ai.sensors.virtual.hub import VirtualSensorHub
from .virtual_sensor_serialization import serialize_virtual_sensor_data as _serialize_sensor_data
from .virtual_sensor_runner import ensure_virtual_sensor_runner


def _dlog(msg, data=None, hypothesis_id=None):
    try:
        with open("debug-17e329.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "17e329", "message": msg, "data": data or {}, "hypothesisId": hypothesis_id, "timestamp": int(time.time() * 1000), "location": "web_server.py"}) + "\n")
    except Exception:
        pass


def _dlog_937(msg, data=None, hypothesis_id=None, run_id="start-btn-issue"):
    # #region agent log
    try:
        with open("debug-937bb1.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "937bb1",
                "runId": run_id,
                "message": msg,
                "data": data or {},
                "hypothesisId": hypothesis_id,
                "timestamp": int(time.time() * 1000),
                "location": "web_server.py",
            }) + "\n")
    except Exception:
        pass
    # #endregion


def _json_safe(value):
    """Recursively convert numpy values to JSON-serializable Python types."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    # NumPy 2: numpy.bool reports __name__ 'bool' but is not builtins.bool; belt-and-suspenders
    if getattr(type(value), "__module__", "") == "numpy" and hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return value


def _mission_data_response_body(data: dict) -> str:
    """Encode mission payload; stdlib json skips DefaultJSONProvider for numpy.bool — use explicit default."""

    def _default(o):
        if o is None or isinstance(o, (str, int, float, bool)):
            return o
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        _ot = type(o)
        if getattr(_ot, "__module__", "") == "numpy" and _ot.__name__ == "bool":
            return bool(o)
        if getattr(type(o), "__module__", "") == "numpy" and hasattr(o, "item"):
            return o.item()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    return json.dumps(data, default=_default, ensure_ascii=False, allow_nan=False)


app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)


class NumpyJSONProvider(DefaultJSONProvider):
    """Make Flask jsonify robust for numpy scalar/array values."""
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        # NumPy 2 reports type name 'bool' for numpy.bool; belt-and-suspenders for jsonify()
        _t = type(o)
        if getattr(_t, "__module__", "") == "numpy" and _t.__name__ == "bool":
            return bool(o)
        return super().default(o)


app.json = NumpyJSONProvider(app)


@app.before_request
def _trace_request_937():
    # #region agent log
    _dlog_937("http_request", {"method": request.method, "path": request.path}, "H5")
    # #endregion


def _static_cache_version() -> str:
    """Bust browser/CDN caches after deploy (Render sets RENDER_GIT_COMMIT)."""
    import os

    commit = (os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("GIT_COMMIT") or "").strip()
    if commit:
        return commit[:12]
    try:
        dash = Path(__file__).parent / "static" / "js" / "dashboard.js"
        return str(int(dash.stat().st_mtime))
    except OSError:
        return "dev"


@app.context_processor
def _inject_static_cache_version():
    return {"static_v": _static_cache_version()}


try:
    from space_debris_ai.arduino_bridge.routes import arduino_bp

    app.register_blueprint(arduino_bp)
except ImportError:
    pass

def run_simulation(num_steps: int = 150, seed: int = 42) -> dict:
    """
    Симуляция для дашборда: без PyTorch, быстрый старт.
    Опасности считаются по расстоянию до мусора, топливу и простым правилам.
    """
    # #region agent log
    _dlog("run_simulation_start", {"num_steps": num_steps, "seed": seed}, "H3")
    # #endregion
    np.random.seed(seed)
    config = EnvConfig(
        dt=0.5,
        max_episode_steps=num_steps + 10,
        num_debris=20,
    )
    env = OrbitalEnv(config=config)
    fusion = SensorFusion()
    sensor_hub = VirtualSensorHub(seed=seed)
    obs, info = env.reset(seed=seed)

    data = {
        "times": [0.0],
        "positions": [env.spacecraft.position.tolist()],
        "velocities": [env.spacecraft.velocity.tolist()],
        "confidences": [1.0],
        "fuels": [float(env.spacecraft.fuel_mass)],
        "debris_counts": [len(env.debris_objects)],
        "debris_positions": [],
        "danger_levels": [0.0],
        "collision_warnings": [],
        "anomaly_detections": [],
        "low_fuel_warnings": [],
        "spacecraft_status": {
            "altitude": float(env.spacecraft.altitude),
            "speed": float(env.spacecraft.speed),
            "mass": float(env.spacecraft.mass),
        },
    }

    sdata = None
    action = None
    for step in range(num_steps - 1):
        action = np.zeros(7)
        action[:3] = np.random.randn(3) * 0.02
        obs, reward, term, trunc, info = env.step(action)
        t = env.spacecraft.time
        data["times"].append(float(t))
        data["positions"].append(env.spacecraft.position.tolist())
        data["velocities"].append(env.spacecraft.velocity.tolist())
        data["fuels"].append(float(env.spacecraft.fuel_mass))
        data["debris_counts"].append(len(env.debris_objects))

        debris_pos = [d.position for d in env.debris_objects]
        debris_vel = [d.velocity for d in env.debris_objects]
        debris_sizes = [d.size for d in env.debris_objects]
        debris_types = [d.debris_type for d in env.debris_objects]
        thrust_mag = float(np.linalg.norm(action[:3]) * config.max_thrust) if action is not None else 0.0

        sdata = sensor_hub.read_all(
            spacecraft_position=env.spacecraft.position,
            spacecraft_velocity=env.spacecraft.velocity,
            spacecraft_attitude=env.spacecraft.attitude,
            spacecraft_angular_velocity=env.spacecraft.angular_velocity,
            debris_positions=debris_pos,
            debris_velocities=debris_vel,
            debris_sizes=debris_sizes,
            debris_types=debris_types,
            thrust_magnitude=thrust_mag,
            timestamp=float(t),
        )

        fused = fusion.fuse(
            gps_data=sdata["gps_data"],
            imu_data=sdata["imu_data"],
            star_tracker_data=sdata["star_tracker_data"],
            magnetometer_data={"magnetic_field": sdata["magnetometer"].magnetic_field},
            sun_sensor_data={"sun_vector": sdata["sun_sensor"].sun_vector,
                             "sun_visible": sdata["sun_sensor"].sun_visible},
        )
        data["confidences"].append(float(fused.confidence))

        danger_level = 0.0
        sc_pos = env.spacecraft.position

        # 1) Близкий мусор — столкновение
        for d in env.debris_objects:
            dist = np.linalg.norm(d.position - sc_pos)
            if dist < 0.15:
                prob = max(0, 1.0 - dist / 0.15)
                data["collision_warnings"].append({
                    "time": float(t),
                    "probability": float(prob),
                    "severity": "HIGH" if dist < 0.05 else "MEDIUM",
                })
                danger_level = max(danger_level, prob)
            elif dist < 0.5 and step % 20 == 0:
                data["collision_warnings"].append({
                    "time": float(t),
                    "probability": 0.3,
                    "severity": "MEDIUM",
                })
                danger_level = max(danger_level, 0.3)

        # 2) Низкое топливо
        fuel_ratio = env.spacecraft.fuel_mass / config.fuel_mass if config.fuel_mass > 0 else 1.0
        if step > num_steps * 0.5:
            fuel_ratio = max(0.0, fuel_ratio - 0.002 * (step - num_steps * 0.5))
        if fuel_ratio < 0.2:
            data["low_fuel_warnings"].append({
                "time": float(t),
                "fuel_remaining": float(env.spacecraft.fuel_mass * fuel_ratio),
                "fuel_ratio": float(fuel_ratio),
            })
            danger_level = max(danger_level, 0.5)

        data["danger_levels"].append(float(danger_level))
        if term or trunc:
            break

    # Store latest virtual sensor snapshot for API
    if sdata is not None:
        data["virtual_sensors"] = _serialize_sensor_data(sdata)

    data["debris_positions"] = [d.position.tolist() for d in env.debris_objects]
    sc_pos = env.spacecraft.position
    sc_vel = env.spacecraft.velocity

    # Debris array: position, velocity (for trajectory), size, mass, type, material (ESA/NASA typical)
    REFERENCE_SIZE_M = 5.0  # Reference spacecraft size (m) for UI comparison
    MATERIAL_BY_TYPE = {
        "fragment": ["Aluminum", "Steel", "Titanium", "Copper"],
        "rocket_body": ["Aluminum", "Aluminum alloy"],
        "satellite": ["Aluminum", "Titanium", "Carbon composite"],
        "tool": ["Steel", "Titanium"],
        "panel": ["Kapton", "Solar cell", "MLI"],
    }
    debris_list = []
    for d in env.debris_objects:
        dist_km = float(np.linalg.norm(d.position - sc_pos))
        rel_vel = float(np.linalg.norm(d.velocity - sc_vel))
        choices = MATERIAL_BY_TYPE.get(d.debris_type, ["Aluminum", "Steel"])
        material = str(np.random.choice(choices))
        debris_list.append({
            "position": d.position.tolist(),
            "velocity": d.velocity.tolist(),
            "size": float(d.size),
            "mass": float(d.mass),
            "debris_type": d.debris_type,
            "object_id": d.object_id,
            "distance_km": dist_km,
            "relative_velocity_km_s": rel_vel,
            "material": material,
        })
    # Два демо-объекта рядом с кораблём: один маленький (выброс в атмосферу/сгорание), один крупный (обход)
    sc_pos = np.asarray(sc_pos)
    sc_vel = np.asarray(sc_vel)
    u = sc_pos / (np.linalg.norm(sc_pos) + 1e-9)
    perp = np.array([-u[1], u[0], 0.0])
    perp = perp / (np.linalg.norm(perp) + 1e-9)
    for idx, (label, size, mass, object_id) in enumerate([
        ("small", 0.45, 35.0, "demo_small"),
        ("large", 2.6, 380.0, "demo_large"),
        ("large2", 2.2, 320.0, "demo_large2"),
    ]):
        offset_km = 0.06 + idx * 0.05
        pos = (sc_pos + perp * offset_km).tolist()
        vel = sc_vel.tolist()
        dist_km = float(offset_km)
        rel_vel = 0.0
        debris_list.insert(idx, {
            "position": pos,
            "velocity": vel,
            "size": size,
            "mass": mass,
            "debris_type": "fragment",
            "object_id": object_id,
            "distance_km": dist_km,
            "relative_velocity_km_s": rel_vel,
            "material": "Aluminum",
        })
    data["debris"] = debris_list

    # Elimination suggestions: based on distance (far → наблюдение) and size/mass (small → сгорание, large → обход)
    # Buttons remain: Обход, Утилизация, Наблюдение — меняется только текст рекомендации.
    elimination_suggestions = []
    for item in debris_list:
        oid = item["object_id"]
        dist = item["distance_km"]
        rel_vel = item["relative_velocity_km_s"]
        size = item["size"]
        mass = item["mass"]
        # Далеко — всегда наблюдение
        if dist >= 0.5:
            suggestion, reason, priority = "monitor", "Объект далеко — наблюдение", 3
        else:
            # Близко: по размеру и массе
            small_size = size < 1.0
            small_mass = mass < 80
            large_size = size >= 2.0
            large_mass = mass >= 200
            if small_size and small_mass:
                suggestion = "capture"
                reason = "Малый объект — выброс в атмосферу и сгорание (после расщипления при необходимости)"
                priority = 1
            elif large_size or large_mass:
                suggestion = "avoid"
                reason = "Крупный объект — захват нецелесообразен, рекомендуется обход"
                priority = 1
            elif size >= 1.0 and size < 2.0:
                suggestion = "capture"
                reason = "Средний объект — утилизация или сведение с орбиты"
                priority = 2
            else:
                if rel_vel >= 0.02:
                    suggestion, reason, priority = "avoid", "Высокая относительная скорость — обход", 1
                else:
                    suggestion, reason, priority = "monitor", "В зоне наблюдения", 2
        elimination_suggestions.append({
            "object_id": oid,
            "suggestion": suggestion,
            "reason": reason,
            "priority": priority,
        })
    data["elimination_suggestions"] = elimination_suggestions

    data["spacecraft_status"]["position"] = sc_pos.tolist()
    data["spacecraft_status"]["reference_size_m"] = REFERENCE_SIZE_M
    velocities_arr = np.array(data["velocities"])
    data["speeds"] = np.linalg.norm(velocities_arr, axis=1).tolist()
    current_danger = data["danger_levels"][-1] if data["danger_levels"] else 0.0
    # #region agent log
    _dlog("run_simulation_end", {"steps_done": len(data["times"])}, "H3")
    # #endregion
    total_warnings = (
        len(data["collision_warnings"]) + len(data["anomaly_detections"]) + len(data["low_fuel_warnings"])
    )
    data["spacecraft_status"].update({
        "fuel": float(env.spacecraft.fuel_mass),
        "debris_remaining": len(env.debris_objects),
        "danger_level": float(current_danger),
        "total_warnings": total_warnings,
    })
    return data


@app.route("/")
def index():
    """Serve main dashboard page."""
    # #region agent log
    _dlog_937("index_served", {"path": "/"}, "H5")
    # #endregion
    return render_template("index.html")


@app.route("/api/mission-data")
def mission_data():
    """API endpoint for mission data. Accepts optional ?seed= for reproducible runs."""
    # #region agent log
    _dlog("mission_data_start", {"args": dict(request.args), "module_file": __file__, "has_json_safe": callable(_json_safe), "has_dlog_937": callable(_dlog_937)}, "H2")
    # #endregion
    default_seed = 42
    try:
        raw = request.args.get("seed")
        if raw is None:
            seed = default_seed
        else:
            seed = int(raw)
    except (ValueError, TypeError):
        seed = default_seed
    try:
        data = run_simulation(num_steps=120, seed=seed)
        data["seed"] = seed
        data = _json_safe(data)
        body = _mission_data_response_body(data)
        # #region agent log
        _dlog("mission_data_ok", {"seed": seed}, "H2")
        _dlog_937(
            "mission_data_ok",
            {
                "seed": seed,
                "has_virtual_sensors": "virtual_sensors" in data,
                "body_len": len(body),
                "response_kind": "Response_preencoded",
            },
            "H4",
        )
        # #endregion
        return Response(body, mimetype="application/json; charset=utf-8")
    except Exception as e:
        # #region agent log
        _dlog("mission_data_error", {"error": str(e), "type": type(e).__name__}, "H2")
        _dlog_937("mission_data_error", {"error": str(e), "type": type(e).__name__}, "H4")
        # #endregion
        raise


@app.route("/api/status")
def status():
    """API endpoint for system status."""
    # Get latest danger level from mission data
    mission_data = run_simulation(num_steps=1, seed=42)
    current_danger = mission_data.get("spacecraft_status", {}).get("danger_level", 0.0)
    total_warnings = mission_data.get("spacecraft_status", {}).get("total_warnings", 0)
    
    if current_danger > 0.7:
        status_text = "CRITICAL"
    elif current_danger > 0.4:
        status_text = "WARNING"
    elif current_danger > 0.1:
        status_text = "CAUTION"
    else:
        status_text = "NOMINAL"
    
    return jsonify({
        "system": "A.R.I.E.S",
        "full_name": "Autonomous Research & Intelligence Earth Satellite",
        "status": status_text,
        "fusion": "ACTIVE",
        "mission": "DEBRIS COLLECTION",
        "danger_level": float(current_danger),
        "total_warnings": total_warnings,
    })


def _disposal_method_for_debris(size_m: float, mass_kg: float, debris_type: str) -> dict:
    """
    Suggest disposal method based on ESA/NASA-style approaches:
    - Very small: laser ablation (NASA ground-based laser)
    - Small: net capture (RemoveDebris)
    - Medium: harpoon or robotic arm
    - Large: rendezvous + capture + deorbit (ADR, ClearSpace-1)
    """
    if size_m < 0.1 or mass_kg < 1:
        return {
            "method": "Laser ablation",
            "description": "Ground-based or on-board pulsed laser to deorbit small debris (NASA approach for 1–10 cm objects).",
        }
    if size_m < 1.0 or mass_kg < 50:
        return {
            "method": "Net capture",
            "description": "Deploy net to capture debris, then deorbit (RemoveDebris mission demonstrated).",
        }
    if size_m < 3.0 or mass_kg < 200:
        return {
            "method": "Harpoon or robotic arm",
            "description": "Harpoon capture or robotic arm grapple, then controlled reentry (RemoveDebris / ADR).",
        }
    return {
        "method": "Rendezvous and capture + deorbit",
        "description": "Close proximity operations, capture and deorbit (ClearSpace-1 / ESA ADRIOS style).",
    }


@app.route("/api/disposal-method", methods=["GET", "POST"])
def disposal_method():
    """
    Return suggested disposal method for debris (ESA/NASA-style).
    GET: ?size=1.5&mass=100&debris_type=fragment
    POST: JSON { "size": 1.5, "mass": 100, "debris_type": "fragment" }
    """
    if request.method == "POST" and request.is_json:
        body = request.get_json() or {}
        size_m = float(body.get("size", 0.5))
        mass_kg = float(body.get("mass", 50))
        debris_type = str(body.get("debris_type", "unknown"))
    else:
        try:
            size_m = float(request.args.get("size", 0.5))
            mass_kg = float(request.args.get("mass", 50))
        except (ValueError, TypeError):
            size_m, mass_kg = 0.5, 50
        debris_type = request.args.get("debris_type", "unknown")
    result = _disposal_method_for_debris(size_m, mass_kg, debris_type)
    return jsonify(result)


@app.route("/api/virtual-sensors")
def virtual_sensors():
    """Latest virtual sensor readings from the background sim (no full replay per request).

    Query ``one_shot=1`` — previous behaviour: run a short standalone simulation (uses ``seed``).
    """
    raw_one = (request.args.get("one_shot") or "").lower()
    if raw_one in ("1", "true", "yes"):
        seed = 42
        try:
            seed = int(request.args.get("seed", 42))
        except (ValueError, TypeError):
            pass
        data = run_simulation(num_steps=10, seed=seed)
        return jsonify(data.get("virtual_sensors", {}))

    runner = ensure_virtual_sensor_runner()
    _seq, latest = runner.get_latest()
    return jsonify(latest)


@app.route("/api/virtual-sensors/stream")
def virtual_sensors_stream():
    """Server-Sent Events: push each new virtual-sensor frame from the background sim."""

    def generate():
        last_seq = -1
        tick = 0
        while True:
            runner = ensure_virtual_sensor_runner()
            seq, latest = runner.get_latest()
            if seq != last_seq and latest:
                last_seq = seq
                yield "data: " + json.dumps(latest, ensure_ascii=False, allow_nan=False) + "\n\n"
            tick += 1
            if tick % 400 == 0:
                yield ": ping\n\n"
            time.sleep(0.025)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/virtual-sensors/reset", methods=["POST"])
def virtual_sensors_reset():
    """Reset background sim; optional JSON body ``{\"seed\": 123}``."""
    runner = ensure_virtual_sensor_runner()
    seed = None
    if request.is_json:
        body = request.get_json(silent=True) or {}
        if "seed" in body and body["seed"] is not None:
            try:
                seed = int(body["seed"])
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "invalid seed"}), 400
    runner.request_reset(seed)
    return jsonify({"ok": True})


def run_server(host="0.0.0.0", port=5001, debug=True):
    """Run the Flask server. host=0.0.0.0 — слушать на всех интерфейсах (доступ с других устройств)."""
    # #region agent log
    _dlog("run_server_entered", {"host": host, "port": port, "module_file": __file__, "has_json_safe": callable(_json_safe), "has_dlog_937": callable(_dlog_937)}, "H1")
    _dlog_937("run_server_entered", {"host": host, "port": port, "debug": debug}, "H5")
    # #endregion
    print("=" * 70)
    print("A.R.I.E.S — Advanced Retrieval & In-Orbit Elimination System")
    print("Web Dashboard Server")
    print("=" * 70)
    print(f"\nServer starting at: http://{host}:{port}")
    print(f"  На этом ПК:        http://127.0.0.1:{port}")
    if host == "0.0.0.0":
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            print(f"  С других устройств: http://{local_ip}:{port}")
        except Exception:
            print(f"  С других устройств: http://<IP этого ПК>:{port}")
    print("\nOpen your browser and navigate to the URL above.")
    if host == "0.0.0.0":
        print(f"  If another device can't connect - allow Python in Windows Firewall (Port TCP {port}).")
    print("Press Ctrl+C to stop the server.")
    ensure_virtual_sensor_runner()
    print("  Virtual sensors: GET /api/virtual-sensors  |  SSE /api/virtual-sensors/stream")
    print("  Arduino: GET /api/arduino/live  |  SSE /api/arduino/stream  (ARDUINO_PORT=COMx, pip install pyserial)\n")
    
    # use_reloader=False при 0.0.0.0 — иначе на Windows сервер может не принимать подключения из сети
    use_reloader = debug and host != "0.0.0.0"
    app.run(host=host, port=port, debug=debug, use_reloader=use_reloader, threaded=True)


if __name__ == "__main__":
    run_server()

