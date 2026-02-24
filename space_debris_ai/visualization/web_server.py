"""
A.R.I.E.S Web Dashboard Server.
Flask server for AetherOS-style web interface.
"""

from flask import Flask, render_template, jsonify, request
import numpy as np
import sys
from pathlib import Path

# Add parent to path
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from space_debris_ai.simulation.environment import OrbitalEnv, EnvConfig
from space_debris_ai.sensors.fusion import SensorFusion


def _dlog(msg, data=None, hypothesis_id=None):
    import json
    import time
    try:
        with open("debug-17e329.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "17e329", "message": msg, "data": data or {}, "hypothesisId": hypothesis_id, "timestamp": int(time.time() * 1000), "location": "web_server.py"}) + "\n")
    except Exception:
        pass


app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)


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

        gps = {"position": env.spacecraft.position, "velocity": env.spacecraft.velocity}
        imu = {"angular_velocity": env.spacecraft.angular_velocity}
        star = {"attitude": env.spacecraft.attitude}
        fused = fusion.fuse(gps_data=gps, imu_data=imu, star_tracker_data=star)
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

        # 2) Аномалия по шагу/сиду
        if step > 20 and (step + seed) % 37 == 0:
            score = 0.4 + np.random.rand() * 0.4
            data["anomaly_detections"].append({
                "time": float(t),
                "score": float(score),
                "type": "TELEMETRY_ANOMALY",
            })
            danger_level = max(danger_level, score)

        # 3) Низкое топливо
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
    data["debris"] = debris_list

    # Elimination suggestions per debris
    elimination_suggestions = []
    for item in debris_list:
        oid = item["object_id"]
        dist = item["distance_km"]
        rel_vel = item["relative_velocity_km_s"]
        size = item["size"]
        if dist < 0.05 and rel_vel < 0.01:
            suggestion, reason, priority = "capture", "Within capture range, low relative velocity", 1
        elif dist < 0.15 and rel_vel >= 0.01:
            suggestion, reason, priority = "avoid", "Collision risk: high relative velocity", 1
        elif dist < 0.5:
            suggestion, reason, priority = "monitor", "Within monitoring range", 2
        else:
            suggestion, reason, priority = "monitor", "Track for future operations", 3
        if size > 2.0 and suggestion in ("monitor", "capture"):
            reason = reason + "; large object, consider deorbit"
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
    return render_template("index.html")


@app.route("/api/mission-data")
def mission_data():
    """API endpoint for mission data. Accepts optional ?seed= for reproducible runs."""
    # #region agent log
    _dlog("mission_data_start", {"args": dict(request.args)}, "H2")
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
        # #region agent log
        _dlog("mission_data_ok", {"seed": seed}, "H2")
        # #endregion
        return jsonify(data)
    except Exception as e:
        # #region agent log
        _dlog("mission_data_error", {"error": str(e), "type": type(e).__name__}, "H2")
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
        "full_name": "Advanced Retrieval & In-Orbit Elimination System",
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


def run_server(host="127.0.0.1", port=5000, debug=True):
    """Run the Flask server."""
    # #region agent log
    _dlog("run_server_entered", {"host": host, "port": port}, "H1")
    # #endregion
    print("=" * 70)
    print("A.R.I.E.S — Advanced Retrieval & In-Orbit Elimination System")
    print("Web Dashboard Server")
    print("=" * 70)
    print(f"\nServer starting at: http://{host}:{port}")
    print("\nOpen your browser and navigate to the URL above.")
    print("Press Ctrl+C to stop the server.\n")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server()

