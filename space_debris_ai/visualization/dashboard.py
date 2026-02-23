"""
A.R.I.E.S — Advanced Retrieval & In-Orbit Elimination System.
Mission dashboard: 3D orbit, trajectory, debris, sensor fusion.
AetherOS-style dark theme.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional

from space_debris_ai.simulation.environment import OrbitalEnv
from space_debris_ai.simulation.physics import EARTH_RADIUS
from space_debris_ai.sensors.fusion import SensorFusion

# AetherOS-style dark palette
DARK_BG = "#0a0a0f"
DARK_PANEL = "#12121a"
ACCENT_BLUE = "#58a6ff"
ACCENT_CYAN = "#00d4ff"  # Primary system accent (AetherOS)
ACCENT_ORANGE = "#ff8c42"
ACCENT_GREEN = "#3fb950"
ACCENT_PURPLE = "#a5a5ff"
TEXT_PRIMARY = "#e0e6ed"
TEXT_SECONDARY = "#6b7a8c"
GRID_COLOR = "#1a2332"

# A.R.I.E.S expansion for UI
ARIES_FULL = "Advanced Retrieval & In-Orbit Elimination System"


def _run_simulation(num_steps: int = 150, seed: Optional[int] = 42) -> dict:
    """Run env for num_steps with zero/low action, collect trajectory and fusion."""
    from space_debris_ai.simulation.environment import EnvConfig
    from space_debris_ai.models import CollisionAvoidanceModule, AnomalyDetector
    from space_debris_ai.core import SystemConfig

    config = EnvConfig(
        dt=0.5,
        max_episode_steps=num_steps + 10,
        num_debris=20,
    )
    env = OrbitalEnv(config=config)
    fusion = SensorFusion()
    
    # Initialize safety modules for danger detection
    system_config = SystemConfig()
    collision_module = CollisionAvoidanceModule(
        config={"lidar_points": 256},
        device="cpu",
    )
    anomaly_module = AnomalyDetector(
        config={"input_dim": 12, "seq_len": 50},
        device="cpu",
    )

    obs, info = env.reset(seed=seed)
    times = [0.0]
    positions = [env.spacecraft.position.copy()]
    velocities = [env.spacecraft.velocity.copy()]
    confidences = [1.0]
    fuels = [env.spacecraft.fuel_mass]
    debris_counts = [len(env.debris_objects)]
    debris_positions: List[np.ndarray] = []
    
    # Danger tracking
    collision_warnings = []
    anomaly_detections = []
    low_fuel_warnings = []
    danger_levels = [0.0]  # Initial danger level at start

    for step in range(num_steps - 1):
        # Small random action so orbit evolves slightly
        action = np.zeros(7)
        action[:3] = np.random.randn(3) * 0.02
        obs, reward, term, trunc, info = env.step(action)

        t = env.spacecraft.time
        times.append(t)
        positions.append(env.spacecraft.position.copy())
        velocities.append(env.spacecraft.velocity.copy())
        fuels.append(env.spacecraft.fuel_mass)
        debris_counts.append(len(env.debris_objects))

        # Simulate sensor inputs from true state and fuse
        gps = {"position": env.spacecraft.position, "velocity": env.spacecraft.velocity}
        imu = {"angular_velocity": env.spacecraft.angular_velocity}
        star = {"attitude": env.spacecraft.attitude}
        fused = fusion.fuse(gps_data=gps, imu_data=imu, star_tracker_data=star)
        confidences.append(float(fused.confidence))
        
        # Check for dangerous situations
        danger_level = 0.0
        danger_messages = []
        
        # 1. Check collision warnings
        try:
            # Simulate lidar/radar data
            # Create dangerous situation: simulate close object approaching
            lidar_data = np.random.randn(256, 6).astype(np.float32)
            # Make some points very close (dangerous)
            if step % 30 == 0 and step > 10:  # Create danger every 30 steps
                lidar_data[:20, :3] = np.random.randn(20, 3) * 0.01  # Very close points
                lidar_data[:20, 3:6] = np.random.randn(20, 3) * 5.0  # High velocity
            
            radar_data = np.random.randn(64).astype(np.float32)
            # Make radar detect high-speed approach
            if step % 30 == 0 and step > 10:
                radar_data[:10] = np.random.randn(10) * 10.0  # High velocity signal
            
            imu_data = np.concatenate([env.spacecraft.angular_velocity, np.random.randn(9)]).astype(np.float32)
            
            collision_result = collision_module.forward({
                "lidar": lidar_data,
                "radar": radar_data,
                "imu": imu_data,
            })
            
            collision_prob = collision_result.get("collision_probability", 0.0)
            # Artificially increase probability for demonstration
            if step % 30 == 0 and step > 10:
                collision_prob = min(1.0, collision_prob + 0.6)
            
            if collision_result.get("avoidance_active", False) or collision_prob > 0.3:
                if collision_prob > 0.3:
                    collision_warnings.append({
                        "time": t,
                        "probability": float(collision_prob),
                        "severity": "HIGH" if collision_prob > 0.7 else "MEDIUM",
                    })
                    danger_level = max(danger_level, collision_prob)
                    danger_messages.append(f"⚠️ СТОЛКНОВЕНИЕ! Вероятность: {collision_prob:.1%}")
        except Exception:
            pass
        
        # 2. Check anomalies
        try:
            # Build telemetry sequence with potential anomalies
            telemetry_seq = np.random.randn(50, 12).astype(np.float32)
            
            # Create dangerous anomaly: sudden acceleration spike
            if step % 40 == 0 and step > 20:  # Create anomaly every 40 steps
                # Add sudden acceleration spike (dangerous)
                telemetry_seq[-10:, :3] = np.random.randn(10, 3) * 10.0  # High acceleration
                telemetry_seq[-10:, 3:6] = np.random.randn(10, 3) * 20.0  # High velocity change
            
            anomaly_result = anomaly_module.forward({
                "telemetry": telemetry_seq,
            })
            
            anomaly_score = anomaly_result.get("anomaly_score", 0.0)
            # Artificially increase score for demonstration
            if step % 40 == 0 and step > 20:
                anomaly_score = min(1.0, anomaly_score + 0.5)
                anomaly_result["is_anomaly"] = True
            
            if anomaly_result.get("is_anomaly", False) or anomaly_score > 0.5:
                anomaly_detections.append({
                    "time": t,
                    "score": float(anomaly_score),
                    "type": str(anomaly_result.get("anomaly_type", "UNKNOWN")),
                })
                danger_level = max(danger_level, anomaly_score)
                danger_messages.append(f"🚨 АНОМАЛИЯ! Уровень: {anomaly_score:.1%}")
        except Exception:
            pass
        
        # 3. Check fuel level
        fuel_ratio = env.spacecraft.fuel_mass / config.fuel_mass if config.fuel_mass > 0 else 1.0
        # Simulate fuel depletion for demonstration
        if step > num_steps * 0.6:  # After 60% of simulation
            fuel_ratio = max(0.0, fuel_ratio - 0.001 * (step - num_steps * 0.6))
        
        if fuel_ratio < 0.2:  # Less than 20% fuel
            low_fuel_warnings.append({
                "time": t,
                "fuel_remaining": float(env.spacecraft.fuel_mass * fuel_ratio),
                "fuel_ratio": float(fuel_ratio),
            })
            danger_level = max(danger_level, 0.5)
            danger_messages.append(f"⛽ НИЗКИЙ УРОВЕНЬ ТОПЛИВА! {fuel_ratio:.1%}")
        
        # 4. Check for very close debris
        for debris in env.debris_objects:
            distance = np.linalg.norm(debris.position - env.spacecraft.position)
            if distance < 0.1:  # Less than 100m
                danger_level = max(danger_level, 0.8)
                danger_messages.append(f"💥 ОПАСНАЯ БЛИЗОСТЬ! Расстояние: {distance*1000:.0f}m")
        
        danger_levels.append(danger_level)

        if term or trunc:
            break

    # Snapshot of debris positions at end (for 3D plot)
    debris_positions = [d.position.copy() for d in env.debris_objects]

    return {
        "times": np.array(times),
        "positions": np.array(positions),
        "velocities": np.array(velocities),
        "confidences": np.array(confidences),
        "fuels": np.array(fuels),
        "debris_counts": np.array(debris_counts),
        "debris_positions": np.array(debris_positions) if debris_positions else np.zeros((0, 3)),
        "dt": config.dt,
        "collision_warnings": collision_warnings,
        "anomaly_detections": anomaly_detections,
        "low_fuel_warnings": low_fuel_warnings,
        "danger_levels": np.array(danger_levels),
    }


def _plot_earth_axis(ax: Axes3D, scale: float = 1.2) -> None:
    """Draw Earth as a beautiful sphere with dark theme styling."""
    u = np.linspace(0, 2 * np.pi, 32)
    v = np.linspace(0, np.pi, 20)
    x = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS * np.outer(np.ones_like(u), np.cos(v))
    
    # Earth (AetherOS-style)
    ax.plot_surface(
        x, y, z,
        color="#1a3a52",
        alpha=0.5,
        edgecolor=ACCENT_CYAN,
        linewidth=0.25,
        shade=True,
    )
    
    # Add subtle grid lines
    for i in range(0, len(u), 8):
        ax.plot(x[i, :], y[i, :], z[i, :], color=GRID_COLOR, alpha=0.2, linewidth=0.5)
    
    r = EARTH_RADIUS * scale
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    
    # Dark theme axis labels
    ax.set_xlabel("X (km)", color=TEXT_PRIMARY, fontsize=9)
    ax.set_ylabel("Y (km)", color=TEXT_PRIMARY, fontsize=9)
    ax.set_zlabel("Z (km)", color=TEXT_PRIMARY, fontsize=9)
    
    # Dark theme axis styling
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GRID_COLOR)
    ax.yaxis.pane.set_edgecolor(GRID_COLOR)
    ax.zaxis.pane.set_edgecolor(GRID_COLOR)
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)


def run_dashboard(
    num_steps: int = 150,
    seed: Optional[int] = 42,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Run a short simulation and show mission dashboard.

    Parameters
    ----------
    num_steps : int
        Number of simulation steps to record.
    seed : int, optional
        Random seed for reproducible run.
    show : bool
        If True, call plt.show() at the end.
    save_path : str, optional
        If set, save figure to this path (e.g. 'mission_dashboard.png').
    """
    data = _run_simulation(num_steps=num_steps, seed=seed)
    t = data["times"]
    pos = data["positions"]
    vel = data["velocities"]
    conf = data["confidences"]
    fuels = data["fuels"]
    debris_counts = data["debris_counts"]
    debris_pos = data["debris_positions"]
    collision_warnings = data.get("collision_warnings", [])
    anomaly_detections = data.get("anomaly_detections", [])
    low_fuel_warnings = data.get("low_fuel_warnings", [])
    danger_levels = data.get("danger_levels", np.zeros_like(t))

    speed = np.linalg.norm(vel, axis=1)
    
    # Calculate overall danger status
    max_danger = np.max(danger_levels) if len(danger_levels) > 0 else 0.0
    current_danger = danger_levels[-1] if len(danger_levels) > 0 else 0.0
    total_warnings = len(collision_warnings) + len(anomaly_detections) + len(low_fuel_warnings)

    # Create figure with dark background
    fig = plt.figure(figsize=(16, 11), facecolor=DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    
    # A.R.I.E.S branding + expansion
    fig.suptitle(
        "A.R.I.E.S · Mission Dashboard",
        fontsize=18,
        fontweight="bold",
        color=TEXT_PRIMARY,
        y=0.98,
        family="monospace",
    )
    fig.text(
        0.5, 0.94,
        ARIES_FULL,
        ha="center",
        fontsize=10,
        color=TEXT_SECONDARY,
    )
    fig.text(
        0.5, 0.905,
        "Mission Control",
        ha="center",
        fontsize=11,
        color=TEXT_SECONDARY,
        style="italic",
    )

    # 3D orbit and debris
    ax3d = fig.add_subplot(2, 2, 1, projection="3d", facecolor=DARK_PANEL)
    ax3d.set_facecolor(DARK_PANEL)
    
    # Trajectory (AetherOS cyan accent)
    ax3d.plot(
        pos[:, 0], pos[:, 1], pos[:, 2],
        color=ACCENT_CYAN,
        linewidth=2.5,
        label="Spacecraft Trajectory",
        alpha=0.9,
    )
    
    # Add start/end markers
    ax3d.scatter(
        pos[0, 0], pos[0, 1], pos[0, 2],
        color=ACCENT_GREEN,
        s=100,
        marker="o",
        edgecolors="white",
        linewidths=1.5,
        label="Start",
        zorder=10,
    )
    ax3d.scatter(
        pos[-1, 0], pos[-1, 1], pos[-1, 2],
        color=ACCENT_ORANGE,
        s=100,
        marker="*",
        edgecolors="white",
        linewidths=1.5,
        label="Current",
        zorder=10,
    )
    
    if debris_pos.shape[0] > 0:
        # Highlight dangerous debris (very close)
        dangerous_debris = []
        safe_debris = []
        for i, d_pos in enumerate(debris_pos):
            dist = np.linalg.norm(d_pos - pos[-1])
            if dist < 0.1:  # Less than 100m
                dangerous_debris.append(d_pos)
            else:
                safe_debris.append(d_pos)
        
        if safe_debris:
            safe_arr = np.array(safe_debris)
            ax3d.scatter(
                safe_arr[:, 0],
                safe_arr[:, 1],
                safe_arr[:, 2],
                c=ACCENT_ORANGE,
                s=50,
                alpha=0.6,
                label=f"Debris ({len(safe_debris)})",
                edgecolors="white",
                linewidths=0.5,
            )
        
        if dangerous_debris:
            danger_arr = np.array(dangerous_debris)
            ax3d.scatter(
                danger_arr[:, 0],
                danger_arr[:, 1],
                danger_arr[:, 2],
                c="#ff4444",
                s=200,
                alpha=0.9,
                label=f"⚠️ ОПАСНЫЙ МУСОР ({len(dangerous_debris)})",
                edgecolors="white",
                linewidths=2,
                marker="X",
            )
    
    _plot_earth_axis(ax3d)
    ax3d.legend(
        loc="upper right",
        fontsize=9,
        facecolor=DARK_PANEL,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_PRIMARY,
    )
    ax3d.set_title("ORBIT TRACK", color=TEXT_PRIMARY, fontsize=12, fontweight="bold", pad=15, family="monospace")

    # Position components vs time
    ax_pos = fig.add_subplot(2, 2, 2, facecolor=DARK_PANEL)
    ax_pos.set_facecolor(DARK_PANEL)
    
    ax_pos.plot(t, pos[:, 0], label="X", color=ACCENT_BLUE, linewidth=2, alpha=0.9)
    ax_pos.plot(t, pos[:, 1], label="Y", color=ACCENT_CYAN, linewidth=2, alpha=0.9)
    ax_pos.plot(t, pos[:, 2], label="Z", color=ACCENT_PURPLE, linewidth=2, alpha=0.9)
    
    ax_pos.set_xlabel("Time (s)", color=TEXT_PRIMARY, fontsize=10)
    ax_pos.set_ylabel("Position (km)", color=TEXT_PRIMARY, fontsize=10)
    ax_pos.set_title("TELEMETRY — ECI", color=TEXT_PRIMARY, fontsize=12, fontweight="bold", pad=10, family="monospace")
    ax_pos.legend(
        facecolor=DARK_PANEL,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_PRIMARY,
        fontsize=9,
    )
    ax_pos.grid(True, alpha=0.2, color=GRID_COLOR, linestyle="--", linewidth=0.8)
    ax_pos.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax_pos.spines["bottom"].set_color(GRID_COLOR)
    ax_pos.spines["top"].set_color(GRID_COLOR)
    ax_pos.spines["right"].set_color(GRID_COLOR)
    ax_pos.spines["left"].set_color(GRID_COLOR)

    # Speed and fusion confidence
    ax_vel = fig.add_subplot(2, 2, 3, facecolor=DARK_PANEL)
    ax_vel.set_facecolor(DARK_PANEL)
    
    ax_vel.plot(
        t, speed,
        color=ACCENT_GREEN,
        label="Speed (km/s)",
        linewidth=2.5,
        alpha=0.9,
    )
    ax_vel.fill_between(t, speed, alpha=0.2, color=ACCENT_GREEN)
    
    ax_vel.set_xlabel("Time (s)", color=TEXT_PRIMARY, fontsize=10)
    ax_vel.set_ylabel("Speed (km/s)", color=TEXT_PRIMARY, fontsize=10)
    ax_vel.set_title("FUSION & VELOCITY", color=TEXT_PRIMARY, fontsize=12, fontweight="bold", pad=10, family="monospace")
    ax_vel.legend(
        loc="upper left",
        facecolor=DARK_PANEL,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_PRIMARY,
        fontsize=9,
    )
    ax_vel.grid(True, alpha=0.2, color=GRID_COLOR, linestyle="--", linewidth=0.8)
    ax_vel.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax_vel.spines["bottom"].set_color(GRID_COLOR)
    ax_vel.spines["top"].set_color(GRID_COLOR)
    ax_vel.spines["right"].set_color(GRID_COLOR)
    ax_vel.spines["left"].set_color(GRID_COLOR)

    ax_conf = ax_vel.twinx()
    ax_conf.plot(
        t, conf,
        color=ACCENT_PURPLE,
        alpha=0.9,
        label="Fusion Confidence",
        linewidth=2,
        linestyle="--",
    )
    ax_conf.fill_between(t, conf, alpha=0.15, color=ACCENT_PURPLE)
    ax_conf.set_ylabel("Confidence", color=ACCENT_PURPLE, fontsize=10)
    ax_conf.legend(
        loc="upper right",
        facecolor=DARK_PANEL,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_PRIMARY,
        fontsize=9,
    )
    ax_conf.set_ylim(0, 1.05)
    ax_conf.tick_params(colors=ACCENT_PURPLE, labelsize=9)
    ax_conf.spines["bottom"].set_color(GRID_COLOR)
    ax_conf.spines["top"].set_color(GRID_COLOR)
    ax_conf.spines["right"].set_color(ACCENT_PURPLE)
    ax_conf.spines["left"].set_color(GRID_COLOR)

    # Fuel and debris count with danger overlay
    ax_res = fig.add_subplot(2, 2, 4, facecolor=DARK_PANEL)
    ax_res.set_facecolor(DARK_PANEL)
    
    ax_res.plot(
        t, fuels,
        color=ACCENT_ORANGE,
        label="Fuel (kg)",
        linewidth=2.5,
        alpha=0.9,
    )
    ax_res.fill_between(t, fuels, alpha=0.2, color=ACCENT_ORANGE)
    
    # Highlight low fuel regions
    if len(low_fuel_warnings) > 0:
        for warning in low_fuel_warnings:
            ax_res.axvline(x=warning["time"], color="#ff4444", linestyle=":", alpha=0.6, linewidth=1.5)
    
    ax_res.set_xlabel("Time (s)", color=TEXT_PRIMARY, fontsize=10)
    ax_res.set_ylabel("Fuel (kg)", color=TEXT_PRIMARY, fontsize=10)
    ax_res.set_title("RESOURCES & DANGER STATUS", color=TEXT_PRIMARY, fontsize=12, fontweight="bold", pad=10, family="monospace")
    ax_res.legend(
        loc="upper right",
        facecolor=DARK_PANEL,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_PRIMARY,
        fontsize=9,
    )
    ax_res.grid(True, alpha=0.2, color=GRID_COLOR, linestyle="--", linewidth=0.8)
    ax_res.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax_res.spines["bottom"].set_color(GRID_COLOR)
    ax_res.spines["top"].set_color(GRID_COLOR)
    ax_res.spines["right"].set_color(GRID_COLOR)
    ax_res.spines["left"].set_color(GRID_COLOR)

    ax_deb = ax_res.twinx()
    ax_deb.plot(
        t, debris_counts,
        color=ACCENT_CYAN,
        alpha=0.9,
        label="Debris Count",
        linewidth=2,
        linestyle=":",
    )
    ax_deb.fill_between(t, debris_counts, alpha=0.15, color=ACCENT_CYAN)
    ax_deb.set_ylabel("Debris Count", color=ACCENT_CYAN, fontsize=10)
    ax_deb.legend(
        loc="center right",
        facecolor=DARK_PANEL,
        edgecolor=GRID_COLOR,
        labelcolor=TEXT_PRIMARY,
        fontsize=9,
    )
    ax_deb.tick_params(colors=ACCENT_CYAN, labelsize=9)
    ax_deb.spines["bottom"].set_color(GRID_COLOR)
    ax_deb.spines["top"].set_color(GRID_COLOR)
    ax_deb.spines["right"].set_color(ACCENT_CYAN)
    ax_deb.spines["left"].set_color(GRID_COLOR)

    # Danger indicator panel
    danger_color = ACCENT_GREEN
    danger_status = "NOMINAL"
    if current_danger > 0.7:
        danger_color = "#ff4444"  # Red
        danger_status = "CRITICAL"
    elif current_danger > 0.4:
        danger_color = ACCENT_ORANGE
        danger_status = "WARNING"
    elif current_danger > 0.1:
        danger_color = "#ffaa00"  # Yellow
        danger_status = "CAUTION"
    
    # Add danger warnings text box
    if total_warnings > 0 or current_danger > 0.1:
        warning_text = []
        if len(collision_warnings) > 0:
            warning_text.append(f"⚠️ Столкновений: {len(collision_warnings)}")
        if len(anomaly_detections) > 0:
            warning_text.append(f"🚨 Аномалий: {len(anomaly_detections)}")
        if len(low_fuel_warnings) > 0:
            warning_text.append(f"⛽ Низкое топливо: {len(low_fuel_warnings)}")
        
        if warning_text:
            fig.text(
                0.02, 0.02,
                "\n".join(warning_text),
                ha="left",
                fontsize=9,
                color=danger_color,
                family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor=DARK_PANEL,
                    edgecolor=danger_color,
                    linewidth=2,
                    alpha=0.9,
                ),
            )
    
    # A.R.I.E.S status line (AetherOS-style) with danger indicator
    status_text = f"A.R.I.E.S · Advanced Retrieval & In-Orbit Elimination System · STATUS: {danger_status} · FUSION ACTIVE"
    if total_warnings > 0:
        status_text += f" · WARNINGS: {total_warnings}"
    
    fig.text(
        0.5, 0.01,
        status_text,
        ha="center",
        fontsize=7,
        color=danger_color if current_danger > 0.1 else TEXT_SECONDARY,
        family="monospace",
        weight="bold" if current_danger > 0.4 else "normal",
    )
    
    # Add danger level plot overlay on resources panel
    if len(danger_levels) > 0:
        # Ensure arrays have same length
        danger_levels_array = np.array(danger_levels)
        min_len = min(len(t), len(danger_levels_array))
        t_danger = t[:min_len]
        danger_levels_array = danger_levels_array[:min_len]
        
        ax_danger = ax_res.twinx()
        ax_danger.plot(
            t_danger, danger_levels_array,
            color=danger_color,
            alpha=0.8,
            label="Уровень опасности",
            linewidth=2.5,
            linestyle="-",
        )
        ax_danger.fill_between(t_danger, danger_levels_array, alpha=0.2, color=danger_color)
        ax_danger.set_ylabel("Уровень опасности", color=danger_color, fontsize=9)
        ax_danger.set_ylim(0, 1.05)
        ax_danger.tick_params(colors=danger_color, labelsize=8)
        ax_danger.spines["right"].set_color(danger_color)
        
        # Add danger threshold lines
        ax_danger.axhline(y=0.7, color="#ff4444", linestyle="--", alpha=0.5, linewidth=1, label="Критично")
        ax_danger.axhline(y=0.4, color=ACCENT_ORANGE, linestyle="--", alpha=0.5, linewidth=1, label="Предупреждение")

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    if save_path:
        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
            facecolor=DARK_BG,
            edgecolor="none",
        )
    if show:
        plt.show()


if __name__ == "__main__":
    run_dashboard(num_steps=120, seed=42, save_path="mission_dashboard.png")
