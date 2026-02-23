"""
Бесконечный генератор данных для обучения.
Генерирует симуляции с разными сидами и сохраняет данные, включая опасности.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np

# Add parent to path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Лёгкий режим: симуляция без PyTorch (только окружение + простые правила опасностей)
def run_simulation_light(num_steps: int = 150, seed: int = 42) -> dict:
    """Симуляция без нейросетей: только OrbitalEnv, SensorFusion и простые опасности."""
    from space_debris_ai.simulation.environment import OrbitalEnv, EnvConfig
    from space_debris_ai.sensors.fusion import SensorFusion

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

        # 1) Опасность: близкий мусор
        for d in env.debris_objects:
            dist = np.linalg.norm(d.position - sc_pos)
            if dist < 0.15:  # 150 m
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

        # 2) Аномалия: симулируем по шагу/сиду
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
    velocities_arr = np.array(data["velocities"])
    data["speeds"] = np.linalg.norm(velocities_arr, axis=1).tolist()
    data["spacecraft_status"]["fuel"] = float(env.spacecraft.fuel_mass)
    data["spacecraft_status"]["debris_remaining"] = len(env.debris_objects)
    data["spacecraft_status"]["danger_level"] = float(np.max(data["danger_levels"]))
    data["spacecraft_status"]["total_warnings"] = (
        len(data["collision_warnings"]) + len(data["anomaly_detections"]) + len(data["low_fuel_warnings"])
    )
    return data


def run_simulation_full(num_steps: int = 150, seed: int = 42) -> dict:
    """Полная симуляция с нейросетями (PyTorch)."""
    from space_debris_ai.visualization.web_server import run_simulation
    return run_simulation(num_steps=num_steps, seed=seed)


def build_dangers_summary(data: dict, seed: int) -> dict:
    """
    Собирает все опасности из симуляции в единый список с типом и серьёзностью.
    """
    dangers = []
    
    for w in data.get("collision_warnings", []):
        dangers.append({
            "type": "collision",
            "type_ru": "столкновение",
            "time": w["time"],
            "severity": w.get("severity", "MEDIUM"),
            "value": w.get("probability", 0.0),
            "description": f"Вероятность столкновения {w.get('probability', 0):.1%}",
        })
    
    for a in data.get("anomaly_detections", []):
        dangers.append({
            "type": "anomaly",
            "type_ru": "аномалия",
            "time": a["time"],
            "severity": "HIGH" if a.get("score", 0) > 0.7 else "MEDIUM",
            "value": a.get("score", 0.0),
            "description": f"Аномалия телеметрии, уровень {a.get('score', 0):.1%}",
        })
    
    for f in data.get("low_fuel_warnings", []):
        dangers.append({
            "type": "low_fuel",
            "type_ru": "низкое топливо",
            "time": f["time"],
            "severity": "HIGH" if f.get("fuel_ratio", 0) < 0.1 else "MEDIUM",
            "value": f.get("fuel_ratio", 0.0),
            "description": f"Остаток топлива {f.get('fuel_ratio', 0):.1%}",
        })
    
    danger_levels = data.get("danger_levels", [])
    max_level = float(np.max(danger_levels)) if danger_levels else 0.0
    
    return {
        "seed": seed,
        "total_count": len(dangers),
        "collision_count": len(data.get("collision_warnings", [])),
        "anomaly_count": len(data.get("anomaly_detections", [])),
        "low_fuel_count": len(data.get("low_fuel_warnings", [])),
        "max_danger_level": max_level,
        "has_danger": max_level > 0.1,
        "events": sorted(dangers, key=lambda x: x["time"]),
    }


def generate_data_continuously(
    output_dir: str = "generated_data",
    num_steps: int = 120,
    start_seed: int = 0,
    save_format: str = "json",  # "json" or "csv"
    max_files: int = None,  # None = бесконечно
    delay_between_runs: float = 0.1,  # Задержка между симуляциями (секунды)
    save_dangers_file: bool = True,  # Отдельный файл опасностей для каждого сида
    use_light: bool = True,  # Лёгкий режим без PyTorch (по умолчанию)
):
    """
    Бесконечно генерирует данные с разными сидами.
    
    Args:
        output_dir: Директория для сохранения данных
        num_steps: Количество шагов симуляции
        start_seed: Начальный сид
        save_format: Формат сохранения ("json" или "csv")
        max_files: Максимальное количество файлов (None = бесконечно)
        delay_between_runs: Задержка между симуляциями
        save_dangers_file: Сохранять ли отдельный JSON с опасностями в папку dangers/
        use_light: True = без PyTorch (быстро), False = полная симуляция с нейросетями
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    run_sim = run_simulation_light if use_light else run_simulation_full
    seed = start_seed
    file_count = 0
    
    print("=" * 70)
    print("A.R.I.E.S — Генератор данных")
    print("=" * 70)
    print(f"Директория: {output_path.absolute()}")
    print(f"Формат: {save_format}")
    print(f"Режим: {'лёгкий (без PyTorch)' if use_light else 'полный (PyTorch)'}")
    print(f"Шагов на симуляцию: {num_steps}")
    print(f"Начальный сид: {start_seed}")
    print("=" * 70)
    print("\nНажмите Ctrl+C для остановки\n")
    
    try:
        while True:
            if max_files and file_count >= max_files:
                print(f"\nДостигнут лимит файлов: {max_files}")
                break
            
            # Генерируем данные
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Сид {seed}... ", end="", flush=True)
            
            try:
                data = run_sim(num_steps=num_steps, seed=seed)
                
                # Сводка по опасностям
                dangers_summary = build_dangers_summary(data, seed)
                data["dangers"] = dangers_summary
                
                # Метаданные
                data["metadata"] = {
                    "seed": seed,
                    "num_steps": num_steps,
                    "timestamp": datetime.now().isoformat(),
                    "collision_warnings": len(data.get("collision_warnings", [])),
                    "anomaly_detections": len(data.get("anomaly_detections", [])),
                    "low_fuel_warnings": len(data.get("low_fuel_warnings", [])),
                    "max_danger_level": float(np.max(data.get("danger_levels", [0.0]))),
                    "dangers_total": dangers_summary["total_count"],
                    "has_danger": dangers_summary["has_danger"],
                }
                
                # Отдельный файл опасностей (удобно для обучения по опасностям)
                if save_dangers_file and dangers_summary["total_count"] > 0:
                    dangers_path = output_path / "dangers"
                    dangers_path.mkdir(parents=True, exist_ok=True)
                    dangers_file = dangers_path / f"dangers_seed_{seed:06d}.json"
                    with open(dangers_file, 'w', encoding='utf-8') as f:
                        json.dump(dangers_summary, f, indent=2, ensure_ascii=False)
                
                # Сохраняем файл
                if save_format == "json":
                    filename = output_path / f"simulation_seed_{seed:06d}.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                elif save_format == "csv":
                    # Сохраняем основные метрики в CSV
                    filename = output_path / f"simulation_seed_{seed:06d}.csv"
                    try:
                        import pandas as pd
                        
                        df_data = {
                            "time": data["times"],
                            "position_x": [p[0] for p in data["positions"]],
                            "position_y": [p[1] for p in data["positions"]],
                            "position_z": [p[2] for p in data["positions"]],
                            "velocity_x": [v[0] for v in data["velocities"]],
                            "velocity_y": [v[1] for v in data["velocities"]],
                            "velocity_z": [v[2] for v in data["velocities"]],
                            "speed": data.get("speeds", []),
                            "fuel": data["fuels"],
                            "confidence": data["confidences"],
                        }
                        
                        # Уровень опасности по шагам
                        if "danger_levels" in data:
                            df_data["danger_level"] = data["danger_levels"]
                        else:
                            df_data["danger_level"] = [0.0] * len(data["times"])
                        
                        # Флаг опасности на шаге (1 = было событие в этот момент)
                        times_set = set(round(ev.get("time", 0), 2) for ev in (
                            data.get("collision_warnings", []) +
                            data.get("anomaly_detections", []) +
                            data.get("low_fuel_warnings", [])
                        ))
                        df_data["danger_event"] = [1 if round(t, 2) in times_set else 0 for t in data["times"]]
                        
                        df = pd.DataFrame(df_data)
                        df.to_csv(filename, index=False)
                    except ImportError:
                        print("⚠ pandas не установлен, используем JSON формат")
                        filename = output_path / f"simulation_seed_{seed:06d}.json"
                        with open(filename, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                
                file_count += 1
                total_warnings = data['metadata']['dangers_total']
                max_d = data['metadata']['max_danger_level']
                dangers_str = f"столк.: {data['metadata']['collision_warnings']}, аном.: {data['metadata']['anomaly_detections']}, топливо: {data['metadata']['low_fuel_warnings']}"
                print(f"✓ Сохранено | опасностей: {total_warnings} ({dangers_str}) | макс. уровень: {max_d:.2f}")
                
            except Exception as e:
                print(f"✗ Ошибка: {e}")
                import traceback
                traceback.print_exc()
            
            seed += 1
            
            if delay_between_runs > 0:
                time.sleep(delay_between_runs)
    
    except KeyboardInterrupt:
        print(f"\n\nОстановлено пользователем.")
        print(f"Всего сгенерировано файлов: {file_count}")
        print(f"Последний сид: {seed - 1}")
        print(f"Данные сохранены в: {output_path.absolute()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Бесконечный генератор данных A.R.I.E.S")
    parser.add_argument("--output-dir", type=str, default="generated_data", help="Директория для сохранения")
    parser.add_argument("--num-steps", type=int, default=120, help="Шагов на симуляцию")
    parser.add_argument("--start-seed", type=int, default=0, help="Начальный сид")
    parser.add_argument("--format", type=str, default="json", choices=["json", "csv"], help="Формат файлов")
    parser.add_argument("--max-files", type=int, default=None, help="Максимум файлов (None = бесконечно)")
    parser.add_argument("--delay", type=float, default=0.1, help="Задержка между симуляциями (сек)")
    parser.add_argument("--no-dangers-file", action="store_true", help="Не сохранять отдельные файлы опасностей в папку dangers/")
    parser.add_argument("--full", action="store_true", help="Полная симуляция с нейросетями (PyTorch); по умолчанию — лёгкий режим без PyTorch")
    
    args = parser.parse_args()
    
    generate_data_continuously(
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        start_seed=args.start_seed,
        save_format=args.format,
        max_files=args.max_files,
        delay_between_runs=args.delay,
        save_dangers_file=not args.no_dangers_file,
        use_light=not args.full,
    )

