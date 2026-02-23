"""
A.R.I.E.S — запуск дашборда из папки space_debris_ai.

Команда для вывода:
  python run_dashboard.py

Дашборд: A.R.I.E.S Mission Control, ORBIT TRACK, TELEMETRY, FUSION, RESOURCES.
Отображает опасные ситуации: столкновения, аномалии, низкое топливо.
Сохранение: mission_dashboard.png в текущей папке.
"""
import sys
import argparse
from pathlib import Path

# Добавить родительскую папку в путь, чтобы пакет space_debris_ai был найден
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from space_debris_ai.visualization.dashboard import run_dashboard

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A.R.I.E.S Mission Dashboard")
    parser.add_argument("--num-steps", type=int, default=120, help="Number of simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-path", type=str, default="mission_dashboard.png", help="Path to save dashboard")
    parser.add_argument("--show", action="store_true", help="Show dashboard window")
    
    args = parser.parse_args()
    
    run_dashboard(
        num_steps=args.num_steps,
        seed=args.seed,
        save_path=args.save_path,
        show=args.show,
    )
