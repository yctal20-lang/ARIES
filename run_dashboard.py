"""
A.R.I.E.S — запуск дашборда Space Debris Collector AI из корня проекта.

Команды для вывода:
  cd C:\\Users\\Asus\\Desktop\\CURSOR
  python run_dashboard.py

Или из папки space_debris_ai:
  cd CURSOR\\space_debris_ai
  python run_dashboard.py

Откроется окно с дашбордом и сохранится mission_dashboard.png в текущей папке.
"""
import sys
from pathlib import Path

# Добавить папку space_debris_ai в путь
_project_root = Path(__file__).resolve().parent
_space_debris_path = _project_root / "space_debris_ai"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from space_debris_ai.visualization.dashboard import run_dashboard

if __name__ == "__main__":
    run_dashboard(num_steps=120, seed=42, save_path="mission_dashboard.png")
