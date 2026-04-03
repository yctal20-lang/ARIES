"""
A.R.I.E.S Web Dashboard Launcher.
Запуск веб-дашборда в стиле AetherOS.

Использование:
    python run_web_dashboard.py

После запуска откройте в браузере: http://127.0.0.1:5000/
С этого ПК: http://127.0.0.1:5000/  или  http://localhost:5000/
С других устройств в той же сети: http://<IP этого ПК>:5000/
Если порт 5000 занят, попробуйте: python run_web_dashboard.py --port 5001
"""

import argparse
import os
import sys
from pathlib import Path

# Add space_debris_ai to path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    from space_debris_ai.visualization.web_server import run_server
except ModuleNotFoundError as e:
    if "flask" in str(e).lower():
        print("Flask is not installed for this Python.")
        print("Install it with:  python -m pip install flask")
        print("(Use the same 'python' you use to run this script.)")
        sys.exit(1)
    raise

def _dlog(msg, data=None, hypothesis_id=None):
    import json
    import time
    try:
        with open("debug-17e329.log", "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "17e329", "message": msg, "data": data or {}, "hypothesisId": hypothesis_id, "timestamp": int(time.time() * 1000), "location": "run_web_dashboard.py"}) + "\n")
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A.R.I.E.S Web Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0 — доступ с других устройств)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument(
        "--arduino-port",
        default=None,
        metavar="COMx",
        help="Serial-порт Arduino (Windows: COM3, COM8, …). Либо задайте ARDUINO_PORT, либо файл space_debris_ai/arduino_bridge/arduino_port.txt",
    )
    parser.add_argument("--no-debug", action="store_true", help="Disable Flask debug mode")
    args = parser.parse_args()
    if args.arduino_port:
        os.environ["ARDUINO_PORT"] = str(args.arduino_port).strip()
    # #region agent log
    _dlog("launch_start", {"host": args.host, "port": args.port}, "H1")
    # #endregion
    try:
        run_server(host=args.host, port=args.port, debug=not args.no_debug)
    except Exception as e:
        # #region agent log
        _dlog("launch_error", {"error": str(e), "type": type(e).__name__}, "H5")
        # #endregion
        raise
    # #region agent log
    _dlog("launch_exited", {}, "H5")
    # #endregion
