"""
Background Serial reader for Arduino (ard.ino sketch).

Typical Serial block (every ~200 ms):
    HALL: DETECTED | HALL: CLEAR
    MAGNETIC: DETECTED | MAGNETIC: CLEAR
    (legacy) Magnetic field detected | Clear space — выставляет и hall, и magnetic
    Distance to object: 144.0 km
    Distance: 45.2 cm   |   Ultrasonic: 12.0 cm   (HC-SR04, поле distance_cm)
    Temperature: 24.5 °C
    Vibration detected! | No vibration
    ---------------

Also supports combined Temperature + Humidity on one line (legacy sketch).
Each parsed block is written to JSON logs (daily rolling + latest snapshot):
    <this_dir>/arduino_YYYY-MM-DD.jsonl  — daily rolling NDJSON log
    <this_dir>/latest.json               — latest reading snapshot
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Save logs into the user's Arduino project folder
# (so you can inspect them while debugging sensors)
_SENSOR_LOGS_DIR = (
    Path(__file__).resolve().parents[2]
    / "ARIES"
    / "space_debris_ai"
    / "arduino"
)
_SENSOR_LOGS_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
_latest: Dict[str, Any] = {
    "distance_km": None,
    "distance_cm": None,
    "magnetic": None,
    "hall": None,
    "temperature": None,
    "humidity": None,
    "vibration": None,
    "raw_block": "",
    "updated_at": None,
    "error": None,
    "seq": 0,
}

_reader_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()

_RE_DISTANCE = re.compile(r"Distance to object\s*:\s*([\d.]+)\s*km", re.IGNORECASE)
# HC-SR04 и аналоги обычно печатают сантиметры
_RE_DISTANCE_CM = re.compile(
    r"(?:Distance to object|Distance|Ultrasonic|HC-SR04)\s*:\s*([\d.]+)\s*cm\b",
    re.IGNORECASE,
)
_RE_DISTANCE_CM_LOOSE = re.compile(r"\b([\d.]+)\s*cm\b", re.IGNORECASE)
_RE_TEMP_HUM = re.compile(
    r"Temperature:\s*([\d.]+)\s*°?\s*C.*Humidity:\s*([\d.]+)\s*%", re.IGNORECASE
)
# ° optional; \b after C can fail on some Unicode builds — fallback below
_RE_TEMP_ONLY = re.compile(r"Temperature:\s*([\d.]+)\s*°?\s*C", re.IGNORECASE)
_RE_TEMP_LOOSE = re.compile(r"Temperature\s*:\s*([\d.]+)", re.IGNORECASE)
_RE_HALL_TAG = re.compile(r"^\s*hall:\s*(detected|clear)\s*$", re.IGNORECASE)
_RE_MAGNETIC_TAG = re.compile(r"^\s*magnetic:\s*(detected|clear)\s*$", re.IGNORECASE)

# Один порт в строке (например COM8). Не зависит от переменных окружения процесса Flask.
_CONFIG_PORT_FILE = Path(__file__).resolve().parent / "arduino_port.txt"


def _port_from_config_file() -> Optional[str]:
    try:
        if not _CONFIG_PORT_FILE.is_file():
            return None
        for raw in _CONFIG_PORT_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            return line
    except Exception:
        pass
    return None


def _find_arduino_port() -> Optional[str]:
    """
    Auto-detect Arduino port (Windows COM or /dev/ttyUSB*).

    We do 2 passes:
    1) keyword match on description/manufacturer
    2) lightweight open+read probe (search for known Arduino substrings)
    """
    try:
        import serial  # type: ignore
        import serial.tools.list_ports
    except Exception:
        return None

    try:
        baud_probe = int(os.environ.get("ARDUINO_BAUD", "9600"))
    except (ValueError, TypeError):
        baud_probe = 9600

    ports = []
    try:
        ports = list(serial.tools.list_ports.comports())
    except Exception:
        ports = []

    # Pass 1: keyword match
    for p in ports:
        desc = (p.description or "").lower()
        mfg = (p.manufacturer or "").lower()
        if any(k in desc for k in ("arduino", "ch340", "cp210", "usb-serial", "ftdi")):
            return p.device
        if any(k in mfg for k in ("arduino", "wch", "silicon", "ftdi")):
            return p.device

    # Pass 2: probe by reading a couple lines
    probe_substrings = (
        "distance to object",
        " cm",
        "ultrasonic",
        "magnetic field",
        "magnetic:",
        "hall:",
        "clear space",
        "vibration detected",
        "no vibration",
        "temperature",
    )

    # Keep it bounded to avoid long stalls
    for p in ports[:12]:
        dev = p.device
        try:
            with serial.Serial(dev, baud_probe, timeout=0.25) as ser:
                # Discard any stale bytes so we don't miss the next packet
                try:
                    ser.reset_input_buffer()
                except Exception:
                    pass

                # Read a few lines and look for known substrings.
                # Need a bit more time because Arduino prints ~200ms blocks.
                deadline = time.time() + 1.4
                while time.time() < deadline:
                    raw = ser.readline()
                    if not raw:
                        continue
                    try:
                        line = raw.decode("utf-8", errors="ignore").strip().lower()
                    except Exception:
                        continue
                    for s in probe_substrings:
                        if s in line:
                            return dev
        except Exception:
            continue

    return None


def get_port() -> Optional[str]:
    env = (os.environ.get("ARDUINO_PORT") or "").strip()
    if env:
        return env
    f = _port_from_config_file()
    if f:
        return f
    return _find_arduino_port()


def get_baud() -> int:
    try:
        return int(os.environ.get("ARDUINO_BAUD", "9600"))
    except (ValueError, TypeError):
        return 9600


def _write_sensor_log(snapshot: Dict[str, Any]) -> None:
    """Append one reading to a daily JSONL log and overwrite latest.json."""
    try:
        ts = datetime.fromtimestamp(snapshot["updated_at"])
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        date_str = ts.strftime("%Y-%m-%d")

        dist = snapshot.get("distance_km")
        dist_cm = snapshot.get("distance_cm")
        mag = snapshot.get("magnetic")
        hall = snapshot.get("hall")
        temp = snapshot.get("temperature")
        hum = snapshot.get("humidity")
        vib = snapshot.get("vibration")

        payload = {
            "timestamp": ts_str,
            "distance_km": dist,
            "distance_cm": dist_cm,
            "magnetic": mag,
            "hall": hall,
            "temperature_c": temp,
            "humidity_pct": hum,
            "vibration": vib,
            "seq": snapshot.get("seq"),
            "updated_at": snapshot.get("updated_at"),
        }

        daily_file = _SENSOR_LOGS_DIR / f"arduino_{date_str}.jsonl"
        with open(daily_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        latest_file = _SENSOR_LOGS_DIR / "latest.json"
        with open(latest_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception:
        pass


def _parse_block(lines: list[str]) -> None:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Разделитель пакета в блок не попадает; на всякий случай пропускаем строки только из дефисов
        if re.match(r"^-{3,}$", stripped):
            continue

        low = stripped.lower()
        mh = _RE_HALL_TAG.match(stripped)
        if mh:
            with _lock:
                _latest["hall"] = mh.group(1).lower() == "detected"
            continue
        mm = _RE_MAGNETIC_TAG.match(stripped)
        if mm:
            with _lock:
                _latest["magnetic"] = mm.group(1).lower() == "detected"
            continue

        # Старый скетч: одна строка на оба смысла (датчик на пине Холла)
        if "magnetic field" in low and "detected" in low:
            with _lock:
                _latest["magnetic"] = True
                _latest["hall"] = True
            continue
        if "clear space" in low:
            with _lock:
                _latest["magnetic"] = False
                _latest["hall"] = False
            continue

        m = _RE_DISTANCE.search(stripped)
        if m:
            with _lock:
                _latest["distance_km"] = float(m.group(1))
            continue

        m = _RE_DISTANCE_CM.search(stripped)
        if not m:
            m = _RE_DISTANCE_CM_LOOSE.search(stripped)
        if m:
            with _lock:
                _latest["distance_cm"] = float(m.group(1))
            continue

        m = _RE_TEMP_HUM.search(stripped)
        if m:
            with _lock:
                _latest["temperature"] = float(m.group(1))
                _latest["humidity"] = float(m.group(2))
            continue

        m = _RE_TEMP_ONLY.search(stripped)
        if not m:
            m = _RE_TEMP_LOOSE.search(stripped)
        if m:
            with _lock:
                _latest["temperature"] = float(m.group(1))
            continue

        if "temperature sensor error" in low or low.strip() == "sensor error":
            with _lock:
                _latest["temperature"] = None
            continue

        if "vibration detected" in low or "micro-impact" in low:
            with _lock:
                _latest["vibration"] = True
            continue
        if "no vibration" in low or "stable flight" in low:
            with _lock:
                _latest["vibration"] = False
            continue

    with _lock:
        _latest["raw_block"] = "\n".join(lines)
        _latest["updated_at"] = time.time()
        _latest["seq"] = int(_latest.get("seq") or 0) + 1
        snapshot = dict(_latest)

    _write_sensor_log(snapshot)


def _reader_loop(port: str, baud: int) -> None:
    import serial as pyserial

    block: list[str] = []
    while not _stop_event.is_set():
        try:
            with pyserial.Serial(port, baud, timeout=0.08) as ser:
                with _lock:
                    _latest["error"] = None
                while not _stop_event.is_set():
                    raw = ser.readline()
                    if not raw:
                        continue
                    try:
                        line = raw.decode("utf-8", errors="replace").strip()
                    except Exception:
                        continue
                    if line.startswith("---") or line == "---------------":
                        if block:
                            _parse_block(block)
                            block = []
                    else:
                        block.append(line)
        except Exception as e:
            with _lock:
                _latest["error"] = str(e)
            time.sleep(2)


def start(port_override: Optional[str] = None) -> Dict[str, Any]:
    """Start the background reader thread. Returns status dict."""
    global _reader_thread
    port = (port_override or "").strip() or get_port()
    baud = get_baud()
    if _reader_thread is not None and _reader_thread.is_alive():
        return {"ok": True, "already_running": True, "port": port, "baud": baud}
    if not port:
        with _lock:
            _latest["error"] = "No Arduino port found (set ARDUINO_PORT=COMx)"
        return {"ok": False, "error": _latest["error"]}

    _stop_event.clear()
    _reader_thread = threading.Thread(target=_reader_loop, args=(port, baud), daemon=True)
    _reader_thread.start()
    return {"ok": True, "port": port, "baud": baud}


def stop() -> None:
    _stop_event.set()


def _sanitize_for_json(d: Dict[str, Any]) -> Dict[str, Any]:
    """NaN/Inf не сериализуются в строгом JSON — отдаём null."""
    out = dict(d)
    for k, v in list(out.items()):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
    return out


def get_latest() -> Dict[str, Any]:
    with _lock:
        return _sanitize_for_json(dict(_latest))


def status() -> Dict[str, Any]:
    port = get_port()
    baud = get_baud()
    running = _reader_thread is not None and _reader_thread.is_alive()
    pyserial_ok = False
    try:
        import serial  # noqa: F401
        pyserial_ok = True
    except ImportError:
        pass

    with _lock:
        err = _latest.get("error")

    return {
        "port": port,
        "baud": baud,
        "running": running,
        "pyserial_installed": pyserial_ok,
        "error": err,
    }
