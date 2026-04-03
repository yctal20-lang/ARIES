"""
Background Serial reader for Arduino (ard.ino sketch).

Typical Serial block (every ~200 ms):
    Magnetic field detected | Clear space
    Distance to object: 144.0 km
    Temperature: 24.5 °C
    Vibration detected! | No vibration
    ---------------

Also supports combined Temperature + Humidity on one line (legacy sketch).
Each parsed block is written to a TXT log directory (daily rolling + latest snapshot):
    <this_dir>/arduino_YYYY-MM-DD.txt  — daily rolling log
    <this_dir>/latest.txt              — latest reading snapshot
"""

from __future__ import annotations

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
    "magnetic": None,
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
_RE_TEMP_HUM = re.compile(
    r"Temperature:\s*([\d.]+)\s*°?\s*C.*Humidity:\s*([\d.]+)\s*%", re.IGNORECASE
)
# ° optional; \b after C can fail on some Unicode builds — fallback below
_RE_TEMP_ONLY = re.compile(r"Temperature:\s*([\d.]+)\s*°?\s*C", re.IGNORECASE)
_RE_TEMP_LOOSE = re.compile(r"Temperature\s*:\s*([\d.]+)", re.IGNORECASE)

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
        "magnetic field",
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
    """Append one reading to the daily log and overwrite latest.txt."""
    try:
        ts = datetime.fromtimestamp(snapshot["updated_at"])
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        date_str = ts.strftime("%Y-%m-%d")

        dist = snapshot.get("distance_km")
        mag = snapshot.get("magnetic")
        temp = snapshot.get("temperature")
        hum = snapshot.get("humidity")
        vib = snapshot.get("vibration")

        dist_s = f"{dist:.1f} km" if dist is not None else "—"
        mag_s = ("DETECTED" if mag else "clear") if mag is not None else "—"
        temp_s = f"{temp:.1f} °C" if temp is not None else "—"
        hum_s = f"{hum:.1f} %" if hum is not None else "—"
        vib_s = ("YES" if vib else "no") if vib is not None else "—"

        line = (
            f"[{ts_str}] "
            f"dist={dist_s} | mag={mag_s} | temp={temp_s} | "
            f"hum={hum_s} | vib={vib_s}\n"
        )

        daily_file = _SENSOR_LOGS_DIR / f"arduino_{date_str}.txt"
        with open(daily_file, "a", encoding="utf-8") as f:
            f.write(line)

        latest_file = _SENSOR_LOGS_DIR / "latest.txt"
        with open(latest_file, "w", encoding="utf-8") as f:
            f.write(f"Timestamp : {ts_str}\n")
            f.write(f"Distance  : {dist_s}\n")
            f.write(f"Magnetic  : {mag_s}\n")
            f.write(f"Temp      : {temp_s}\n")
            f.write(f"Humidity  : {hum_s}\n")
            f.write(f"Vibration : {vib_s}\n")
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

        if "magnetic field" in stripped.lower():
            with _lock:
                _latest["magnetic"] = True
            continue
        if "clear space" in stripped.lower():
            with _lock:
                _latest["magnetic"] = False
            continue

        m = _RE_DISTANCE.search(stripped)
        if m:
            with _lock:
                _latest["distance_km"] = float(m.group(1))
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

        low = stripped.lower()
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
