"""Flask blueprint: /api/arduino/* — live sensor data from Arduino."""

import json
import time
from pathlib import Path

from flask import Blueprint, Response, jsonify, request, stream_with_context

from space_debris_ai.arduino_bridge import serial_reader

# Keep logs in the user's Arduino project folder
_SENSOR_LOGS_DIR = (
    Path(__file__).resolve().parents[2]
    / "ARIES"
    / "space_debris_ai"
    / "arduino"
)

arduino_bp = Blueprint("arduino", __name__, url_prefix="/api/arduino")


@arduino_bp.get("/status")
def arduino_status():
    return jsonify(serial_reader.status())


@arduino_bp.get("/live")
def arduino_live():
    data = serial_reader.get_latest()
    return jsonify(data)


@arduino_bp.get("/stream")
def arduino_stream():
    """Server-Sent Events: push each new parsed Serial block as JSON (real-time UI)."""

    def generate():
        last_seq = -1
        tick = 0
        while True:
            data = serial_reader.get_latest()
            seq = data.get("seq")
            ts = data.get("updated_at")
            if ts is not None and seq is not None and seq != last_seq:
                last_seq = seq
                yield "data: " + json.dumps(data) + "\n\n"
            tick += 1
            if tick % 500 == 0:
                yield ": ping\n\n"
            time.sleep(0.012)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@arduino_bp.post("/start")
def arduino_start():
    port_override = None
    if request.is_json:
        body = request.get_json(silent=True)
        if isinstance(body, dict):
            port_override = (body.get("port") or body.get("arduino_port") or "").strip() or None
    result = serial_reader.start(port_override=port_override)
    code = 200 if result.get("ok") else 503
    return jsonify(result), code


@arduino_bp.post("/stop")
def arduino_stop():
    serial_reader.stop()
    return jsonify({"ok": True})


@arduino_bp.get("/logs")
def arduino_logs_index():
    """List available daily log files in sensor_logs/."""
    files = []
    if _SENSOR_LOGS_DIR.exists():
        for f in sorted(_SENSOR_LOGS_DIR.glob("arduino_*.txt"), reverse=True):
            stat = f.stat()
            files.append({
                "filename": f.name,
                "size_bytes": stat.st_size,
                "modified": stat.st_mtime,
            })
    return jsonify({"logs_dir": str(_SENSOR_LOGS_DIR), "files": files})


@arduino_bp.get("/logs/latest")
def arduino_logs_latest():
    """Return the latest.txt snapshot as plain text."""
    latest_file = _SENSOR_LOGS_DIR / "latest.txt"
    if not latest_file.exists():
        return jsonify({"error": "No data yet — Arduino not connected or no readings saved"}), 404
    content = latest_file.read_text(encoding="utf-8")
    return Response(content, mimetype="text/plain")


@arduino_bp.get("/logs/<filename>")
def arduino_logs_file(filename: str):
    """Return last N lines of a daily log file.
    Query params: ?lines=100 (default 100).
    """
    if not filename.endswith(".txt") or "/" in filename or "\\" in filename:
        return jsonify({"error": "Invalid filename"}), 400
    log_file = _SENSOR_LOGS_DIR / filename
    if not log_file.exists():
        return jsonify({"error": "File not found"}), 404
    try:
        from flask import request as _req
        n = int(_req.args.get("lines", 100))
    except (ValueError, TypeError):
        n = 100
    n = max(1, min(n, 5000))
    all_lines = log_file.read_text(encoding="utf-8").splitlines()
    tail = all_lines[-n:]
    return jsonify({
        "filename": filename,
        "total_lines": len(all_lines),
        "returned_lines": len(tail),
        "lines": tail,
    })
