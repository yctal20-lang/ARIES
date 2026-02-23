"""
Visualization: matplotlib dashboard and optional web dashboard.
"""

from .dashboard import run_dashboard

# run_server (Flask) not imported here so the package works without Flask.
# Use: from space_debris_ai.visualization.web_server import run_server

__all__ = ["run_dashboard"]
