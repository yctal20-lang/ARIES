"""
Visualization: matplotlib dashboard and optional web dashboard.
"""

# Lazy imports to avoid pulling matplotlib when only the web server is needed.
# Use: from space_debris_ai.visualization.dashboard import run_dashboard
# Use: from space_debris_ai.visualization.web_server import run_server

__all__ = ["run_dashboard"]


def __getattr__(name):
    if name == "run_dashboard":
        from .dashboard import run_dashboard
        return run_dashboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
