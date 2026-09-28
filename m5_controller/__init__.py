"""
M5 Agentic Controller - dispatches queries to M0 (catalog), M1 (parser),
M2 (retrieval), M3 (GeoChat), and M4 (change detection).
"""
# Apply the M2 wiring patch (must be imported before dispatch_table)
from . import m2_wiring_patch  # noqa: F401