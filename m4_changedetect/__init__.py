"""
M4 - CHANGE DETECTION & SCORING Package.

Main module re-exports. Exposes main detect_change interface and canonical ChangeResult schema.

NOTE: This module's directory is `m4_changedetect` (lowercase, no underscore)
for consistency with other modules (m0_catalog, m3_vlm, m5_controller, m6_api).
The original code referenced `m4_changedetect` (lowercase, no underscore)
which causes ModuleNotFoundError on case-sensitive filesystems (Linux).
"""
import os
import sys

# Ensure we use the correct module name
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from m4_changedetect.change_detector import ChangeDetector, detect_change
from m4_changedetect.schemas.change_schema import ChangeResult

__all__ = [
    "ChangeDetector",
    "detect_change",
    "ChangeResult",
]