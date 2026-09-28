"""
Optical package re-exports for M4 Change Detection.
"""

from m4_changedetect.optical.ndvi import (
    compute_ndvi,
    compute_ndvi_difference,
    compute_mean_absolute_ndvi_delta,
)
from m4_changedetect.optical.ndwi import (
    compute_ndwi,
    compute_ndwi_difference,
    compute_mean_absolute_ndwi_delta,
)
from m4_changedetect.optical.optical_analyzer import analyze_optical

__all__ = [
    "compute_ndvi",
    "compute_ndvi_difference",
    "compute_mean_absolute_ndvi_delta",
    "compute_ndwi",
    "compute_ndwi_difference",
    "compute_mean_absolute_ndwi_delta",
    "analyze_optical",
]
