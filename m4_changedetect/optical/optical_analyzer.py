"""
Sentinel-2 Optical Change Analysis Orchestration Module.

Orchestrates temporal Sentinel-2 optical change analysis by combining NDVI
and NDWI temporal computations. Delegates all mathematical calculations to
ndvi.py and ndwi.py without formula duplication or workflow-level fallback logic.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

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


def _extract_band(image: Any, keys: Tuple[str, ...]) -> Optional[np.ndarray]:
    """
    Helper function to extract a band array from dictionary or object attributes.

    :param image: Input image dictionary or object.
    :param keys: Tuple of possible key/attribute names for the band.
    :return: NumPy array if found, else None.
    """
    if isinstance(image, dict):
        for k in keys:
            if k in image and image[k] is not None:
                return np.asarray(image[k])
            # Case-insensitive key check
            for dict_key in image.keys():
                if dict_key.lower() == k.lower() and image[dict_key] is not None:
                    return np.asarray(image[dict_key])
    else:
        for attr in keys:
            if hasattr(image, attr) and getattr(image, attr) is not None:
                return np.asarray(getattr(image, attr))
    return None


def analyze_optical(
    image_t1: Any,
    image_t2: Any,
    metadata_t1: Optional[Dict[str, Any]] = None,
    metadata_t2: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Orchestrates Sentinel-2 optical change analysis across t1 and t2 observations.

    Combines NDVI and NDWI computations using pure mathematical functions in
    ndvi.py and ndwi.py. Returns a structured internal dictionary appropriate
    for consumption by the M4 scoring layer.

    :param image_t1: Optical image / bands container at observation t1.
    :param image_t2: Optical image / bands container at observation t2.
    :param metadata_t1: Optional observation metadata for t1.
    :param metadata_t2: Optional observation metadata for t2.
    :return: Dictionary containing internal optical metrics and difference layers:
             - ndvi_t1, ndvi_t2, ndvi_diff, mean_abs_ndvi_delta
             - ndwi_t1, ndwi_t2, ndwi_diff, mean_abs_ndwi_delta (or None if Green band missing)
             - has_ndwi
    :raises ValueError: If image inputs are missing required bands ('nir', 'red') or if shapes mismatch.
    """
    if image_t1 is None or image_t2 is None:
        raise ValueError("Both image_t1 and image_t2 must be provided for optical analysis.")

    # Extract required bands for NDVI
    nir_keys = ("nir", "NIR", "b8", "B8", "near_infrared")
    red_keys = ("red", "RED", "b4", "B4")
    green_keys = ("green", "GREEN", "b3", "B3")

    nir_t1 = _extract_band(image_t1, nir_keys)
    red_t1 = _extract_band(image_t1, red_keys)

    nir_t2 = _extract_band(image_t2, nir_keys)
    red_t2 = _extract_band(image_t2, red_keys)

    if nir_t1 is None or red_t1 is None or nir_t2 is None or red_t2 is None:
        raise ValueError("Missing required optical band data ('nir' or 'red') for NDVI analysis.")

    # Compute NDVI metrics via ndvi.py
    ndvi_t1 = compute_ndvi(nir_t1, red_t1)
    ndvi_t2 = compute_ndvi(nir_t2, red_t2)
    ndvi_diff = compute_ndvi_difference(ndvi_t1, ndvi_t2)
    mean_abs_ndvi_delta = compute_mean_absolute_ndvi_delta(ndvi_t1, ndvi_t2)

    # Extract optional Green band for NDWI
    green_t1 = _extract_band(image_t1, green_keys)
    green_t2 = _extract_band(image_t2, green_keys)

    ndwi_t1: Optional[np.ndarray] = None
    ndwi_t2: Optional[np.ndarray] = None
    ndwi_diff: Optional[np.ndarray] = None
    mean_abs_ndwi_delta: Optional[float] = None

    if green_t1 is not None and green_t2 is not None:
        ndwi_t1 = compute_ndwi(green_t1, nir_t1)
        ndwi_t2 = compute_ndwi(green_t2, nir_t2)
        ndwi_diff = compute_ndwi_difference(ndwi_t1, ndwi_t2)
        mean_abs_ndwi_delta = compute_mean_absolute_ndwi_delta(ndwi_t1, ndwi_t2)

    return {
        "ndvi_t1": ndvi_t1,
        "ndvi_t2": ndvi_t2,
        "ndvi_diff": ndvi_diff,
        "mean_abs_ndvi_delta": mean_abs_ndvi_delta,
        "ndwi_t1": ndwi_t1,
        "ndwi_t2": ndwi_t2,
        "ndwi_diff": ndwi_diff,
        "mean_abs_ndwi_delta": mean_abs_ndwi_delta,
        "has_ndwi": ndwi_diff is not None,
    }
