"""
Sentinel-1 SAR Change Analysis Orchestration Module.

Orchestrates temporal Sentinel-1 Synthetic Aperture Radar (SAR) change analysis.
Delegates all mathematical calculations to log_ratio.py without formula duplication
or workflow-level fallback decisions.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from m4_changedetect.sar.log_ratio import (
    compute_sar_log_ratio,
    compute_sar_absolute_ratio,
    compute_mean_absolute_sar_delta,
)


def _extract_sar_intensity(image: Any) -> Optional[np.ndarray]:
    """
    Helper function to extract SAR intensity array from numpy arrays, dictionaries, or objects.

    :param image: Input image array, dictionary, or object.
    :return: NumPy array if found, else None.
    """
    if image is None:
        return None

    # Case 1: Direct NumPy array or array-like
    if isinstance(image, (np.ndarray, list, float, int)):
        arr = np.asarray(image)
        if arr.size > 0:
            return arr

    # Case 2: Dictionary key lookup
    keys = ("sar", "SAR", "vv", "VV", "vh", "VH", "intensity", "backscatter")
    if isinstance(image, dict):
        for k in keys:
            if k in image and image[k] is not None:
                return np.asarray(image[k])
            for dict_key in image.keys():
                if dict_key.lower() == k.lower() and image[dict_key] is not None:
                    return np.asarray(image[dict_key])
    else:
        # Case 3: Object attribute lookup
        for attr in keys:
            if hasattr(image, attr) and getattr(image, attr) is not None:
                return np.asarray(getattr(image, attr))

    return None


def analyze_sar(
    image_t1: Any,
    image_t2: Any,
    metadata_t1: Optional[Dict[str, Any]] = None,
    metadata_t2: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Orchestrates Sentinel-1 SAR change analysis across t1 and t2 observations.

    Delegates logarithmic intensity ratio calculations to log_ratio.py.
    Returns a structured internal dictionary appropriate for consumption by the
    M4 change scoring layer.

    :param image_t1: SAR intensity image/container at observation t1.
    :param image_t2: SAR intensity image/container at observation t2.
    :param metadata_t1: Optional observation metadata for t1.
    :param metadata_t2: Optional observation metadata for t2.
    :return: Dictionary containing internal SAR change metrics and log-ratio layers:
             - sar_t1, sar_t2, sar_log_ratio, sar_abs_ratio, mean_abs_sar_delta
    :raises ValueError: If image inputs are missing required SAR intensity data or if shapes mismatch.
    """
    if image_t1 is None or image_t2 is None:
        raise ValueError("Both image_t1 and image_t2 must be provided for SAR analysis.")

    sar_t1 = _extract_sar_intensity(image_t1)
    sar_t2 = _extract_sar_intensity(image_t2)

    if sar_t1 is None or sar_t2 is None:
        raise ValueError("Missing required SAR intensity/backscatter data in image inputs.")

    # Compute SAR log-ratio metrics via log_ratio.py functions
    sar_log_ratio = compute_sar_log_ratio(sar_t1, sar_t2)
    sar_abs_ratio = compute_sar_absolute_ratio(sar_t1, sar_t2)
    mean_abs_sar_delta = compute_mean_absolute_sar_delta(sar_t1, sar_t2)

    return {
        "sar_t1": sar_t1,
        "sar_t2": sar_t2,
        "sar_log_ratio": sar_log_ratio,
        "sar_abs_ratio": sar_abs_ratio,
        "mean_abs_sar_delta": mean_abs_sar_delta,
    }
