"""
Change scoring and normalization module for M4 Change Detection.

Normalizes raw change maps and combines multi-sensor optical (NDVI + NDWI)
and SAR analysis into a canonical normalized scalar change_score in range [0.0, 1.0].
Focused strictly on mathematical score computation. Does NOT contain workflow-level fallback decisions,
user-facing explanation generation, or LLM calls.
"""

from typing import Any, Dict, Optional, Union
import numpy as np


def normalize_change_map(change_map: Union[np.ndarray, list, float]) -> np.ndarray:
    """
    Normalizes a raw spatial change map or array to scale range [0.0, 1.0].

    Since maximum theoretical index delta is 2.0 (range [-1.0, 1.0]), absolute
    deltas are scaled by 2.0 and clipped to [0.0, 1.0].

    :param change_map: Raw change magnitude map or delta array.
    :return: NumPy float64 array normalized to range [0.0, 1.0].
    """
    if change_map is None:
        return np.array([], dtype=np.float64)

    arr = np.abs(np.asarray(change_map, dtype=np.float64))
    if arr.size == 0:
        return arr

    # Max index delta magnitude is 2.0
    normalized = np.clip(arr / 2.0, 0.0, 1.0)
    return normalized


def calculate_change_score(
    optical_results: Optional[Dict[str, Any]] = None,
    sar_results: Optional[Dict[str, Any]] = None
) -> Optional[float]:
    """
    Computes a canonical, deterministic scalar normalized change_score in [0.0, 1.0].

    Normalizes NDVI and NDWI mean absolute deltas (scaled by max delta 2.0),
    handles NDVI-only and combined NDVI+NDWI optical measurements, and clamps
    the final score strictly within [0.0, 1.0].

    :param optical_results: Analysis dictionary returned by optical_analyzer (or None).
    :param sar_results: Optional SAR analysis dictionary (or None).
    :return: Scalar change_score in [0.0, 1.0], or None if inputs are missing/empty.
    """
    optical_score: Optional[float] = None
    sar_score: Optional[float] = None

    # --- 1. Compute Optical Score ---
    if optical_results and isinstance(optical_results, dict):
        ndvi_delta: Optional[float] = None
        ndwi_delta: Optional[float] = None

        # Extract NDVI delta
        if "mean_abs_ndvi_delta" in optical_results and optical_results["mean_abs_ndvi_delta"] is not None:
            ndvi_delta = float(optical_results["mean_abs_ndvi_delta"])
        elif "ndvi_diff" in optical_results and optical_results["ndvi_diff"] is not None:
            diff_arr = np.asarray(optical_results["ndvi_diff"])
            if diff_arr.size > 0:
                ndvi_delta = float(np.mean(np.abs(diff_arr)))

        # Extract NDWI delta
        has_ndwi = optical_results.get("has_ndwi", False)
        if has_ndwi and "mean_abs_ndwi_delta" in optical_results and optical_results["mean_abs_ndwi_delta"] is not None:
            ndwi_delta = float(optical_results["mean_abs_ndwi_delta"])
        elif has_ndwi and "ndwi_diff" in optical_results and optical_results["ndwi_diff"] is not None:
            diff_arr = np.asarray(optical_results["ndwi_diff"])
            if diff_arr.size > 0:
                ndwi_delta = float(np.mean(np.abs(diff_arr)))

        # Compute normalized sub-scores [0.0, 1.0] (max index delta = 2.0)
        score_ndvi: Optional[float] = min(max(ndvi_delta / 2.0, 0.0), 1.0) if ndvi_delta is not None else None
        score_ndwi: Optional[float] = min(max(ndwi_delta / 2.0, 0.0), 1.0) if ndwi_delta is not None else None

        if score_ndvi is not None and score_ndwi is not None:
            optical_score = 0.5 * score_ndvi + 0.5 * score_ndwi
        elif score_ndvi is not None:
            optical_score = score_ndvi
        elif score_ndwi is not None:
            optical_score = score_ndwi

    # --- 2. Compute SAR Score ---
    if sar_results and isinstance(sar_results, dict):
        if "mean_abs_sar_delta" in sar_results and sar_results["mean_abs_sar_delta"] is not None:
            sar_delta = float(sar_results["mean_abs_sar_delta"])
            # SAR log ratio max scale ~ 2.0
            sar_score = min(max(sar_delta / 2.0, 0.0), 1.0)
        elif "sar_log_ratio" in sar_results and sar_results["sar_log_ratio"] is not None:
            ratio_arr = np.asarray(sar_results["sar_log_ratio"])
            if ratio_arr.size > 0:
                sar_delta = float(np.mean(np.abs(ratio_arr)))
                sar_score = min(max(sar_delta / 2.0, 0.0), 1.0)

    # --- 3. Combine Sensor Scores & Clamp ---
    if optical_score is not None and sar_score is not None:
        composite = 0.5 * optical_score + 0.5 * sar_score
    elif optical_score is not None:
        composite = optical_score
    elif sar_score is not None:
        composite = sar_score
    else:
        return None

    # Strict clamping to [0.0, 1.0]
    final_score = min(max(composite, 0.0), 1.0)
    return round(float(final_score), 4)
