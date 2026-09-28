"""
Deterministic Stage 3 explanation generator module for M4 Change Detection.

Produces deterministic, rule-based natural language change descriptions from
bi-temporal observation statistics and sensor metrics.
Focused strictly on text construction. Does NOT contain workflow-level fallback decisions,
external network/API calls, or LLM calls.
"""

from typing import Any, Dict, Optional
import numpy as np


def _format_signed_float(val: float) -> str:
    """Helper to format a float with explicit sign e.g. +0.25 or -0.15."""
    return f"{val:+.2f}"


def generate_change_description(
    change_score: Optional[float],
    optical_results: Optional[Dict[str, Any]] = None,
    sar_results: Optional[Dict[str, Any]] = None,
    metadata_t1: Optional[Dict[str, Any]] = None,
    metadata_t2: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generates a deterministic Stage 3 natural language change description.

    Converts computed change scores and sensor metrics (NDVI, NDWI, SAR) into a concise,
    factual summary describing observed temporal changes without making unsupported causal claims.

    :param change_score: Computed scalar change score in [0.0, 1.0] (or None).
    :param optical_results: Analysis dictionary returned by optical_analyzer (or None).
    :param sar_results: Analysis dictionary returned by sar_analyzer (or None).
    :param metadata_t1: Optional observation metadata for t1.
    :param metadata_t2: Optional observation metadata for t2.
    :return: Deterministic natural language description string.
    """
    if change_score is None and not optical_results and not sar_results:
        return "Insufficient data provided for temporal change description."

    # 1. Categorize Change Score Magnitude
    score_val = float(change_score) if change_score is not None else 0.0
    score_val = max(0.0, min(1.0, score_val))

    if score_val < 0.05:
        magnitude_str = "no significant observed temporal change"
    elif score_val < 0.30:
        magnitude_str = "minor observed temporal change"
    elif score_val < 0.70:
        magnitude_str = "moderate observed temporal change"
    else:
        magnitude_str = "significant observed temporal change"

    parts = [f"Bi-temporal analysis indicates {magnitude_str} (change score: {score_val:.4f})."]

    # 2. Extract Optical Metrics (NDVI & NDWI)
    optical_parts = []
    if optical_results and isinstance(optical_results, dict):
        # NDVI
        if "ndvi_diff" in optical_results and optical_results["ndvi_diff"] is not None:
            diff_arr = np.asarray(optical_results["ndvi_diff"])
            if diff_arr.size > 0:
                mean_signed_ndvi = float(np.mean(diff_arr))
                if mean_signed_ndvi > 0.05:
                    optical_parts.append(f"increase in vegetation index (NDVI delta: {_format_signed_float(mean_signed_ndvi)})")
                elif mean_signed_ndvi < -0.05:
                    optical_parts.append(f"decrease in vegetation index (NDVI delta: {_format_signed_float(mean_signed_ndvi)})")
                else:
                    optical_parts.append(f"stable vegetation index (NDVI delta: {_format_signed_float(mean_signed_ndvi)})")
        elif "mean_abs_ndvi_delta" in optical_results and optical_results["mean_abs_ndvi_delta"] is not None:
            abs_delta = float(optical_results["mean_abs_ndvi_delta"])
            if abs_delta > 0.05:
                optical_parts.append(f"observed vegetation index variation (NDVI abs delta: {abs_delta:.2f})")
            else:
                optical_parts.append("stable vegetation index (NDVI)")

        # NDWI
        has_ndwi = optical_results.get("has_ndwi", False)
        if has_ndwi and "ndwi_diff" in optical_results and optical_results["ndwi_diff"] is not None:
            diff_arr = np.asarray(optical_results["ndwi_diff"])
            if diff_arr.size > 0:
                mean_signed_ndwi = float(np.mean(diff_arr))
                if mean_signed_ndwi > 0.05:
                    optical_parts.append(f"increase in water index (NDWI delta: {_format_signed_float(mean_signed_ndwi)})")
                elif mean_signed_ndwi < -0.05:
                    optical_parts.append(f"decrease in water index (NDWI delta: {_format_signed_float(mean_signed_ndwi)})")
                else:
                    optical_parts.append(f"stable water index (NDWI delta: {_format_signed_float(mean_signed_ndwi)})")
        elif has_ndwi and "mean_abs_ndwi_delta" in optical_results and optical_results["mean_abs_ndwi_delta"] is not None:
            abs_delta = float(optical_results["mean_abs_ndwi_delta"])
            if abs_delta > 0.05:
                optical_parts.append(f"observed water index variation (NDWI abs delta: {abs_delta:.2f})")
            else:
                optical_parts.append("stable water index (NDWI)")

    if optical_parts:
        parts.append(f"Sentinel-2 optical metrics show {', '.join(optical_parts)}.")

    # 3. Extract SAR Metrics
    sar_parts = []
    if sar_results and isinstance(sar_results, dict):
        if "sar_log_ratio" in sar_results and sar_results["sar_log_ratio"] is not None:
            ratio_arr = np.asarray(sar_results["sar_log_ratio"])
            if ratio_arr.size > 0:
                mean_signed_sar = float(np.mean(ratio_arr))
                if mean_signed_sar > 0.05:
                    sar_parts.append(f"increase in SAR backscatter intensity (log-ratio delta: {_format_signed_float(mean_signed_sar)})")
                elif mean_signed_sar < -0.05:
                    sar_parts.append(f"decrease in SAR backscatter intensity (log-ratio delta: {_format_signed_float(mean_signed_sar)})")
                else:
                    sar_parts.append(f"stable SAR backscatter intensity (log-ratio delta: {_format_signed_float(mean_signed_sar)})")
        elif "mean_abs_sar_delta" in sar_results and sar_results["mean_abs_sar_delta"] is not None:
            abs_delta = float(sar_results["mean_abs_sar_delta"])
            if abs_delta > 0.05:
                sar_parts.append(f"observed SAR backscatter variation (log-ratio abs delta: {abs_delta:.2f})")
            else:
                sar_parts.append("stable SAR backscatter intensity")

    if sar_parts:
        parts.append(f"Sentinel-1 SAR metrics show {', '.join(sar_parts)}.")

    return " ".join(parts)
