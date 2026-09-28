"""
Pure Sentinel-2 NDWI (Normalized Difference Water Index) calculation module.

Focused strictly on mathematical computation of NDWI and NDWI temporal differences.
Does NOT contain workflow-level fallback decisions, user explanations, or LLM calls.
"""

from typing import Union
import numpy as np


def compute_ndwi(green: Union[np.ndarray, list, float], nir: Union[np.ndarray, list, float]) -> np.ndarray:
    """
    Computes Normalized Difference Water Index (NDWI) from Sentinel-2 Green and NIR band data.

    Formula:
        NDWI = (Green - NIR) / (Green + NIR)

    Requirements:
        - Supports NumPy arrays (1D, 2D, or multi-dimensional).
        - Returns float NumPy array preserving shape.
        - Zero denominators (Green + NIR == 0) are handled safely without producing NaN or Inf (result = 0.0).
        - Input arrays are preserved (not mutated in-place).

    :param green: Green band data (NumPy array or array-like).
    :param nir: Near-Infrared band data (NumPy array or array-like).
    :return: NumPy float array containing computed NDWI values.
    :raises ValueError: If Green and NIR input array shapes mismatch.
    """
    green_arr = np.asarray(green, dtype=np.float64)
    nir_arr = np.asarray(nir, dtype=np.float64)

    if green_arr.shape != nir_arr.shape:
        raise ValueError(
            f"Shape mismatch in compute_ndwi: Green shape {green_arr.shape} does not match NIR shape {nir_arr.shape}."
        )

    numerator = green_arr - nir_arr
    denominator = green_arr + nir_arr

    # Safe division: where denominator is zero, result is 0.0 without producing NaN or Inf
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = np.where(denominator != 0.0, numerator / denominator, 0.0)

    # Clean any residual NaN or Inf values
    ndwi = np.nan_to_num(ndwi, nan=0.0, posinf=1.0, neginf=-1.0)
    return ndwi


def compute_ndwi_difference(ndwi_t1: Union[np.ndarray, list, float], ndwi_t2: Union[np.ndarray, list, float]) -> np.ndarray:
    """
    Computes temporal difference between NDWI at t1 and t2.

    Formula:
        delta_ndwi = ndwi_t2 - ndwi_t1

    :param ndwi_t1: NDWI array at observation t1.
    :param ndwi_t2: NDWI array at observation t2.
    :return: NumPy float array representing pixel-wise delta NDWI.
    :raises ValueError: If ndwi_t1 and ndwi_t2 shapes mismatch.
    """
    arr_t1 = np.asarray(ndwi_t1, dtype=np.float64)
    arr_t2 = np.asarray(ndwi_t2, dtype=np.float64)

    if arr_t1.shape != arr_t2.shape:
        raise ValueError(
            f"Shape mismatch in compute_ndwi_difference: ndwi_t1 shape {arr_t1.shape} does not match ndwi_t2 shape {arr_t2.shape}."
        )

    delta_ndwi = arr_t2 - arr_t1
    return delta_ndwi


def compute_mean_absolute_ndwi_delta(ndwi_t1: Union[np.ndarray, list, float], ndwi_t2: Union[np.ndarray, list, float]) -> float:
    """
    Computes the mean absolute delta between NDWI at t1 and t2.
    Utilized by the M4 temporal change-scoring pipeline.

    Formula:
        mean_abs_delta = mean(|ndwi_t2 - ndwi_t1|)

    :param ndwi_t1: NDWI array at observation t1.
    :param ndwi_t2: NDWI array at observation t2.
    :return: Float scalar representing the mean absolute water change.
    """
    delta = compute_ndwi_difference(ndwi_t1, ndwi_t2)
    return float(np.mean(np.abs(delta)))
