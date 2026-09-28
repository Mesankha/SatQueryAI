"""
Pure Sentinel-2 NDVI (Normalized Difference Vegetation Index) calculation module.

Focused strictly on mathematical computation of NDVI and NDVI temporal differences.
Does NOT contain workflow-level fallback decisions, user explanations, or LLM calls.
"""

from typing import Union
import numpy as np


def compute_ndvi(nir: Union[np.ndarray, list, float], red: Union[np.ndarray, list, float]) -> np.ndarray:
    """
    Computes Normalized Difference Vegetation Index (NDVI) from Sentinel-2 NIR and Red band data.

    Formula:
        NDVI = (NIR - Red) / (NIR + Red)

    Requirements:
        - Supports NumPy arrays (1D, 2D, or multi-dimensional).
        - Returns float NumPy array preserving shape.
        - Zero denominators (NIR + Red == 0) are handled safely without producing NaN or Inf (result = 0.0).
        - Input arrays are preserved (not mutated in-place).

    :param nir: Near-Infrared band data (NumPy array or array-like).
    :param red: Red band data (NumPy array or array-like).
    :return: NumPy float array containing computed NDVI values.
    :raises ValueError: If NIR and Red input array shapes mismatch.
    """
    nir_arr = np.asarray(nir, dtype=np.float64)
    red_arr = np.asarray(red, dtype=np.float64)

    if nir_arr.shape != red_arr.shape:
        raise ValueError(
            f"Shape mismatch in compute_ndvi: NIR shape {nir_arr.shape} does not match Red shape {red_arr.shape}."
        )

    numerator = nir_arr - red_arr
    denominator = nir_arr + red_arr

    # Safe division: where denominator is zero, result is 0.0 without producing NaN or Inf
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(denominator != 0.0, numerator / denominator, 0.0)

    # Clean any residual NaN or Inf values
    ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=1.0, neginf=-1.0)
    return ndvi


def compute_ndvi_difference(ndvi_t1: Union[np.ndarray, list, float], ndvi_t2: Union[np.ndarray, list, float]) -> np.ndarray:
    """
    Computes temporal difference between NDVI at t1 and t2.

    Formula:
        delta_ndvi = ndvi_t2 - ndvi_t1

    :param ndvi_t1: NDVI array at observation t1.
    :param ndvi_t2: NDVI array at observation t2.
    :return: NumPy float array representing pixel-wise delta NDVI.
    :raises ValueError: If ndvi_t1 and ndvi_t2 shapes mismatch.
    """
    arr_t1 = np.asarray(ndvi_t1, dtype=np.float64)
    arr_t2 = np.asarray(ndvi_t2, dtype=np.float64)

    if arr_t1.shape != arr_t2.shape:
        raise ValueError(
            f"Shape mismatch in compute_ndvi_difference: ndvi_t1 shape {arr_t1.shape} does not match ndvi_t2 shape {arr_t2.shape}."
        )

    delta_ndvi = arr_t2 - arr_t1
    return delta_ndvi


def compute_mean_absolute_ndvi_delta(ndvi_t1: Union[np.ndarray, list, float], ndvi_t2: Union[np.ndarray, list, float]) -> float:
    """
    Computes the mean absolute delta between NDVI at t1 and t2.
    Utilized by the M4 temporal change-scoring pipeline.

    Formula:
        mean_abs_delta = mean(|ndvi_t2 - ndvi_t1|)

    :param ndvi_t1: NDVI array at observation t1.
    :param ndvi_t2: NDVI array at observation t2.
    :return: Float scalar representing the mean absolute change.
    """
    delta = compute_ndvi_difference(ndvi_t1, ndvi_t2)
    return float(np.mean(np.abs(delta)))
