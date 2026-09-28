"""
Pure SAR (Synthetic Aperture Radar) log-ratio calculation module.

Focused strictly on mathematical computation of Sentinel-1 SAR intensity log-ratios.
Does NOT contain workflow-level fallback decisions, user explanations, or LLM calls.
"""

from typing import Union
import numpy as np


def compute_sar_log_ratio(
    sar_t1: Union[np.ndarray, list, float],
    sar_t2: Union[np.ndarray, list, float],
    eps: float = 1e-7
) -> np.ndarray:
    """
    Computes logarithmic intensity ratio between Sentinel-1 SAR images at t1 and t2.

    Formula:
        log_ratio = log10((sar_t2 + eps) / (sar_t1 + eps))

    Requirements:
        - Supports NumPy arrays (1D, 2D, or multi-dimensional).
        - Returns float NumPy array preserving shape.
        - Zero or negative intensity values are handled safely (clipped to non-negative) to prevent log10(0) or NaN.
        - Numerical stability ensured via epsilon addition (eps=1e-7).
        - Input arrays are preserved (not mutated in-place).

    :param sar_t1: SAR intensity/backscatter data at observation t1.
    :param sar_t2: SAR intensity/backscatter data at observation t2.
    :param eps: Epsilon value to prevent division by zero or log10(0) (default 1e-7).
    :return: NumPy float array containing signed log-ratio values.
    :raises ValueError: If sar_t1 and sar_t2 array shapes mismatch.
    """
    arr_t1 = np.clip(np.asarray(sar_t1, dtype=np.float64), 0.0, None)
    arr_t2 = np.clip(np.asarray(sar_t2, dtype=np.float64), 0.0, None)

    if arr_t1.shape != arr_t2.shape:
        raise ValueError(
            f"Shape mismatch in compute_sar_log_ratio: sar_t1 shape {arr_t1.shape} does not match sar_t2 shape {arr_t2.shape}."
        )

    # Safe log-ratio formula: log10((sar_t2 + eps) / (sar_t1 + eps))
    ratio = (arr_t2 + eps) / (arr_t1 + eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log10(ratio)

    # Clean residual non-finite values if any
    log_ratio = np.nan_to_num(log_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    return log_ratio


def compute_sar_absolute_ratio(
    sar_t1: Union[np.ndarray, list, float],
    sar_t2: Union[np.ndarray, list, float],
    eps: float = 1e-7
) -> np.ndarray:
    """
    Computes absolute logarithmic change intensity for SAR imagery.

    Formula:
        abs_log_ratio = |log10((sar_t2 + eps) / (sar_t1 + eps))|

    :param sar_t1: SAR intensity data at observation t1.
    :param sar_t2: SAR intensity data at observation t2.
    :param eps: Epsilon value for numerical stability.
    :return: NumPy float array containing non-negative absolute log-ratio values.
    """
    signed = compute_sar_log_ratio(sar_t1, sar_t2, eps=eps)
    return np.abs(signed)


def compute_mean_absolute_sar_delta(
    sar_t1: Union[np.ndarray, list, float],
    sar_t2: Union[np.ndarray, list, float],
    eps: float = 1e-7
) -> float:
    """
    Computes the mean absolute log ratio delta between SAR observations at t1 and t2.
    Utilized by the M4 temporal change-scoring pipeline.

    Formula:
        mean_abs_sar_delta = mean(|log10((sar_t2 + eps) / (sar_t1 + eps))|)

    :param sar_t1: SAR intensity data at observation t1.
    :param sar_t2: SAR intensity data at observation t2.
    :param eps: Epsilon value for numerical stability.
    :return: Float scalar representing the mean absolute SAR change.
    """
    abs_log = compute_sar_absolute_ratio(sar_t1, sar_t2, eps=eps)
    return float(np.mean(abs_log))
