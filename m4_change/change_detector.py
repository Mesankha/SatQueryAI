"""
M4 Change Detection - Deterministic optical and SAR change analysis.
No neural models, only rasterio/numpy operations.
"""
import logging
from typing import Any, Optional
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from shared.config import get_config
from shared.schemas import UniformError, ErrorType

logger = logging.getLogger(__name__)


class UniformError(Exception):
    """Uniform error for M4 module."""
    def __init__(self, module: str, error_type: str, message: str):
        self.module = module
        self.error_type = error_type
        self.message = message
        super().__init__(f"[{module}] {error_type}: {message}")


def _read_band_as_float(dataset: rasterio.DatasetReader, band_idx: int) -> np.ndarray:
    """Read a band as float32 array."""
    return dataset.read(band_idx).astype(np.float32)


def _get_band_indices(dataset: rasterio.DatasetReader, band_names: list[str]) -> dict[str, int]:
    """Map band names to indices from dataset descriptions."""
    indices = {}
    descriptions = dataset.descriptions if dataset.descriptions else []
    for i, desc in enumerate(descriptions, start=1):
        if desc:
            for name in band_names:
                if name.lower() in desc.lower():
                    indices[name] = i
    # Fallback to sequential if not found
    for i, name in enumerate(band_names, start=1):
        if name not in indices and i <= dataset.count:
            indices[name] = i
    return indices


def _compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Compute NDVI from NIR and Red bands."""
    denom = nir + red
    denom = np.where(denom == 0, 1e-10, denom)
    return (nir - red) / denom


def _compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute NDWI from Green and NIR bands."""
    denom = green + nir
    denom = np.where(denom == 0, 1e-10, denom)
    return (green - nir) / denom


def _read_optical_bands(path: str) -> dict[str, np.ndarray]:
    """Read optical bands (B2=Blue, B3=Green, B4=Red, B8=NIR) from file."""
    with rasterio.open(path) as ds:
        # Try to identify bands by description
        band_map = _get_band_indices(ds, ["B2", "B3", "B4", "B8", "blue", "green", "red", "nir"])
        
        bands = {}
        for name in ["B2", "B3", "B4", "B8"]:
            idx = band_map.get(name)
            if idx and idx <= ds.count:
                bands[name.lower()] = _read_band_as_float(ds, idx)
            else:
                # Fallback: assume order B2,B3,B4,B8
                fallback_idx = {"B2": 1, "B3": 2, "B4": 3, "B8": 4}[name]
                if fallback_idx <= ds.count:
                    bands[name.lower()] = _read_band_as_float(ds, fallback_idx)
                    logger.warning(f"Band {name} not found in metadata, using fallback index {fallback_idx}")
        
        return bands


def _read_sar_bands(path: str) -> dict[str, np.ndarray]:
    """Read SAR bands (VV, VH) from file."""
    with rasterio.open(path) as ds:
        band_map = _get_band_indices(ds, ["VV", "VH", "vv", "vh"])
        bands = {}
        for name in ["VV", "VH"]:
            idx = band_map.get(name)
            if idx and idx <= ds.count:
                bands[name.lower()] = _read_band_as_float(ds, idx)
            else:
                # Fallback: assume VV=1, VH=2
                fallback_idx = 1 if name == "VV" else 2
                if fallback_idx <= ds.count:
                    bands[name.lower()] = _read_band_as_float(ds, fallback_idx)
                    logger.warning(f"SAR band {name} not found in metadata, using fallback index {fallback_idx}")
        return bands


def detect_optical_change(t1_path: str, t2_path: str) -> dict[str, Any]:
    """
    Detect change between two optical images using NDVI/NDWI.
    
    Returns: {"change_score": float|None, "ndvi_delta": float, "ndwi_delta": float,
              "changed_fraction": float, "change_description": str, "notes": str}
    change_score = min(1.0, changed_fraction); None if either file unreadable.
    """
    config = get_config()
    change_config = config.get("change", {})
    ndvi_threshold = change_config.get("ndvi_threshold", 0.5)
    
    try:
        # Read bands from both images
        t1_bands = _read_optical_bands(t1_path)
        t2_bands = _read_optical_bands(t2_path)
        
        # Compute NDVI for both times
        nir1, red1 = t1_bands.get("b8"), t1_bands.get("b4")
        nir2, red2 = t2_bands.get("b8"), t2_bands.get("b4")
        
        if nir1 is None or red1 is None or nir2 is None or red2 is None:
            return {
                "change_score": None,
                "ndvi_delta": 0.0,
                "ndwi_delta": 0.0,
                "changed_fraction": 0.0,
                "change_description": "Change detection unavailable: missing required NIR/Red bands",
                "notes": "Required bands (B4=Red, B8=NIR) not found in one or both images"
            }
        
        ndvi1 = _compute_ndvi(nir1, red1)
        ndvi2 = _compute_ndvi(nir2, red2)
        ndvi_delta_map = ndvi2 - ndvi1
        ndvi_delta = float(np.mean(np.abs(ndvi_delta_map)))
        
        # Compute NDWI if Green available
        green1, green2 = t1_bands.get("b3"), t2_bands.get("b3")
        if green1 is not None and green2 is not None:
            ndwi1 = _compute_ndwi(green1, nir1)
            ndwi2 = _compute_ndwi(green2, nir2)
            ndwi_delta_map = ndwi2 - ndwi1
            ndwi_delta = float(np.mean(np.abs(ndwi_delta_map)))
        else:
            ndwi_delta = 0.0
        
        # Changed fraction: pixels where |NDVI delta| > threshold
        changed_mask = np.abs(ndvi_delta_map) > ndvi_threshold
        changed_fraction = float(np.mean(changed_mask))
        
        # Change score = min(1.0, changed_fraction)
        change_score = min(1.0, changed_fraction)
        
        # Generate description
        if changed_fraction > 0.3:
            desc = f"Significant change detected: {changed_fraction:.1%} of area shows NDVI shift > {ndvi_threshold}"
        elif changed_fraction > 0.1:
            desc = f"Moderate change detected: {changed_fraction:.1%} of area shows NDVI shift > {ndvi_threshold}"
        elif changed_fraction > 0.01:
            desc = f"Minor change detected: {changed_fraction:.1%} of area shows NDVI shift > {ndvi_threshold}"
        else:
            desc = "Negligible change detected"
        
        return {
            "change_score": change_score,
            "ndvi_delta": ndvi_delta,
            "ndwi_delta": ndwi_delta,
            "changed_fraction": changed_fraction,
            "change_description": desc,
            "notes": f"NDVI threshold={ndvi_threshold}"
        }
        
    except Exception as e:
        logger.error(f"Optical change detection failed: {e}")
        return {
            "change_score": None,
            "ndvi_delta": 0.0,
            "ndwi_delta": 0.0,
            "changed_fraction": 0.0,
            "change_description": f"Change detection failed: {str(e)}",
            "notes": f"Exception: {type(e).__name__}"
        }


def detect_sar_change(t1_path: str, t2_path: str) -> dict[str, Any]:
    """
    Detect change between two SAR images using VV/VH log-ratio.
    
    Returns: {"change_score": float|None, "log_ratio_mean": float, "log_ratio_std": float,
              "changed_fraction": float, "change_description": str, "notes": str}
    """
    config = get_config()
    change_config = config.get("change", {})
    log_ratio_threshold = change_config.get("sar_log_ratio_threshold", 2.0)
    
    try:
        t1_bands = _read_sar_bands(t1_path)
        t2_bands = _read_sar_bands(t2_path)
        
        vv1, vh1 = t1_bands.get("vv"), t1_bands.get("vh")
        vv2, vh2 = t2_bands.get("vv"), t2_bands.get("vh")
        
        if vv1 is None or vv2 is None:
            return {
                "change_score": None,
                "log_ratio_mean": 0.0,
                "log_ratio_std": 0.0,
                "changed_fraction": 0.0,
                "change_description": "Change detection unavailable: missing VV band",
                "notes": "Required VV band not found in one or both images"
            }
        
        # Compute log-ratio for VV (primary for SAR change)
        eps = 1e-10
        log_ratio_vv = np.log10((vv2 + eps) / (vv1 + eps))
        log_ratio_mean = float(np.mean(log_ratio_vv))
        log_ratio_std = float(np.std(log_ratio_vv))
        
        # Changed fraction: pixels where |log-ratio| > threshold
        changed_mask = np.abs(log_ratio_vv) > log_ratio_threshold
        changed_fraction = float(np.mean(changed_mask))
        
        change_score = min(1.0, changed_fraction)
        
        if changed_fraction > 0.3:
            desc = f"Significant SAR change: {changed_fraction:.1%} of area shows log-ratio > {log_ratio_threshold} dB"
        elif changed_fraction > 0.1:
            desc = f"Moderate SAR change: {changed_fraction:.1%} of area shows log-ratio > {log_ratio_threshold} dB"
        elif changed_fraction > 0.01:
            desc = f"Minor SAR change: {changed_fraction:.1%} of area shows log-ratio > {log_ratio_threshold} dB"
        else:
            desc = "Negligible SAR change detected"
        
        return {
            "change_score": change_score,
            "log_ratio_mean": log_ratio_mean,
            "log_ratio_std": log_ratio_std,
            "changed_fraction": changed_fraction,
            "change_description": desc,
            "notes": f"Log-ratio threshold={log_ratio_threshold}"
        }
        
    except Exception as e:
        logger.error(f"SAR change detection failed: {e}")
        return {
            "change_score": None,
            "log_ratio_mean": 0.0,
            "log_ratio_std": 0.0,
            "changed_fraction": 0.0,
            "change_description": f"SAR change detection failed: {str(e)}",
            "notes": f"Exception: {type(e).__name__}"
        }