"""
Deterministic SAR feature extraction.
Never calls any neural model. Reads VV/VH bands with rasterio.
"""
import logging
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling

from shared.config import get_config

logger = logging.getLogger(__name__)


class UniformError(Exception):
    """Uniform error for M4 module."""
    def __init__(self, module: str, error_type: str, message: str):
        self.module = module
        self.error_type = error_type
        self.message = message
        super().__init__(f"[{module}] {error_type}: {message}")


def _read_band_as_db(dataset: rasterio.DatasetReader, band_idx: int) -> np.ndarray:
    """Read a band and convert to dB scale (10 * log10)."""
    data = dataset.read(band_idx).astype(np.float32)
    # Convert to dB: 10 * log10(power), avoid log(0)
    data = np.maximum(data, 1e-10)
    return 10.0 * np.log10(data)


def _get_band_indices(dataset: rasterio.DatasetReader) -> tuple[int, int, str]:
    """
    Determine VV and VH band indices from dataset metadata.
    Returns (vv_idx, vh_idx, notes).
    """
    notes = ""
    descriptions = dataset.descriptions if dataset.descriptions else []
    
    # Try to find VV/VH from band descriptions
    vv_idx = None
    vh_idx = None
    for i, desc in enumerate(descriptions, start=1):
        if desc and "vv" in desc.lower():
            vv_idx = i
        elif desc and "vh" in desc.lower():
            vh_idx = i
    
    # Fallback: assume band 1 = VV, band 2 = VH (common Sentinel-1 GRD order)
    if vv_idx is None:
        vv_idx = 1
        notes += "Assumed band 1 = VV; "
    if vh_idx is None:
        vh_idx = 2 if dataset.count >= 2 else 1
        notes += "Assumed band 2 = VH; "
    
    if vv_idx == vh_idx and dataset.count >= 2:
        vh_idx = 2 if vv_idx == 1 else 1
        notes += "Adjusted VH to different band; "
    
    return vv_idx, vh_idx, notes


def extract_sar_features(sar_path: str, reference_path: str | None = None) -> dict[str, Any]:
    """
    Deterministic only. Never calls any neural model.
    
    Returns: {"water_fraction": float, "builtup_fraction": float,
              "log_ratio_mean": float, "mean_vv_db": float, "mean_vh_db": float,
              "notes": str}
    
    Reads VV/VH with rasterio (bands named/passed per config; if band order is
    ambiguous, read metadata and record the assumption in "notes").
    water: pixels with VV dB < sar_water_max_vv_db.
    built-up: VV > sar_builtup_min_vv_db AND VH > sar_builtup_min_vh_db.
    log_ratio: vs reference_path if given else vs scene spatial mean.
    Raises UniformError(module="M4", error_type="invalid_input") on unreadable file.
    """
    config = get_config()
    
    # Get thresholds from config
    water_max_vv_db = config.get("change", {}).get("sar_water_max_vv_db", -17.0)
    builtup_min_vv_db = config.get("change", {}).get("sar_builtup_min_vv_db", -10.0)
    builtup_min_vh_db = config.get("change", {}).get("sar_builtup_min_vh_db", -20.0)
    
    # Open SAR image
    try:
        with rasterio.open(sar_path) as ds:
            if ds.count < 1:
                raise UniformError(module="M4", error_type="invalid_input", 
                                 message=f"SAR image has no bands: {sar_path}")
            
            vv_idx, vh_idx, notes = _get_band_indices(ds)
            
            # Read VV band
            if vv_idx > ds.count:
                raise UniformError(module="M4", error_type="invalid_input",
                                 message=f"VV band index {vv_idx} exceeds band count {ds.count}")
            vv_db = _read_band_as_db(ds, vv_idx)
            
            # Read VH band if available
            if vh_idx <= ds.count and vh_idx != vv_idx:
                vh_db = _read_band_as_db(ds, vh_idx)
            else:
                # If only one band, use VV for VH too (with note)
                vh_db = vv_db.copy()
                notes += "Single-band SAR; VH copied from VV; "
            
            # Compute water fraction
            water_mask = vv_db < water_max_vv_db
            water_fraction = float(np.mean(water_mask))
            
            # Compute built-up fraction
            builtup_mask = (vv_db > builtup_min_vv_db) & (vh_db > builtup_min_vh_db)
            builtup_fraction = float(np.mean(builtup_mask))
            
            # Compute mean VV/VH dB
            mean_vv_db = float(np.mean(vv_db))
            mean_vh_db = float(np.mean(vh_db))
            
            # Compute log ratio
            if reference_path:
                try:
                    with rasterio.open(reference_path) as ref_ds:
                        ref_vv_idx, ref_vh_idx, ref_notes = _get_band_indices(ref_ds)
                        notes += ref_notes
                        ref_vv_db = _read_band_as_db(ref_ds, ref_vv_idx)
                        # Resample if shapes differ
                        if ref_vv_db.shape != vv_db.shape:
                            ref_vv_db = ref_ds.read(ref_vv_idx, out_shape=vv_db.shape, 
                                                     resampling=Resampling.bilinear).astype(np.float32)
                            ref_vv_db = np.maximum(ref_vv_db, 1e-10)
                            ref_vv_db = 10.0 * np.log10(ref_vv_db)
                        log_ratio = vv_db - ref_vv_db
                except Exception as e:
                    logger.warning(f"Failed to read reference SAR for log-ratio: {e}")
                    log_ratio = vv_db - np.mean(vv_db)
                    notes += "Reference read failed; used scene mean; "
            else:
                log_ratio = vv_db - np.mean(vv_db)
            
            log_ratio_mean = float(np.mean(log_ratio))
            
    except rasterio.RasterioIOError as e:
        raise UniformError(module="M4", error_type="invalid_input",
                         message=f"Cannot read SAR file {sar_path}: {e}")
    except UniformError:
        raise
    except Exception as e:
        raise UniformError(module="M4", error_type="invalid_input",
                         message=f"SAR feature extraction failed: {e}")
    
    return {
        "water_fraction": water_fraction,
        "builtup_fraction": builtup_fraction,
        "log_ratio_mean": log_ratio_mean,
        "mean_vv_db": mean_vv_db,
        "mean_vh_db": mean_vh_db,
        "notes": notes.strip(),
    }