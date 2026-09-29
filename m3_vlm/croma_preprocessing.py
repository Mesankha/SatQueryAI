"""
SAR Preprocessing for CROMA-S1.
Converts GeoTIFF SAR imagery to CROMA input format:
- Calibrated dB scale (gamma-naught)
- 120x120 tiles
- 2 channels (VV, VH) - handles RISAT single-pol by duplication
- Normalization: per-channel mean ± 2σ → float 0-1 (SatMAE convention)
"""
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch

from shared.config import get_config

logger = logging.getLogger(__name__)


# CROMA input specifications
CROMA_TILE_SIZE = 120
CROMA_PATCH_SIZE = 16
# 120/16 = 7.5 -> 7 patches per dimension (floor) = 49 patches
# CROMA likely uses padding or different tiling; actual num_patches depends on model
CROMA_NUM_PATCHES = (CROMA_TILE_SIZE // CROMA_PATCH_SIZE) ** 2  # 49 for 120x120


def _read_band_as_db(dataset: rasterio.DatasetReader, band_idx: int) -> np.ndarray:
    """Read a band and convert to dB scale (10 * log10)."""
    data = dataset.read(band_idx).astype(np.float32)
    # Convert to dB: 10 * log10(power), avoid log(0)
    data = np.maximum(data, 1e-10)
    return 10.0 * np.log10(data)


def _get_sar_band_indices(dataset: rasterio.DatasetReader) -> Tuple[int, int, str]:
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


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    """
    Normalize channel using per-channel mean ± 2σ, rescaled to 0-1.
    SatMAE/SeCo convention for SAR.
    """
    mean = float(np.mean(channel))
    std = float(np.std(channel))
    
    # Clip to mean ± 2σ
    lower = mean - 2 * std
    upper = mean + 2 * std
    channel = np.clip(channel, lower, upper)
    
    # Rescale to 0-1
    if upper > lower:
        channel = (channel - lower) / (upper - lower)
    else:
        channel = np.zeros_like(channel)
    
    # Clamp to exact [0, 1] to avoid floating point edge cases
    channel = np.clip(channel, 0.0, 1.0)
    
    return channel.astype(np.float32)


def _resize_to_croma_tile(image: np.ndarray, target_size: int = CROMA_TILE_SIZE) -> np.ndarray:
    """Resize image to CROMA tile size using rasterio Resampling."""
    if image.shape[-2:] == (target_size, target_size):
        return image
    
    # Use rasterio for proper geospatial resampling
    H, W = image.shape[-2:]
    resampled = np.zeros((target_size, target_size), dtype=np.float32)
    
    # Simple bilinear resize using numpy/scipy approach
    from scipy.ndimage import zoom
    zoom_h = target_size / H
    zoom_w = target_size / W
    resampled = zoom(image, (zoom_h, zoom_w), order=1)
    
    return resampled.astype(np.float32)


def preprocess_sar_for_croma(
    sar_path: str,
    tile_size: int = CROMA_TILE_SIZE,
    return_metadata: bool = False
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    """
    Preprocess SAR GeoTIFF for CROMA encoder.
    
    Args:
        sar_path: Path to SAR GeoTIFF file
        tile_size: Target tile size (default 120 for CROMA)
        return_metadata: If True, return preprocessing metadata
        
    Returns:
        Tensor of shape (1, 2, tile_size, tile_size) - batch, channels (VV, VH), H, W
        Optional metadata dict with preprocessing info
    """
    metadata = {
        "source_path": sar_path,
        "tile_size": tile_size,
        "normalization": "per_channel_mean_2sigma_0to1",
        "notes": "",
        "assumptions": [],
    }
    
    try:
        with rasterio.open(sar_path) as ds:
            if ds.count < 1:
                raise ValueError(f"SAR image has no bands: {sar_path}")
            
            # Get band indices
            vv_idx, vh_idx, band_notes = _get_sar_band_indices(ds)
            metadata["notes"] += band_notes
            metadata["band_indices"] = {"VV": vv_idx, "VH": vh_idx}
            
            # Read VV band
            if vv_idx > ds.count:
                raise ValueError(f"VV band index {vv_idx} exceeds band count {ds.count}")
            vv_db = _read_band_as_db(ds, vv_idx)
            
            # Read VH band if available
            if vh_idx <= ds.count and vh_idx != vv_idx:
                vh_db = _read_band_as_db(ds, vh_idx)
                metadata["polarization"] = "dual_pol"
            else:
                # Single polarization (e.g., RISAT) - duplicate VV for VH
                vh_db = vv_db.copy()
                metadata["polarization"] = "single_pol_duplicated"
                metadata["assumptions"].append("Single-pol SAR: VH channel duplicated from VV")
                metadata["notes"] += "Single-band SAR; VH copied from VV; "
            
            # Store original stats for reference
            metadata["original_stats"] = {
                "vv_mean_db": float(np.mean(vv_db)),
                "vv_std_db": float(np.std(vv_db)),
                "vh_mean_db": float(np.mean(vh_db)),
                "vh_std_db": float(np.std(vh_db)),
            }
            
            # Normalize each channel independently (SatMAE convention)
            vv_norm = _normalize_channel(vv_db)
            vh_norm = _normalize_channel(vh_db)
            
            metadata["normalized_stats"] = {
                "vv_mean": float(np.mean(vv_norm)),
                "vv_std": float(np.std(vv_norm)),
                "vh_mean": float(np.mean(vh_norm)),
                "vh_std": float(np.std(vh_norm)),
            }
            
            # Resize to CROMA tile size
            vv_resized = _resize_to_croma_tile(vv_norm, tile_size)
            vh_resized = _resize_to_croma_tile(vh_norm, tile_size)
            
            # Stack channels: (2, H, W) -> (1, 2, H, W) for batch
            sar_tensor = np.stack([vv_resized, vh_resized], axis=0)  # (2, 120, 120)
            sar_tensor = sar_tensor[np.newaxis, ...]  # (1, 2, 120, 120)
            
            # Convert to torch tensor
            tensor = torch.from_numpy(sar_tensor).float()
            
            metadata["output_shape"] = list(tensor.shape)
            metadata["croma_ready"] = True
            
            return tensor, metadata if return_metadata else None
            
    except rasterio.RasterioIOError as e:
        raise ValueError(f"Cannot read SAR file {sar_path}: {e}")
    except Exception as e:
        raise ValueError(f"SAR preprocessing failed: {e}")


def preprocess_sar_tile_for_croma(
    vv_data: np.ndarray,
    vh_data: Optional[np.ndarray] = None,
    tile_size: int = CROMA_TILE_SIZE
) -> torch.Tensor:
    """
    Preprocess already-loaded SAR arrays for CROMA.
    Assumes data is already in dB scale (gamma-naught).
    
    Args:
        vv_data: VV band as dB values (H, W)
        vh_data: VH band as dB values (H, W), or None to duplicate VV
        tile_size: Target tile size
        
    Returns:
        Tensor of shape (1, 2, tile_size, tile_size)
    """
    # Normalize
    vv_norm = _normalize_channel(vv_data)
    
    if vh_data is not None:
        vh_norm = _normalize_channel(vh_data)
        polarization = "dual_pol"
    else:
        vh_norm = vv_norm.copy()
        polarization = "single_pol_duplicated"
    
    # Resize
    vv_resized = _resize_to_croma_tile(vv_norm, tile_size)
    vh_resized = _resize_to_croma_tile(vh_norm, tile_size)
    
    # Stack and batch
    sar_tensor = np.stack([vv_resized, vh_resized], axis=0)[np.newaxis, ...]
    return torch.from_numpy(sar_tensor).float()


def validate_croma_input(tensor: torch.Tensor) -> Tuple[bool, str]:
    """
    Validate tensor meets CROMA input requirements.
    Returns (is_valid, error_message).
    """
    if tensor.dim() != 4:
        return False, f"Expected 4D tensor (B, C, H, W), got {tensor.dim()}D"
    
    B, C, H, W = tensor.shape
    if C != 2:
        return False, f"Expected 2 channels (VV, VH), got {C}"
    
    if H != CROMA_TILE_SIZE or W != CROMA_TILE_SIZE:
        return False, f"Expected {CROMA_TILE_SIZE}x{CROMA_TILE_SIZE}, got {H}x{W}"
    
    if torch.isnan(tensor).any():
        return False, "Tensor contains NaN values"
    
    if torch.isinf(tensor).any():
        return False, "Tensor contains Inf values"
    
    # Check range (should be 0-1 after normalization)
    if tensor.min() < -0.1 or tensor.max() > 1.1:
        return False, f"Tensor values outside expected [0,1] range: [{tensor.min():.3f}, {tensor.max():.3f}]"
    
    return True, "Valid"