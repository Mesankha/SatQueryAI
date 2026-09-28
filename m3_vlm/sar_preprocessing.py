"""
M3 SAR Preprocessing — Handle Sentinel-1 SAR imagery properly.

Sentinel-1 SAR characteristics:
  - Single polarization or dual (VV, VH)
  - Linear power scale (very wide dynamic range)
  - Speckle noise (multiplicative)
  - Different visual semantics:
      * VV (co-pol): sensitive to surface roughness, structure
      * VH (cross-pol): sensitive to volume scattering (vegetation)
      * VV/VH ratio: helps distinguish water/urban/vegetation

Standard preprocessing pipeline (per scientific literature):
  1. Radiometric calibration (already done in GRD product)
  2. Speckle filtering (Lee filter, 5x5 window)
  3. Log transform (compress dynamic range to RGB-like 0-255)
  4. Normalize to 0-255 uint8
  5. Stack as pseudo-RGB: [VV, VH, VV/VH] or [VV, VV, VV] for single-pol

This module handles:
  - GeoTIFF SAR reading (any band count)
  - Speckle filtering (Lee filter implementation)
  - Log transform + normalization
  - Pseudo-RGB channel composition
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAR-specific image transformations
# ---------------------------------------------------------------------------

def to_db(x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Convert linear power to decibels: 10*log10(x). Compresses dynamic range."""
    return 10.0 * np.log10(np.maximum(x, eps))


def from_db(x_db: np.ndarray) -> np.ndarray:
    """Convert dB back to linear: 10^(x/10)."""
    return np.power(10.0, x_db / 10.0)


def lee_filter(image: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Lee speckle filter for SAR imagery.

    Reference: Lee, J.-S. (1980). Digital Image Enhancement and Noise Filtering
    by Use of Local Statistics. IEEE Trans. PAMI, 2(2), 165-168.

    Args:
        image: 2D SAR amplitude/power array
        window_size: Filter window (odd, default 5)

    Returns:
        Filtered image, same shape
    """
    if not SCIPY_AVAILABLE:
        logger.warning("scipy not available, skipping Lee filter")
        return image

    if window_size % 2 == 0:
        window_size += 1

    # Convert to dB for filtering
    img_db = to_db(image)

    # Local mean and variance using uniform filter
    mean = ndimage.uniform_filter(img_db, size=window_size)
    sqr_mean = ndimage.uniform_filter(img_db ** 2, size=window_size)
    var = sqr_mean - mean ** 2

    # Noise variance (for amplitude SAR, ~1/(2*N_looks))
    # For Sentinel-1 GRD, equivalent number of looks is ~4
    n_looks = 4.0
    noise_var = 1.0 / n_looks

    # Lee filter weight
    weight = var / (var + noise_var) if (var + noise_var).any() else 0.0
    weight = np.clip(weight, 0, 1)

    # Filtered image in dB
    filtered_db = mean + weight * (img_db - mean)

    # Convert back to linear
    return from_db(filtered_db)


def enhanced_lee_filter(image: np.ndarray, window_size: int = 7,
                       damping: float = 2.0) -> np.ndarray:
    """
    Enhanced Lee filter (Lopes et al. 1990).
    Preserves edges and point targets better than basic Lee.
    """
    if not SCIPY_AVAILABLE:
        return image

    if window_size % 2 == 0:
        window_size += 1

    img_db = to_db(image)
    mean = ndimage.uniform_filter(img_db, size=window_size)
    sqr_mean = ndimage.uniform_filter(img_db ** 2, size=window_size)
    var = np.maximum(sqr_mean - mean ** 2, 0)

    cu = 1.0 / np.sqrt(2.0)  # coefficient of variation for pure noise (amplitude)
    cv = np.sqrt(np.maximum(var, 0)) / (np.abs(mean) + 1e-10)  # local CV

    # Three zones:
    # 1. Homogeneous (cv < cu): use mean
    # 2. Heterogeneous (cu <= cv < sqrt(damping)*cu): Lee weighted
    # 3. Heterogeneous extreme (cv >= sqrt(damping)*cu): preserve original
    cu_max = np.sqrt(damping) * cu

    weight = np.where(
        cv < cu, 0.0,
        np.where(
            cv < cu_max,
            (cv ** 2 - cu ** 2) / (cv ** 2 * (1 + cu ** 2)),
            1.0,
        ),
    )

    filtered_db = mean + weight * (img_db - mean)
    return from_db(filtered_db)


def normalize_to_uint8(image: np.ndarray, percentile_clip: float = 99.5) -> np.ndarray:
    """Normalize to 0-255 uint8 with percentile clipping for robust contrast."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    lo = np.percentile(finite, 100 - percentile_clip)
    hi = np.percentile(finite, percentile_clip)
    if hi - lo < 1e-6:
        hi = lo + 1.0

    clipped = np.clip(image, lo, hi)
    normalized = (clipped - lo) / (hi - lo) * 255.0
    return normalized.astype(np.uint8)


def detect_sar_bands(arr_stack: np.ndarray) -> dict:
    """
    Auto-detect Sentinel-1 polarization bands in a stacked array.

    Sentinel-1 GRD conventions:
      - band 1: VV intensity
      - band 2: VH intensity (if dual-pol)
      - or band 1: HH, band 2: HV (if HH+HV)

    For our purposes we just need to identify which channels look like
    which polarization. Heuristics:
      - VV usually stronger than VH (because cross-pol is weaker)
      - so the brighter band is likely VV
    """
    n_bands = arr_stack.shape[0]
    mean_intensity = [arr_stack[i].mean() for i in range(n_bands)]
    sorted_bands = np.argsort(mean_intensity)[::-1]  # brightest first

    result = {
        "n_bands": n_bands,
        "mean_intensity": mean_intensity,
        "brightest_first": sorted_bands.tolist(),
    }

    if n_bands == 1:
        result["interpretation"] = "single-pol (likely VV or HH)"
    elif n_bands == 2:
        result["interpretation"] = "dual-pol (likely VV+VH or HH+HV)"
        result["vv_idx"] = int(sorted_bands[0])  # brighter
        result["vh_idx"] = int(sorted_bands[1])  # dimmer
    elif n_bands >= 3:
        result["interpretation"] = f"{n_bands}-band (likely RGB or composite)"

    return result


def detect_modality(filename: str) -> str:
    """
    Detect the modality of an image from its filename.

    Returns:
        "sar" if filename suggests SAR
        "optical" if filename suggests optical RGB
        "fusion" if filename suggests paired data
        "unknown" otherwise
    """
    fn = filename.lower()

    # SAR indicators
    if any(kw in fn for kw in [
        "sar", "s1_", "sentinel-1", "sentinel1", "grd",
        "_vv_", "_vh_", "pc_sar_", "hf_sar_"
    ]):
        return "sar"

    # Fusion indicators (contains both)
    if "fusion" in fn or "paired" in fn or "combined" in fn:
        return "fusion"

    # Optical indicators
    if any(kw in fn for kw in [
        "s2_", "sentinel-2", "sentinel2", "optical", "rgb",
        "ms_", "msi_", "pc_opt_", "bigearthnet"
    ]):
        return "optical"

    return "optical"  # default to optical for unknown


def sar_to_pseudo_rgb(
    image_path: str,
    target_size: int = 504,  # GeoChat's native input resolution
    apply_lee_filter: bool = True,
    lee_window: int = 5,
    percentile_clip: float = 99.5,
) -> Tuple[Image.Image, dict]:
    """
    Convert a Sentinel-1 SAR image to a pseudo-RGB image suitable for GeoChat.

    Channel composition for dual-pol (VV, VH):
      - R: VV (co-pol, surface roughness)
      - G: VH (cross-pol, volume scattering)
      - B: VV/VH ratio (texture indicator)

    For single-pol: replicate to 3 channels.

    Returns:
        (PIL.Image, metadata dict with preprocessing details)
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL required for SAR preprocessing")

    # Read the file
    arr_stack = _load_sar_geotiff(image_path)
    band_info = detect_sar_bands(arr_stack)

    # Apply speckle filter
    if apply_lee_filter and band_info["n_bands"] >= 1:
        filtered = np.stack([
            enhanced_lee_filter(arr_stack[i], window_size=lee_window)
            for i in range(arr_stack.shape[0])
        ])
    else:
        filtered = arr_stack

    # Convert to dB for better dynamic range
    filtered_db = to_db(filtered)

    # Compose pseudo-RGB
    if filtered_db.shape[0] == 1:
        # Single polarization: replicate
        r = g = b = filtered_db[0]
    elif filtered_db.shape[0] == 2:
        # Dual-pol: VV, VH, ratio
        vv_idx = band_info.get("vv_idx", 0)
        vh_idx = band_info.get("vh_idx", 1)
        r = filtered_db[vv_idx]
        g = filtered_db[vh_idx]
        # VV/VH ratio (in dB, this is subtraction)
        ratio = filtered_db[vv_idx] - filtered_db[vh_idx]
        # Normalize ratio to similar range as VV, VH
        ratio_centered = ratio + 10.0  # shift to be positive
        b = ratio_centered
    else:
        # 3+ bands: use first 3
        r, g, b = filtered_db[0], filtered_db[1], filtered_db[2]

    # Normalize each channel to 0-255
    r_uint8 = normalize_to_uint8(r, percentile_clip)
    g_uint8 = normalize_to_uint8(g, percentile_clip)
    b_uint8 = normalize_to_uint8(b, percentile_clip)

    # Stack to RGB
    rgb = np.stack([r_uint8, g_uint8, b_uint8], axis=-1)

    # Resize to GeoChat's expected size
    pil_img = Image.fromarray(rgb, mode="RGB")
    if max(pil_img.size) != target_size:
        pil_img = pil_img.resize(
            (target_size, target_size), Image.BILINEAR  # BILINEAR for SAR (preserves backscatter)
        )

    metadata = {
        "source_path": str(image_path),
        "bands_detected": band_info,
        "preprocessing": {
            "lee_filter": apply_lee_filter,
            "lee_window": lee_window if apply_lee_filter else None,
            "log_transform": True,
            "percentile_clip": percentile_clip,
            "target_size": target_size,
            "channel_composition": "VV/VH/ratio" if filtered.shape[0] == 2 else "replicated",
        },
    }
    return pil_img, metadata


def _load_sar_geotiff(image_path: str) -> np.ndarray:
    """
    Load a SAR GeoTIFF as a stack of 2D arrays (bands, H, W).
    Tries rasterio first (for multi-band GeoTIFFs), then PIL as fallback.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"SAR image not found: {image_path}")

    if RASTERIO_AVAILABLE:
        try:
            with rasterio.open(image_path) as src:
                bands = src.count
                if bands == 1:
                    arr = src.read(1)
                    return arr[np.newaxis, ...]  # (1, H, W)
                else:
                    return np.stack([src.read(i + 1) for i in range(bands)])
        except Exception as e:
            logger.debug(f"rasterio failed for {image_path}: {e}")

    # Fallback: PIL (only works for single-band TIFFs)
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL and rasterio both unavailable")
    img = Image.open(image_path)
    arr = np.array(img)
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    return arr  # assume (H, W, C) -> convert below

    # If shape is (H, W, C), transpose
    if arr.ndim == 3:
        return arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...]


def save_sar_as_png(sar_path: str, png_path: str, target_size: int = 504) -> dict:
    """Convert SAR to PNG and save. Returns preprocessing metadata."""
    pil_img, metadata = sar_to_pseudo_rgb(sar_path, target_size=target_size)
    pil_img.save(png_path, "PNG")
    return metadata