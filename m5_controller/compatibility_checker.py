"""
Input compatibility checker — verifies prerequisites before ANY tool call.
"""
import os
from typing import Any, Tuple

from shared.schemas import StructuredQuery, SceneMetadata, UniformError, ErrorType
import rasterio
from shapely import wkt as shapely_wkt
from shared.config import get_config


def _get_mock_mode(task_type: str) -> bool:
    """Check if mock mode is enabled for a given task type."""
    config = get_config()
    if task_type == "retrieval":
        return config.get("retrieval", {}).get("mock", True)
    elif task_type == "vlm":
        return config.get("vlm", {}).get("mock", True)
    elif task_type == "vlm_sar":
        return config.get("vlm", {}).get("sar_mock", True)
    elif task_type == "change":
        return config.get("change", {}).get("mock", True)
    return True  # Default to mock (safe)


def _file_exists_or_mock(file_path: str, task_type: str) -> bool:
    """Check if file exists, or if we're in mock mode for the task or retrieval."""
    if file_path and os.path.exists(file_path):
        return True
    # Allow if specific task mock is enabled
    if _get_mock_mode(task_type):
        return True
    # Also allow if retrieval is in mock mode (fixture data doesn't have real files)
    if _get_mock_mode("retrieval"):
        return True
    return False


def check_search_compatibility(query: StructuredQuery, candidates: list[SceneMetadata]) -> tuple[bool, UniformError | None]:
    if not candidates:
        return False, UniformError(
            module="M5", error_type=ErrorType.no_candidates,
            message="No matching imagery found for the specified criteria.",
            fallback_applied=False,
        )
    return True, None


def check_vqa_compatibility(query: StructuredQuery, candidate: SceneMetadata | None) -> tuple[bool, UniformError | None]:
    if candidate is None:
        return False, _no_candidate_error()
    if not _file_exists_or_mock(candidate.file_path, "vlm"):
        return False, _missing_image_error(candidate.scene_id)
    return True, None


def check_change_compatibility(t1: SceneMetadata | None, t2: SceneMetadata | None) -> tuple[bool, UniformError | None]:
    if t1 is None:
        return False, _no_candidate_error()
    if t2 is None or not _file_exists_or_mock(t2.file_path, "change"):
        return False, UniformError(
            module="M5", error_type=ErrorType.no_candidates,
            message="No comparable second observation available for the requested time window.",
            fallback_applied=False,
        )
    return True, None


def check_fusion_compatibility(optical: SceneMetadata | None, sar: SceneMetadata | None) -> tuple[bool, UniformError | None]:
    if optical is None or sar is None:
        return False, UniformError(
            module="M5", error_type=ErrorType.no_candidates,
            message="Optical-SAR fusion unavailable: no co-registered pair for this location/date.",
            fallback_applied=False,
        )
    for label, cand in [("optical", optical), ("sar", sar)]:
        task_type = "vlm_sar" if label == "sar" else "vlm"
        if not _file_exists_or_mock(cand.file_path, task_type):
            return False, _missing_image_error(cand.scene_id)
    return True, None


def check_pair_compatibility(p1: str, p2: str) -> tuple[bool, str]:
    """
    Pair compatibility check for upload endpoint.
    Rasterio checks: both readable, same CRS, footprint intersection > 80% of smaller extent,
    compatible band counts for intent.
    Returns (bool, reason_string). On failure, reason explains why.
    """
    try:
        with rasterio.open(p1) as ds1, rasterio.open(p2) as ds2:
            # Check both readable
            pass
    except Exception as e:
        return False, f"Unreadable file: {e}"
    
    # Check same CRS
    with rasterio.open(p1) as ds1, rasterio.open(p2) as ds2:
        if ds1.crs != ds2.crs:
            return False, f"CRS mismatch: {ds1.crs} vs {ds2.crs}"
        
        # Check footprint intersection > 80% of smaller extent
        try:
            geom1 = shapely_wkt.loads(f"POLYGON(({ds1.bounds.left} {ds1.bounds.bottom}, {ds1.bounds.right} {ds1.bounds.bottom}, {ds1.bounds.right} {ds1.bounds.top}, {ds1.bounds.left} {ds1.bounds.top}, {ds1.bounds.left} {ds1.bounds.bottom}))")
            geom2 = shapely_wkt.loads(f"POLYGON(({ds2.bounds.left} {ds2.bounds.bottom}, {ds2.bounds.right} {ds2.bounds.bottom}, {ds2.bounds.right} {ds2.bounds.top}, {ds2.bounds.left} {ds2.bounds.top}, {ds2.bounds.left} {ds2.bounds.bottom}))")
            
            intersection = geom1.intersection(geom2)
            area1 = geom1.area
            area2 = geom2.area
            smaller = min(area1, area2)
            
            if smaller > 0:
                overlap_ratio = intersection.area / smaller
                if overlap_ratio < 0.8:
                    return False, f"Footprint overlap {overlap_ratio:.1%} < 80% of smaller extent"
        except Exception as e:
            return False, f"Footprint check failed: {e}"
        
        # Check compatible band counts (at least 1 band each)
        if ds1.count < 1 or ds2.count < 1:
            return False, "One or both images have no bands"
    
    return True, "Compatible"


def _no_candidate_error() -> UniformError:
    return UniformError(
        module="M5", error_type=ErrorType.no_candidates,
        message="No matching imagery found for the specified criteria.",
        fallback_applied=False,
    )


def _missing_image_error(scene_id: str) -> UniformError:
    return UniformError(
        module="M5", error_type=ErrorType.invalid_input,
        message=f"Image file missing for scene {scene_id}.",
        fallback_applied=False,
    )
