"""
Confidence band computation for ResultItem.
Determines High/Medium/Low confidence based on pipeline state.
"""
from typing import Optional, Dict, Any
from shared.schemas import ResultItem, ExecutionTrace, SceneMetadata


def compute_band(
    *,
    used_fallback: bool,
    trace_fallback_used: bool,
    single_observation: bool,
    cloud_cover_pct: Optional[float],
    coregistration_ok: Optional[bool],
    sar_cross_check: Optional[Dict[str, Any]] = None
) -> tuple[str, str]:
    """
    Compute confidence band and reason.
    
    Rules:
    - low if used_fallback or trace fallback on a primary model
    - medium if single_observation or (cloud_cover_pct or 0) > 30 or coregistration is False
    - high otherwise
    - SAR cross-check: if deterministic and model-derived SAR outputs disagree, lower confidence
    
    Returns: (band, reason)
    """
    reasons = []
    
    # Low confidence triggers
    if used_fallback:
        reasons.append("parser fallback used")
    if trace_fallback_used:
        reasons.append("model fallback used")
    
    if reasons:
        return "low", "Low — " + "; ".join(reasons)
    
    # Medium confidence triggers
    medium_reasons = []
    if single_observation:
        medium_reasons.append("single observation")
    if cloud_cover_pct is not None and cloud_cover_pct > 30:
        medium_reasons.append(f"cloud cover {cloud_cover_pct:.0f}%")
    if coregistration_ok is False:
        medium_reasons.append("coregistration failed")
    
    # SAR cross-check disagreement → medium (or low if severe)
    if sar_cross_check:
        disagreement = sar_cross_check.get("disagreement", False)
        severity = sar_cross_check.get("severity", "minor")
        if disagreement:
            if severity == "major":
                return "low", "Low — SAR model/deterministic disagreement (major)"
            medium_reasons.append("SAR model/deterministic disagreement")
    
    if medium_reasons:
        return "medium", "Medium — " + "; ".join(medium_reasons)
    
    # High confidence
    return "high", "High — primary models, clear observation, good registration"


def attach_confidence(
    result: ResultItem,
    scene: SceneMetadata,
    query_used_fallback: bool,
    trace: ExecutionTrace,
    sar_cross_check: Optional[Dict[str, Any]] = None
) -> None:
    """
    Attach confidence band and reason to ResultItem.
    Also adds freshness metadata.
    """
    # Determine single observation (no paired scene)
    single_obs = not scene.paired_scene_id
    
    # Cloud cover from scene metadata
    cloud_cover = scene.cloud_cover
    
    # Coregistration - assume OK if both scenes have valid geometry
    coreg_ok = scene.geometry_wkt is not None and scene.geometry_wkt.strip() != ""
    
    band, reason = compute_band(
        used_fallback=query_used_fallback,
        trace_fallback_used=trace.fallback_used,
        single_observation=single_obs,
        cloud_cover_pct=cloud_cover,
        coregistration_ok=coreg_ok,
        sar_cross_check=sar_cross_check
    )
    
    result.confidence_band = band
    result.confidence_reason = reason
    
    # Add freshness metadata
    if scene.acquisition_time:
        from datetime import datetime, timezone
        result.metadata["most_recent_observation"] = scene.acquisition_time.isoformat()
        age_days = (datetime.now(timezone.utc) - scene.acquisition_time).days
        result.metadata["observation_age_days"] = age_days