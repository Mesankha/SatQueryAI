"""
M2 Retrieval - Stage 3 Multi-Factor Score Fusion.

S_composite = w_geo * S_geo + w_temporal * S_temporal + w_semantic * S_semantic + w_modality * S_modality

Plus legacy BM25/Composite scorers.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# Legacy academic scorers

class BM25Scorer:
    """BM25 scoring (legacy)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score_query(self, query: str, doc_id: str) -> float:
        # Simple placeholder
        return 1.0


class CompositeScorer:
    """Legacy composite scorer (BM25 + semantic)."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {"semantic": 0.5, "bm25": 0.3, "metadata": 0.2}

    def score_candidate(self, candidate: Any, query: Any) -> float:
        return 1.0


# Satellite-specific score fusion (Stage 3)

def _haversine_km(lon1, lat1, lon2, lat2) -> float:
    """Haversine distance in km."""
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2*math.asin(math.sqrt(a))
    return R * c


def _bbox_centroid(bbox: Optional[List[float]]) -> Optional[Tuple[float, float]]:
    """Get centroid of a bbox [min_lon, min_lat, max_lon, max_lat]."""
    if bbox is None or len(bbox) != 4:
        return None
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _bbox_overlap_ratio(candidate_bbox: List[float], query_bbox: List[float]) -> float:
    """Compute area overlap ratio between two bboxes."""
    c_min_lon, c_min_lat, c_max_lon, c_max_lat = candidate_bbox
    q_min_lon, q_min_lat, q_max_lon, q_max_lat = query_bbox
    # Intersection
    i_min_lon = max(c_min_lon, q_min_lon)
    i_min_lat = max(c_min_lat, q_min_lat)
    i_max_lon = min(c_max_lon, q_max_lon)
    i_max_lat = min(c_max_lat, q_max_lat)
    if i_min_lon >= i_max_lon or i_min_lat >= i_max_lat:
        return 0.0
    i_area = (i_max_lon - i_min_lon) * (i_max_lat - i_min_lat)
    c_area = max((c_max_lon - c_min_lon) * (c_max_lat - c_min_lat), 1e-9)
    return i_area / c_area


def normalize_change_map(values: List[float]) -> List[float]:
    """Normalize a list of change values to [0, 1]."""
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx - mn < 1e-9:
        return [0.5] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def calculate_change_score(ndvi_diff: float, ndwi_diff: float, sar_ratio: float) -> float:
    """Calculate combined change score from multiple modalities."""
    ndvi_score = abs(ndvi_diff)
    ndwi_score = abs(ndwi_diff)
    sar_score = abs(sar_ratio - 1.0)  # ratio=1.0 means no change
    return (ndvi_score + ndwi_score + sar_score) / 3.0


class SatelliteScoreEngine:
    """Stage 3: Multi-factor score fusion engine.

    Computes a composite ranking score using:
      - S_geo: spatial proximity (Haversine distance) or area overlap
      - S_temporal: date proximity
      - S_semantic: from Stage 2 (propagated)
      - S_modality: sensor capability match + cloud cover penalty
    """

    def __init__(self, weights=None):
        from m2_retrieval.config import SatelliteScoreWeights
        self.weights = weights or SatelliteScoreWeights()
        if hasattr(self.weights, "validate"):
            self.weights.validate()

    def _compute_geo_score(self, scene: Any, query: Any) -> float:
        """Compute spatial geo score (0-1, higher = closer)."""
        cand_bbox = getattr(scene, "aoi", None)
        query_aoi = getattr(query, "aoi", None)

        if query_aoi is None or cand_bbox is None:
            return 0.5  # neutral
        # If query_aoi is a bbox, compute overlap
        if isinstance(query_aoi, (list, tuple)) and len(query_aoi) == 4:
            try:
                return _bbox_overlap_ratio(cand_bbox, list(query_aoi))
            except Exception:
                return 0.5
        return 0.5

    def _compute_temporal_score(self, scene: Any, query: Any) -> float:
        """Compute temporal score (0-1, higher = closer in time)."""
        cand_date = getattr(scene, "acquisition_date", None)
        query_range = getattr(query, "date_range", None)
        if cand_date is None or query_range is None:
            return 0.5
        if isinstance(query_range, (list, tuple)) and len(query_range) == 2:
            start, end = query_range
            if start and end and cand_date >= start and cand_date <= end:
                return 1.0  # within range
            try:
                from datetime import datetime
                cd = datetime.fromisoformat(cand_date.replace("Z", "+00:00"))
                if start:
                    sd = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    if cd < sd:
                        delta_days = abs((sd - cd).days)
                        return max(0.0, 1.0 - delta_days / 365.0)
            except Exception:
                pass
        return 0.5

    def _compute_modality_score(self, scene: Any, query: Any) -> float:
        """Compute modality score (0-1, higher = better sensor match)."""
        cand_sensor = getattr(scene, "sensor", None)
        query_sensor = getattr(query, "sensor", None)
        cloud_cover = getattr(scene, "cloud_cover", None)
        cloud_max = getattr(query, "cloud_cover_max", None)

        score = 1.0
        if query_sensor is not None and cand_sensor is not None:
            if isinstance(query_sensor, (list, tuple)):
                if cand_sensor in query_sensor:
                    score = 1.0
                else:
                    score = 0.3
            elif cand_sensor != query_sensor:
                score = 0.3

        # Cloud cover penalty
        if cloud_cover is not None and cloud_max is not None:
            if cloud_cover > cloud_max:
                score *= 0.5
        return score

    def _compute_semantic_score(self, scene: Any) -> float:
        """Extract semantic score from Stage 2 (default 0.5)."""
        breakdown = getattr(scene, "score_breakdown", None) or {}
        return float(breakdown.get("semantic_score", 0.5))

    def score_candidate(self, scene: Any, query: Any, semantic_score: Optional[float] = None) -> Tuple[float, Dict[str, float]]:
        """Calculate Stage 3 composite score for a single candidate.

        Returns (composite_score, score_breakdown_dict).
        """
        s_geo = self._compute_geo_score(scene, query)
        s_temp = self._compute_temporal_score(scene, query)
        s_sem = semantic_score if semantic_score is not None else self._compute_semantic_score(scene)
        s_mod = self._compute_modality_score(scene, query)

        w = self.weights
        if hasattr(w, 'w_geo'):
            w_g = w.w_geo
            w_t = w.w_temporal
            w_s = w.w_semantic
            w_m = w.w_modality
        else:
            weight_map = getattr(w, "weights", None)
            if isinstance(weight_map, dict):
                w_g = weight_map.get("geo", 0.3)
                w_t = weight_map.get("temporal", 0.25)
                w_s = weight_map.get("semantic", 0.3)
                w_m = weight_map.get("modality", 0.15)
            else:
                # Safely handle objects with attribute-based weights (e.g., SatelliteScoreWeights)
                w_g = getattr(w, "geo", 0.3)
                w_t = getattr(w, "temporal", 0.25)
                w_s = getattr(w, "semantic", 0.3)
                w_m = getattr(w, "modality", 0.15)

        # Shift weights for change detection
        extra = getattr(query, "extra_params", {})
        if extra.get("change_flag"):
            w_t = 0.40
            w_s = 0.40
            w_g = 0.10
            w_m = 0.10

        composite = (w_g * s_geo + w_t * s_temp +
                     w_s * s_sem + w_m * s_mod)

        breakdown = {
            "geo": s_geo,
            "temporal": s_temp,
            "semantic": s_sem,
            "modality": s_mod,
        }
        return float(composite), breakdown

    def score_and_rank_candidates(
        self,
        candidates: List[Any],
        query: Any,
        top_k: Optional[int] = None
    ) -> List[Any]:
        """Calculate Stage 3 scores for all candidates, attach to objects, sort by score."""
        if not candidates:
            return []

        scored_candidates = []
        for c in candidates:
            c_score, breakdown = self.score_candidate(c, query)

            if hasattr(c, "score"):
                c.score = c_score
            if hasattr(c, "score_breakdown"):
                c.score_breakdown = breakdown

            if isinstance(c, dict):
                c["score"] = c_score
                c["composite_score"] = c_score
                c["score_breakdown"] = breakdown

            scored_candidates.append(c)

        scored_candidates.sort(
            key=lambda x: (
                -(getattr(x, "score", 0.0) if hasattr(x, "score") and x.score is not None
                  else (x.get("score", 0.0) if isinstance(x, dict) else 0.0)),
                str(getattr(x, "scene_id", x.get("scene_id", "") if isinstance(x, dict) else ""))
            )
        )

        if top_k is not None and top_k > 0 and top_k < len(scored_candidates):
            scored_candidates = scored_candidates[:top_k]

        return scored_candidates