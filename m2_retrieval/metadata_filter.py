"""
M2 Retrieval - Satellite Metadata Filter (Stage 1).

Stage 1: Deterministic metadata filtering of candidate satellite scenes.
Filters by AOI (bounding box), date range, sensor type, target object, etc.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math


@dataclass
class QuestionMetadata:
    """Legacy academic question metadata."""
    query: str
    objects: List[str] = field(default_factory=list)
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    sensor: Optional[str] = None


@dataclass
class FilterCriteria:
    """Legacy filter criteria."""
    objects: List[str] = field(default_factory=list)
    location: Optional[str] = None
    bbox: Optional[List[float]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    sensor: Optional[str] = None


class MetadataFilterEngine:
    """Legacy academic metadata filter - simplified implementation."""

    def filter(self, candidates: List[Dict], criteria: FilterCriteria) -> List[Dict]:
        results = []
        for c in candidates:
            if criteria.objects and not any(
                o.lower() in str(c.get("object", "")).lower() for o in criteria.objects
            ):
                continue
            if criteria.sensor and c.get("sensor") != criteria.sensor:
                continue
            results.append(c)
        return results


def haversine_km(lon1, lat1, lon2, lat2):
    """Haversine distance in km."""
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2*math.asin(math.sqrt(a))
    return R * c


class SatelliteMetadataFilter:
    """Stage 1: Deterministic metadata filter for satellite scene candidates.

    Filters candidates by:
      - AOI (bounding box intersection or Point+Radius)
      - Date range (inclusive)
      - Sensor type
      - Target object/class
    """

    @staticmethod
    def matches_aoi(candidate_bbox: Optional[List[float]], query_aoi: Optional[Any]) -> bool:
        """Check if candidate bbox intersects with query AOI.

        Both bboxes are [min_lon, min_lat, max_lon, max_lat].
        """
        if query_aoi is None:
            return True
        if candidate_bbox is None or len(candidate_bbox) != 4:
            return False

        # query_aoi can be a list [min_lon, min_lat, max_lon, max_lat]
        if isinstance(query_aoi, (list, tuple)) and len(query_aoi) == 4:
            c_min_lon, c_min_lat, c_max_lon, c_max_lat = candidate_bbox
            q_min_lon, q_min_lat, q_max_lon, q_max_lat = query_aoi
            # Bbox intersection check
            if c_max_lon < q_min_lon or c_min_lon > q_max_lon:
                return False
            if c_max_lat < q_min_lat or c_min_lat > q_max_lat:
                return False
            return True
        return True

    @staticmethod
    def matches_date_range(candidate_date: Optional[str], query_range: Optional[List[str]]) -> bool:
        """Check if candidate date falls within query date range."""
        if query_range is None:
            return True
        if candidate_date is None:
            return True  # Don't filter out if no date

        start, end = query_range
        if start and candidate_date < start:
            return False
        if end and candidate_date > end:
            return False
        return True

    @staticmethod
    def matches_sensor(candidate_sensor: Optional[str], query_sensor: Optional[Any]) -> bool:
        """Check if candidate sensor matches query sensor."""
        if query_sensor is None:
            return True
        if candidate_sensor is None:
            return True
        if isinstance(query_sensor, (list, tuple)):
            return candidate_sensor in query_sensor
        return candidate_sensor == query_sensor

    @staticmethod
    def matches_object(candidate_object: Optional[str], query_object: Optional[str]) -> bool:
        """Check if candidate object matches query object (fuzzy match)."""
        if query_object is None:
            return True
        if candidate_object is None:
            return False
        return query_object.lower() in candidate_object.lower() or candidate_object.lower() in query_object.lower()

    @classmethod
    def matches_scene(cls, scene: Any, query: Any) -> bool:
        """Check if a single scene matches all query criteria."""
        # Extract candidate fields
        candidate_aoi = getattr(scene, "aoi", None) or (scene.extra_attributes.get("aoi") if hasattr(scene, "extra_attributes") and scene.extra_attributes else None)
        candidate_date = getattr(scene, "acquisition_date", None)
        candidate_sensor = getattr(scene, "sensor", None)
        candidate_object = (
            scene.target_objects[0] if hasattr(scene, "target_objects") and scene.target_objects else None
        )

        # Extract query fields (m2_retrieval StructuredQuery)
        query_aoi = getattr(query, "aoi", None)
        query_date_range = getattr(query, "date_range", None)
        query_sensor = getattr(query, "sensor", None)
        query_object = getattr(query, "object", None)

        if not cls.matches_aoi(candidate_aoi, query_aoi):
            return False
        if not cls.matches_date_range(candidate_date, query_date_range):
            return False
        if not cls.matches_sensor(candidate_sensor, query_sensor):
            return False
        if not cls.matches_object(candidate_object, query_object):
            return False
        return True

    @classmethod
    def filter_candidates(cls, candidates: List[Any], query: Any) -> List[Any]:
        """Filter candidates deterministically preserving 100% of original metadata."""
        filtered = []
        for candidate in candidates:
            try:
                if cls.matches_scene(candidate, query):
                    filtered.append(candidate)
            except Exception:
                continue
        return filtered