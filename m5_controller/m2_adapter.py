"""
M2 Adapter - Bridges M0 catalog + M1 StructuredQuery to the real M2 retrieval engine.

This module:
  1. Reads scenes from M0 SQLite catalog
  2. Converts shared.schemas.SceneMetadata -> m2_retrieval.SceneMetadata
  3. Converts shared.schemas.StructuredQuery -> m2_retrieval.StructuredQuery
  4. Calls the real M2 satellite pipeline:
     - Stage 1: SatelliteMetadataFilter (deterministic)
     - Stage 2: SatelliteSemanticRanker (RemoteCLIP/GeoRSCLIP)
     - Stage 3: SatelliteScoreEngine (multi-factor fusion)
  5. Converts results back to shared.schemas.SceneMetadata

The M2 satellite pipeline already has 76 passing tests, so we trust its
output and just do the schema translation.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup path to find M2 module
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
_M2_PATH = _PROJECT_ROOT / "M2_retrieval"
if str(_M2_PATH) not in sys.path:
    sys.path.insert(0, str(_M2_PATH))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter result cache (avoid re-loading model on every query)
# ---------------------------------------------------------------------------
_pipeline_cache: Dict[str, Any] = {}


def _get_m2_pipeline():
    """
    Lazily create and cache the M2 pipeline singletons.

    Returns tuple: (SatelliteMetadataFilter class, SatelliteScoreEngine, SatelliteSemanticRanker)
    """
    global _pipeline_cache
    if "pipeline" in _pipeline_cache:
        return _pipeline_cache["pipeline"]

    from m2_retrieval import (
        SatelliteMetadataFilter,
        SatelliteScoreEngine,
        SatelliteSemanticRanker,
        SatelliteScoreWeights,
    )

    # Use default weights (w_geo=0.30, w_temp=0.25, w_sem=0.30, w_mod=0.15)
    weights = SatelliteScoreWeights()
    score_engine = SatelliteScoreEngine(weights=weights)

    # Semantic ranker (uses RemoteCLIP -> GeoRSCLIP -> metadata-only fallback chain)
    ranker = SatelliteSemanticRanker(embedding_dim=512)

    _pipeline_cache["pipeline"] = (SatelliteMetadataFilter, score_engine, ranker)
    logger.info("M2 pipeline initialized (RemoteCLIP -> GeoRSCLIP -> metadata-only fallback)")
    return _pipeline_cache["pipeline"]


# ---------------------------------------------------------------------------
# Schema converters
# ---------------------------------------------------------------------------

def m1_to_m2_query(m1_query) -> Any:
    """
    Convert shared.schemas.StructuredQuery -> m2_retrieval.StructuredQuery.

    Field mapping:
      m1.query_text           -> m2.text_query
      m1.location             -> m2.aoi (geocoded to bbox if possible)
      m1.aoi                  -> m2.aoi (direct if polygon, else bbox)
      m1.start_date + end_date -> m2.date_range
      m1.sensor               -> m2.sensor
      m1.object               -> m2.object
    """
    from m2_retrieval.satellite_index import StructuredQuery as M2StructuredQuery

    # Build AOI
    aoi = None
    if m1_query.aoi is not None:
        # m1.aoi is a dict {type: "Polygon", coordinates: [[[lon,lat],...]]}
        coords = m1_query.aoi.coordinates
        if coords and len(coords) > 0 and len(coords[0]) > 0:
            lons = [c[0] for c in coords[0]]
            lats = [c[1] for c in coords[0]]
            aoi = [min(lons), min(lats), max(lons), max(lats)]
    elif m1_query.location:
        # Try to geocode the location to a bbox
        aoi = _geocode_location_to_bbox(m1_query.location)

    # Build date range
    date_range = None
    if m1_query.start_date or m1_query.end_date:
        start = m1_query.start_date.isoformat() if m1_query.start_date else None
        end = m1_query.end_date.isoformat() if m1_query.end_date else None
        date_range = [start, end]

    # Sensor mapping
    sensor = None
    if m1_query.sensor:
        if m1_query.sensor.value == "both":
            sensor = ["Sentinel-1", "Sentinel-2"]
        else:
            sensor = m1_query.sensor.value

    # Object/class
    object_filter = m1_query.object
    text_query = m1_query.query_text
    if m1_query.location and not aoi:
        text_query = f"{m1_query.location} {text_query}".strip()

    return M2StructuredQuery(
        text_query=text_query,
        aoi=aoi,
        date_range=date_range,
        sensor=sensor,
        object=object_filter,
        cloud_cover_max=20.0,
        extra_params={
            "task_type": m1_query.task_type.value if hasattr(m1_query.task_type, "value") else str(m1_query.task_type),
            "change_flag": m1_query.change_flag,
            "event": m1_query.event,
        },
    )


def _geocode_location_to_bbox(location: str) -> Optional[List[float]]:
    """
    Convert a location name to a [min_lon, min_lat, max_lon, max_lat] bbox.
    Uses a small built-in lookup of major Indian cities + global cities.
    For unknown locations, returns None (text search will be used instead).
    """
    if not location:
        return None
    loc_lower = location.lower().strip()

    # Major Indian cities + neighboring regions
    KNOWN_BBOXES = {
        "delhi": (77.05, 28.40, 77.35, 28.80),
        "new delhi": (77.10, 28.50, 77.30, 28.70),
        "mumbai": (72.75, 18.85, 73.05, 19.30),
        "bombay": (72.75, 18.85, 73.05, 19.30),
        "bangalore": (77.45, 12.85, 77.75, 13.15),
        "bengaluru": (77.45, 12.85, 77.75, 13.15),
        "chennai": (80.15, 12.95, 80.40, 13.25),
        "madras": (80.15, 12.95, 80.40, 13.25),
        "kolkata": (88.25, 22.45, 88.55, 22.75),
        "calcutta": (88.25, 22.45, 88.55, 22.75),
        "hyderabad": (78.30, 17.25, 78.60, 17.55),
        "pune": (73.70, 18.45, 74.00, 18.75),
        "ahmedabad": (72.50, 22.95, 72.80, 23.25),
        "jaipur": (75.65, 26.80, 75.95, 27.10),
        "lucknow": (80.85, 26.75, 81.15, 27.05),
        # --- Assam hackathon AOIs (7 required) ---
        "guwahati": (91.55, 26.05, 91.85, 26.25),
        "silchar": (92.72, 24.75, 92.87, 24.88),
        "barpeta": (90.92, 26.28, 91.08, 26.40),
        "tezpur": (92.72, 26.58, 92.88, 26.70),
        "majuli": (93.80, 26.85, 94.30, 27.15),
        "jorhat": (94.10, 26.68, 94.30, 26.82),
        "dibrugarh": (94.85, 27.42, 95.00, 27.55),
        # --- Assam state / regions ---
        "assam": (90.50, 25.50, 95.00, 27.50),
        "kerala": (76.20, 9.95, 76.50, 10.50),
        "rajasthan": (72.50, 26.50, 74.00, 28.00),
        "west bengal": (87.50, 22.50, 89.00, 24.00),
        "kashmir": (74.00, 33.50, 76.00, 35.00),
        "ladakh": (76.00, 32.00, 80.00, 36.00),
        "sri lanka": (80.00, 6.00, 82.00, 10.00),
        "nepal": (80.00, 26.00, 88.00, 30.00),
        "bhutan": (88.00, 26.00, 92.00, 28.50),
        "north india": (75.0, 26.0, 85.0, 32.0),
        "south india": (76.0, 8.0, 80.0, 14.0),
        "east india": (85.0, 20.0, 92.0, 27.0),
        "west india": (70.0, 18.0, 75.0, 25.0),
    }

    # Exact match
    if loc_lower in KNOWN_BBOXES:
        return list(KNOWN_BBOXES[loc_lower])

    # Partial match
    for city, bbox in KNOWN_BBOXES.items():
        if city in loc_lower or loc_lower in city:
            return list(bbox)

    # Try to use geocoder if available (optional)
    try:
        from geopy.geocoders import Nominatim  # type: ignore
        geolocator = Nominatim(user_agent="satquery-ai/1.0")
        location_obj = geolocator.geocode(location, timeout=5)
        if location_obj:
            lat, lon = location_obj.latitude, location_obj.longitude
            # ~10km box around point
            delta = 0.1
            return [lon - delta, lat - delta, lon + delta, lat + delta]
    except Exception:
        pass

    return None


def m0_to_m2_scene(m0_scene) -> Any:
    """
    Convert shared.schemas.SceneMetadata -> m2_retrieval.SceneMetadata.
    Parses geometry_wkt to AOI bbox.
    """
    from m2_retrieval.satellite_index import SceneMetadata as M2SceneMetadata

    # Parse WKT polygon to bbox
    aoi = _wkt_to_bbox(m0_scene.geometry_wkt) if m0_scene.geometry_wkt else None

    # Map modality
    sensor_value = m0_scene.sensor
    if m0_scene.modality == "sar":
        sensor_value = "SAR"

    return M2SceneMetadata(
        scene_id=m0_scene.scene_id,
        aoi=aoi,
        acquisition_date=m0_scene.acquisition_time.isoformat() if m0_scene.acquisition_time else None,
        sensor=sensor_value,
        cloud_cover=m0_scene.cloud_cover,
        target_objects=[m0_scene.object] if m0_scene.object else [],
        extra_attributes={
            "file_path": m0_scene.file_path,
            "modality": m0_scene.modality,
            "paired_scene_id": m0_scene.paired_scene_id,
        },
    )


def _wkt_to_bbox(wkt: str) -> Optional[List[float]]:
    """Parse WKT POLYGON to [min_lon, min_lat, max_lon, max_lat] bbox."""
    if not wkt or "POLYGON" not in wkt.upper():
        return None
    try:
        import re
        m = re.search(r"POLYGON\s*\(\(([^)]+)\)\)", wkt, re.IGNORECASE)
        if not m:
            return None
        coords_str = m.group(1)
        coords = []
        for pair in coords_str.split(","):
            parts = pair.strip().split()
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
                coords.append((lon, lat))
        if not coords:
            return None
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return [min(lons), min(lats), max(lons), max(lats)]
    except Exception:
        return None


def m2_to_m0_scene(m2_scene) -> Any:
    """
    Convert m2_retrieval.SceneMetadata -> shared.schemas.SceneMetadata.
    Preserves score and score_breakdown.
    """
    from shared.schemas import SceneMetadata as SharedSceneMetadata

    # Build geometry WKT from AOI bbox
    geometry_wkt = ""
    if m2_scene.aoi and len(m2_scene.aoi) == 4:
        min_lon, min_lat, max_lon, max_lat = m2_scene.aoi
        geometry_wkt = (
            f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
            f"{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
        )
    else:
        geometry_wkt = "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"

    # Map sensor
    sensor = m2_scene.sensor or "Sentinel-2"
    if sensor == "SAR":
        sensor = "Sentinel-1"
    modality = m2_scene.extra_attributes.get("modality", "optical")
    if sensor == "Sentinel-1" and modality == "optical":
        modality = "sar"
    if sensor == "Sentinel-2" and modality == "sar":
        modality = "multispectral"

    # Parse acquisition date
    acq_time = None
    if m2_scene.acquisition_date:
        try:
            dt = datetime.fromisoformat(m2_scene.acquisition_date.replace("Z", "+00:00"))
            acq_time = dt
        except Exception:
            pass

    return SharedSceneMetadata(
        scene_id=m2_scene.scene_id,
        sensor=sensor,
        modality=modality,
        acquisition_time=acq_time or datetime.now(timezone.utc),
        geometry_wkt=geometry_wkt,
        file_path=m2_scene.extra_attributes.get("file_path"),
        cloud_cover=m2_scene.cloud_cover,
        object=m2_scene.target_objects[0] if m2_scene.target_objects else None,
        paired_scene_id=m2_scene.extra_attributes.get("paired_scene_id"),
    )


# ---------------------------------------------------------------------------
# Main adapter: read from M0, run through M2, return to M5
# ---------------------------------------------------------------------------

def real_m2_retrieve(m1_query, top_k: int = 10, db_path: Optional[str] = None) -> List[Any]:
    """
    The real M2 retrieval function that replaces mock_m2_retrieve.

    Pipeline:
      1. Read all scenes from M0 SQLite catalog
      2. Convert to M2 SceneMetadata format
      3. Build M2 StructuredQuery from M1 output
      4. Stage 1: Filter by AOI/date/sensor/object
      5. Stage 2: Semantic ranking (RemoteCLIP/GeoRSCLIP)
      6. Stage 3: Multi-factor score fusion
      7. Convert results back to shared.SceneMetadata
      8. Resolve URLs to local cached paths
      9. Return top-K

    Args:
        m1_query: shared.schemas.StructuredQuery from M1
        top_k: Max number of results to return
        db_path: Optional path to M0 catalog (default: ./data/catalog.db)

    Returns:
        List of shared.schemas.SceneMetadata (top-K by composite score)
    """
    from m0_catalog.catalog import get_catalog
    from m0_catalog.image_resolver import resolve_image

    # 1. Load scenes from M0
    try:
        catalog = get_catalog(read_only=True)
    except Exception as e:
        logger.warning(f"M0 catalog not available: {e}. Falling back to fixtures.")
        from shared.validate import load_fixture_scenes
        return load_fixture_scenes("fixtures/fixtures_scenes.json")[:top_k]

    try:
        all_scenes = catalog.get_all_scenes()
    except Exception as e:
        logger.warning(f"Failed to read M0 catalog: {e}")
        catalog.close()
        from shared.validate import load_fixture_scenes
        return load_fixture_scenes("fixtures/fixtures_scenes.json")[:top_k]

    catalog.close()

    if not all_scenes:
        logger.info("M0 catalog is empty - falling back to fixtures")
        from shared.validate import load_fixture_scenes
        return load_fixture_scenes("fixtures/fixtures_scenes.json")[:top_k]

    logger.info(f"M0 catalog has {len(all_scenes)} scenes")

    # 2. Convert to M2 format
    m2_scenes = [m0_to_m2_scene(s) for s in all_scenes]

    # 3. Build M2 query
    m2_query = m1_to_m2_query(m1_query)

    # 4-6. Run through the 3-stage M2 pipeline
    try:
        SatelliteMetadataFilter, score_engine, ranker = _get_m2_pipeline()

        # Stage 1: Metadata filtering
        filtered = SatelliteMetadataFilter.filter_candidates(m2_scenes, m2_query)
        logger.info(f"Stage 1 (filter): {len(m2_scenes)} -> {len(filtered)} candidates")
        if not filtered:
            # If filter removes everything, relax the filter (no AOI/date match for this query)
            # and let semantic ranking pick best from all
            filtered = m2_scenes[:50]  # limit to first 50 for performance

        # Stage 2: Semantic ranking (may be skipped if RemoteCLIP unavailable)
        ranked, encoder_used = ranker.rank_candidates(filtered, m2_query, top_k=top_k * 2)
        logger.info(f"Stage 2 (semantic rank): encoder={encoder_used}, top_k={top_k * 2}")

        # Stage 3: Multi-factor score fusion
        final = score_engine.score_and_rank_candidates(ranked, m2_query, top_k=top_k)
        logger.info(f"Stage 3 (score fusion): returned {len(final)} scenes")
    except Exception as e:
        logger.error(f"M2 pipeline error: {e}. Using filtered scenes as-is.")
        final = m2_scenes[:top_k]

    # 7. Convert back to shared.SceneMetadata
    results = []
    for m2_scene in final:
        try:
            shared_scene = m2_to_m0_scene(m2_scene)
            if hasattr(m2_scene, 'score'):
                shared_scene.score = m2_scene.score
            if hasattr(m2_scene, 'score_breakdown'):
                shared_scene.score_breakdown = m2_scene.score_breakdown
            # 8. Resolve URLs to local cached paths
            if shared_scene.file_path and (shared_scene.file_path.startswith("http") or shared_scene.file_path.startswith("s3://")):
                resolved = resolve_image(shared_scene.file_path)
                if resolved:
                    shared_scene.file_path = resolved
            results.append(shared_scene)
        except Exception as e:
            logger.debug(f"Failed to convert scene {m2_scene.scene_id}: {e}")
            continue

    return results[:top_k]


# ---------------------------------------------------------------------------
# M0 catalog population from real sources
# ---------------------------------------------------------------------------

def populate_m0_from_sar_lora(sar_lora_dir, db_path: str = "./data/catalog.db") -> int:
    """
    Populate M0 catalog from the SAR LoRA training data (already downloaded
    by acquire_sar.py). Each scene is a real Sentinel-1 acquisition.
    """
    from m0_catalog.catalog import Catalog
    from shared.schemas import SceneMetadata

    sar_lora_dir = Path(sar_lora_dir)
    catalog = Catalog(db_path, read_only=False)
    count = 0

    # Read scenes metadata
    meta_file = sar_lora_dir / "scenes_metadata.jsonl"
    if not meta_file.exists():
        print(f"No scenes_metadata.jsonl found at {meta_file}")
        catalog.close()
        return 0

    with open(meta_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Get land cover from city name or use unknown
            from m3_vlm.acquire_sar import _infer_land_cover
            label = _infer_land_cover(rec.get("bbox", [77, 28, 78, 29]))
            if not label or label == "mixed":
                label = "unknown"

            try:
                scene = SceneMetadata(
                    scene_id=rec["scene_id"],
                    sensor="Sentinel-1" if "sentinel-1" in rec.get("collection", "") else "Sentinel-2",
                    modality="sar" if "sentinel-1" in rec.get("collection", "") else "optical",
                    acquisition_time=datetime.fromisoformat(
                        rec["datetime"].replace("Z", "+00:00")
                    ),
                    geometry_wkt=f"POLYGON(({rec['bbox'][0]} {rec['bbox'][1]}, "
                                  f"{rec['bbox'][2]} {rec['bbox'][1]}, "
                                  f"{rec['bbox'][2]} {rec['bbox'][3]}, "
                                  f"{rec['bbox'][0]} {rec['bbox'][3]}, "
                                  f"{rec['bbox'][0]} {rec['bbox'][1]}))",
                    file_path=rec.get("local_image") or rec.get("signed_href"),
                    cloud_cover=rec.get("properties", {}).get("eo:cloud_cover"),
                    object=label,
                    paired_scene_id=None,
                )
                catalog.insert_scene(scene, dataset_origin=f"planetary_{sar_lora_dir.name}")
                count += 1
            except Exception as e:
                logger.debug(f"Failed to insert scene: {e}")
                continue

    catalog.close()
    logger.info(f"Populated M0 catalog with {count} scenes from {sar_lora_dir}")
    return count


def populate_m0_from_planetary(
    bbox: Tuple[float, float, float, float],
    date_range: Tuple[str, str],
    label: str = "unknown",
    db_path: str = "./data/catalog.db",
    max_scenes: int = 100,
) -> int:
    """
    Query Planetary Computer and add scenes directly to M0 catalog.
    Use this for a quick demo without downloading files.
    """
    try:
        import planetary_computer
        import pystac_client
    except ImportError:
        logger.error("Install: pip install planetary-computer pystac-client")
        return 0

    from m0_catalog.catalog import Catalog
    from shared.schemas import SceneMetadata

    catalog = Catalog(db_path, read_only=False)
    count = 0

    client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1/"
    )

    for collection in ["sentinel-2-l2a", "sentinel-1-grd"]:
        try:
            search = client.search(
                collections=[collection],
                bbox=list(bbox),
                datetime=f"{date_range[0]}/{date_range[1]}",
                limit=max_scenes,
            )
            items = list(search.items())
        except Exception as e:
            logger.debug(f"Search failed for {collection}: {e}")
            continue

        for item in items:
            try:
                # Get signed URL for the visual asset
                file_path = None
                if "visual" in item.assets:
                    file_path = planetary_computer.sign(item.assets["visual"].href)
                elif "thumbnail" in item.assets:
                    file_path = planetary_computer.sign(item.assets["thumbnail"].href)

                is_sar = "sentinel-1" in collection
                scene = SceneMetadata(
                    scene_id=item.id,
                    sensor="Sentinel-1" if is_sar else "Sentinel-2",
                    modality="sar" if is_sar else "optical",
                    acquisition_time=datetime.fromisoformat(
                        str(item.datetime).replace("Z", "+00:00")
                    ),
                    geometry_wkt=f"POLYGON(({bbox[0]} {bbox[1]}, {bbox[2]} {bbox[1]}, "
                                  f"{bbox[2]} {bbox[3]}, {bbox[0]} {bbox[3]}, "
                                  f"{bbox[0]} {bbox[1]}))",
                    file_path=file_path,
                    cloud_cover=item.properties.get("eo:cloud_cover") if not is_sar else None,
                    object=label,
                    paired_scene_id=None,
                )
                catalog.insert_scene(scene, dataset_origin=f"planetary_{collection}")
                count += 1
            except Exception as e:
                logger.debug(f"Failed to insert {item.id}: {e}")
                continue

    catalog.close()
    logger.info(f"Populated M0 catalog with {count} scenes from Planetary Computer")
    return count