"""
M2 Retrieval Engine - Real retrieval with metadata filtering and text similarity.
Stage 1: SQL metadata filter (sensor, date range, AOI intersect)
Stage 2: Text similarity using sentence-transformers
Stage 3: Attach real score, sort desc, return top_k
"""
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from shapely import wkt as shapely_wkt
from shapely.geometry import box as shapely_box

from shared.schemas import StructuredQuery, SceneMetadata
from shared.config import get_config

logger = logging.getLogger(__name__)

# Global text embedder singleton
_text_embedder: Optional[SentenceTransformer] = None


def _get_text_embedder() -> Optional[SentenceTransformer]:
    """Lazy load the sentence transformer for text similarity."""
    global _text_embedder
    if _text_embedder is not None:
        return _text_embedder

    config = get_config()
    model_name = config.get("retrieval", {}).get("text_embedder", "sentence-transformers/all-MiniLM-L6-v2")

    try:
        _text_embedder = SentenceTransformer(model_name)
        logger.info(f"Loaded text embedder: {model_name}")
    except Exception as e:
        logger.warning(f"Failed to load text embedder {model_name}: {e}")
        _text_embedder = None

    return _text_embedder


def _build_text_for_scene(scene: SceneMetadata) -> str:
    """Build searchable text representation of a scene."""
    parts = []
    if scene.sensor:
        parts.append(scene.sensor)
    if scene.modality:
        parts.append(scene.modality)
    if scene.object:
        parts.append(scene.object)
    # Could add location from geometry_wkt centroid
    return " ".join(parts)


def _normalize_to_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _scene_matches_query(scene: SceneMetadata, query: StructuredQuery) -> bool:
    """Check if scene matches query metadata filters."""
    # Sensor match
    if query.sensor and query.sensor.value != "both":
        if scene.sensor != query.sensor.value and scene.sensor != "both":
            return False

    # Date range filter - normalize both to UTC for comparison
    scene_time = _normalize_to_utc(scene.acquisition_time)
    if query.start_date:
        start_date = _normalize_to_utc(query.start_date)
        if scene_time < start_date:
            return False
    if query.end_date:
        end_date = _normalize_to_utc(query.end_date)
        if scene_time > end_date:
            return False

    # AOI intersect (if query has AOI)
    if query.aoi and query.aoi.coordinates:
        try:
            scene_geom = shapely_wkt.loads(scene.geometry_wkt)
            query_geom = shapely_wkt.loads(query.aoi.model_dump_json())
            if not scene_geom.intersects(query_geom):
                return False
        except Exception:
            # If geometry parsing fails, don't filter on AOI
            pass

    return True


def retrieve(
    query: StructuredQuery,
    db_path: str,
    top_k: int = 5,
) -> list[SceneMetadata]:
    """
    Real M2 retrieval from catalog database.

    Stage 1: SQL metadata filter - sensor match, acquisition_time in range, AOI intersect
    Stage 2: Text similarity - sentence-transformers cosine between query_text and scene text
    Stage 3: Attach real score, sort desc, return top_k. Never raises.
    """
    config = get_config()
    retrieval_config = config.get("retrieval", {})
    default_top_k = retrieval_config.get("top_k", 5)
    top_k = top_k or default_top_k

    if not Path(db_path).exists():
        logger.warning(f"Catalog database not found: {db_path}")
        return []

    # Stage 1: SQL metadata filter
    candidates = []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build SQL query with filters
        sql = "SELECT * FROM scenes WHERE 1=1"
        params = []

        if query.sensor and query.sensor.value != "both":
            sql += " AND (sensor = ? OR sensor = 'both')"
            params.append(query.sensor.value)

        if query.start_date:
            sql += " AND acquisition_time >= ?"
            params.append(query.start_date.isoformat())

        if query.end_date:
            sql += " AND acquisition_time <= ?"
            params.append(query.end_date.isoformat())

        # Note: AOI intersect via shapely is done in Python after fetch for simplicity
        # Could be optimized with SpatiaLite if available

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            try:
                row_dict = dict(row)
                scene = SceneMetadata(
                    scene_id=row_dict["scene_id"],
                    sensor=row_dict["sensor"],
                    modality=row_dict["modality"],
                    acquisition_time=datetime.fromisoformat(row_dict["acquisition_time"]),
                    geometry_wkt=row_dict["geometry_wkt"],
                    file_path=row_dict.get("file_path"),
                    cloud_cover=row_dict.get("cloud_cover"),
                    object=row_dict.get("object"),
                    paired_scene_id=row_dict.get("paired_scene_id"),
                    wavelengths_nm=json.loads(row_dict["wavelengths_nm"]) if row_dict.get("wavelengths_nm") else None,
                )
                # AOI intersect check
                if _scene_matches_query(scene, query):
                    candidates.append(scene)
            except Exception as e:
                logger.warning(f"Failed to parse scene {row_dict['scene_id']}: {e}")

    except Exception as e:
        logger.warning(f"Database query failed: {e}")
        return []

    if not candidates:
        return []

    # Stage 2: Text similarity
    embedder = _get_text_embedder()
    if embedder is not None and query.query_text:
        try:
            query_emb = embedder.encode([query.query_text], normalize_embeddings=True)[0]
            scene_texts = [_build_text_for_scene(s) for s in candidates]
            scene_embs = embedder.encode(scene_texts, normalize_embeddings=True)

            # Cosine similarity (dot product since normalized)
            similarities = np.dot(scene_embs, query_emb)

            # Scenes without meaningful text get neutral score 0.5
            for i, text in enumerate(scene_texts):
                if not text.strip():
                    similarities[i] = 0.5

            # Attach scores
            for i, scene in enumerate(candidates):
                scene.score = float(similarities[i])
        except Exception as e:
            logger.warning(f"Text similarity computation failed: {e}")
            # Fallback: neutral scores
            for scene in candidates:
                scene.score = 0.5
    else:
        # No embedder or no query text - neutral scores
        for scene in candidates:
            scene.score = 0.5

    # Stage 3: Sort by score desc, return top_k
    candidates.sort(key=lambda s: getattr(s, "score", 0.5), reverse=True)
    return candidates[:top_k]