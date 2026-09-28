"""
M2 Wiring Patch - Fixes the db_path propagation bug in m2_adapter.py.

The original m2_adapter.py uses get_catalog() which doesn't accept a db_path.
This patch provides a fixed version that uses Catalog() directly.

To apply this patch:
  1. The dispatch_table.py already imports m2_retrieve which uses real_m2_retrieve
  2. This file provides a corrected real_m2_retrieve that honors the db_path
  3. The M0 catalog path can be configured via M0_CATALOG_DB env var
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Setup path
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
_M2_PATH = _PROJECT_ROOT / "M2_retrieval"
if str(_M2_PATH) not in sys.path:
    sys.path.insert(0, str(_M2_PATH))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Default catalog path (can be overridden by env var)
DEFAULT_CATALOG_DB = os.getenv("M0_CATALOG_DB", "./data/catalog.db")


def _resolve_catalog_path(db_path: Optional[str] = None) -> str:
    """Resolve the catalog DB path."""
    if db_path:
        return db_path
    env_path = os.getenv("M0_CATALOG_DB")
    if env_path:
        return env_path
    return DEFAULT_CATALOG_DB


def real_m2_retrieve_fixed(m1_query, top_k: int = 10, db_path: Optional[str] = None) -> List[Any]:
    """
    Fixed version of real_m2_retrieve that correctly uses the db_path argument.
    Reads from M0 catalog, runs through M2 satellite pipeline.
    """
    from m0_catalog.catalog import Catalog
    from m0_catalog.image_resolver import resolve_image

    actual_db_path = _resolve_catalog_path(db_path)
    logger.debug(f"M2 reading from catalog: {actual_db_path}")

    # 1. Load scenes from M0
    catalog = None
    try:
        catalog = Catalog(actual_db_path, read_only=True)
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

    logger.info(f"M0 catalog has {len(all_scenes)} scenes (from {actual_db_path})")

    # 2-6. Run through M2 pipeline
    try:
        # Import the converters from the m2_adapter module
        from m5_controller.m2_adapter import (
            m0_to_m2_scene, m1_to_m2_query, m2_to_m0_scene, _get_m2_pipeline
        )

        m2_scenes = [m0_to_m2_scene(s) for s in all_scenes]
        m2_query = m1_to_m2_query(m1_query)

        SatelliteMetadataFilter, score_engine, ranker = _get_m2_pipeline()

        # Stage 1: Metadata filtering
        filtered = SatelliteMetadataFilter.filter_candidates(m2_scenes, m2_query)
        logger.info(f"Stage 1 (filter): {len(m2_scenes)} -> {len(filtered)} candidates")
        if not filtered:
            filtered = m2_scenes[:50]

        # Stage 2: Semantic ranking
        ranked, encoder_used = ranker.rank_candidates(filtered, m2_query, top_k=top_k * 2)
        logger.info(f"Stage 2 (semantic rank): encoder={encoder_used}, top_k={top_k * 2}")

        # Stage 3: Multi-factor score fusion
        final = score_engine.score_and_rank_candidates(ranked, m2_query, top_k=top_k)
        logger.info(f"Stage 3 (score fusion): returned {len(final)} scenes")
    except Exception as e:
        logger.error(f"M2 pipeline error: {e}. Using scenes as-is.")
        final = m2_scenes[:top_k] if "m2_scenes" in dir() else all_scenes[:top_k]

    # 7. Convert back to shared.SceneMetadata
    results = []
    for m2_scene in final:
        try:
            shared_scene = m2_to_m0_scene(m2_scene)
            # 8. Resolve URLs to local cached paths
            if shared_scene.file_path and (shared_scene.file_path.startswith("http") or shared_scene.file_path.startswith("s3://")):
                resolved = resolve_image(shared_scene.file_path)
                if resolved:
                    shared_scene.file_path = resolved
            results.append(shared_scene)
        except Exception as e:
            logger.debug(f"Failed to convert scene: {e}")
            continue

    return results[:top_k]


# Monkey-patch the m2_adapter module on import
def _install_patch():
    """Install the fixed real_m2_retrieve into the m2_adapter module."""
    try:
        import m5_controller.m2_adapter as m2_adapter_module
        m2_adapter_module.real_m2_retrieve = real_m2_retrieve_fixed
        logger.info("M2 wiring patch installed")
    except Exception as e:
        logger.warning(f"Could not install M2 patch: {e}")


# Auto-install on import
_install_patch()