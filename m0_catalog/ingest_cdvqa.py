"""
M0 CDVQA Adapter
Ingests CDVQA dataset (bi-temporal pairs + Q/A annotations) into catalog.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from shared.schemas import SceneMetadata
from .catalog import Catalog
from .config import get_m0_config


def parse_cdvqa_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
    """Parse CDVQA JSON metadata."""
    with open(metadata_path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def ingest_cdvqa(
    catalog: Catalog,
    source_path: str,
    dataset_origin: str = "cdvqa",
    max_pairs: Optional[int] = None,
) -> int:
    """
    Ingest CDVQA dataset into catalog.

    CDVQA contains bi-temporal pairs with Q/A annotations.
    Creates two scene records per pair linked by paired_scene_id.
    """
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"CDVQA source not found: {source_path}")

    # Look for metadata files
    metadata_files = list(source.glob("*.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No JSON metadata found in {source_path}")

    metadata_path = metadata_files[0]
    print(f"Loading CDVQA metadata from {metadata_path}")

    pairs = parse_cdvqa_metadata(metadata_path)
    print(f"Found {len(pairs)} bi-temporal pairs")

    config = get_m0_config()
    target_dir = Path(config.output.image_root) / dataset_origin
    target_dir.mkdir(parents=True, exist_ok=True)

    ingested = 0
    for i, pair in enumerate(pairs):
        if max_pairs and ingested >= max_pairs * 2:  # 2 scenes per pair
            break

        pair_id = str(uuid.uuid4())
        t1_info = pair.get("t1", {})
        t2_info = pair.get("t2", {})

        # Create scene for t1
        scene_t1 = SceneMetadata(
            scene_id=str(uuid.uuid4()),
            sensor=t1_info.get("sensor", "Sentinel-2"),
            modality=t1_info.get("modality", "optical"),
            acquisition_time=None,  # Would parse from metadata
            geometry_wkt=t1_info.get("geometry_wkt", "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"),
            file_path=t1_info.get("file_path"),
            cloud_cover=t1_info.get("cloud_cover"),
            object=t1_info.get("object"),
            paired_scene_id=None,  # Will set after t2 created
        )

        # Create scene for t2
        scene_t2 = SceneMetadata(
            scene_id=str(uuid.uuid4()),
            sensor=t2_info.get("sensor", "Sentinel-2"),
            modality=t2_info.get("modality", "optical"),
            acquisition_time=None,
            geometry_wkt=t2_info.get("geometry_wkt", "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"),
            file_path=t2_info.get("file_path"),
            cloud_cover=t2_info.get("cloud_cover"),
            object=t2_info.get("object"),
            paired_scene_id=None,
        )

        # Link them
        scene_t1.paired_scene_id = scene_t2.scene_id
        scene_t2.paired_scene_id = scene_t1.scene_id

        # Store Q/A in object field or as metadata
        qa_text = pair.get("qa", {}).get("question", "") + " " + pair.get("qa", {}).get("answer", "")
        scene_t1.object = (scene_t1.object or "") + f" | QA: {qa_text}" if scene_t1.object else f"QA: {qa_text}"

        # Copy images if they exist
        for scene in [scene_t1, scene_t2]:
            if scene.file_path and Path(scene.file_path).exists():
                target_path = target_dir / f"{scene.scene_id}.tif"
                import shutil
                shutil.copy2(scene.file_path, target_path)
                scene.file_path = str(target_path)

        catalog.insert_scene(scene_t1, dataset_origin)
        catalog.insert_scene(scene_t2, dataset_origin)
        ingested += 2

        if ingested % 50 == 0:
            print(f"Ingested {ingested} scenes ({ingested // 2} pairs)...")

    print(f"CDVQA ingestion complete: {ingested} scenes ({ingested // 2} pairs)")
    return ingested


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest CDVQA into M0 catalog")
    parser.add_argument("--source", default="./data/raw/cdvqa", help="CDVQA source path")
    parser.add_argument("--max-pairs", type=int, help="Max pairs to ingest")
    parser.add_argument("--db", default="./data/catalog.db", help="Catalog database path")

    args = parser.parse_args()

    catalog = Catalog(args.db, read_only=False)
    ingest_cdvqa(catalog, args.source, max_pairs=args.max_pairs)
    catalog.close()