"""
M0 BigEarthNet.txt Adapter
Ingests BigEarthNet metadata and patches into catalog.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.schemas import SceneMetadata
from .catalog import Catalog
from .config import get_m0_config


def parse_bigearthnet_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
    """
    Parse BigEarthNet.txt metadata file.
    Expected format: CSV or JSON with patch information.
    """
    if metadata_path.suffix == ".json":
        with open(metadata_path) as f:
            return json.load(f)
    elif metadata_path.suffix == ".csv":
        import csv
        with open(metadata_path) as f:
            reader = csv.DictReader(f)
            return list(reader)
    else:
        raise ValueError(f"Unsupported metadata format: {metadata_path.suffix}")


def extract_patch_info(patch_dir: Path, patch_name: str) -> Dict[str, Any]:
    """Extract information from a BigEarthNet patch directory."""
    patch_path = patch_dir / patch_name
    info = {"patch_name": patch_name}

    # Look for label files
    labels_file = patch_path / f"{patch_name}_labels_metadata.json"
    if labels_file.exists():
        with open(labels_file) as f:
            labels_data = json.load(f)
            info["labels"] = labels_data.get("labels", [])
            info["tile_id"] = labels_data.get("tile_id", "")

    # Look for band files (Sentinel-2 has 12 bands)
    band_files = list(patch_path.glob("*_B*.tif"))
    info["bands"] = [f.name for f in band_files]

    # Check for RGB preview
    rgb_file = patch_path / f"{patch_name}_RGB.tif"
    if rgb_file.exists():
        info["rgb_path"] = str(rgb_file)

    return info


def create_scene_from_patch(patch_info: Dict[str, Any], patch_dir: Path, dataset_origin: str = "bigearthnet_txt") -> SceneMetadata:
    """Create SceneMetadata from BigEarthNet patch info."""
    import uuid

    patch_name = patch_info["patch_name"]

    # Extract coordinates from patch name (BigEarthNet format: S2A_MSIL1C_20170615_T32TMT_N0205_R051_T32TMT_20170615)
    # For simplicity, use a default location
    geometry_wkt = "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"

    # Determine primary land cover from labels
    labels = patch_info.get("labels", [])
    primary_object = labels[0] if labels else "unknown"

    # Use RGB file as primary image path if available
    file_path = patch_info.get("rgb_path") or str(patch_dir / patch_name)

    return SceneMetadata(
        scene_id=str(uuid.uuid4()),
        sensor="Sentinel-2",
        modality="multispectral",
        acquisition_time=None,  # Would need to parse from patch name
        geometry_wkt=geometry_wkt,
        file_path=file_path,
        cloud_cover=None,
        object=primary_object,
        paired_scene_id=None,
    )


def ingest_bigearthnet(
    catalog: Catalog,
    source_path: str,
    dataset_origin: str = "bigearthnet_txt",
    max_patches: Optional[int] = None,
) -> int:
    """
    Ingest BigEarthNet dataset into catalog.

    Args:
        catalog: Catalog instance
        source_path: Path to BigEarthNet root directory
        dataset_origin: Origin identifier for this dataset
        max_patches: Maximum number of patches to ingest (for testing)

    Returns:
        Number of scenes ingested
    """
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"BigEarthNet source not found: {source_path}")

    # Look for metadata file
    metadata_files = list(source.glob("*.json")) + list(source.glob("*.csv"))
    if not metadata_files:
        raise FileNotFoundError(f"No metadata file found in {source_path}")

    metadata_path = metadata_files[0]
    print(f"Loading metadata from {metadata_path}")

    metadata = parse_bigearthnet_metadata(metadata_path)
    print(f"Found {len(metadata)} patches in metadata")

    # Find patch directories
    patch_dirs = [d for d in source.iterdir() if d.is_dir() and not d.name.startswith(".")]

    ingested = 0
    for i, patch_meta in enumerate(metadata):
        if max_patches and ingested >= max_patches:
            break

        patch_name = patch_meta.get("patch_name") or patch_meta.get("image_id") or patch_meta.get("filename")
        if not patch_name:
            continue

        # Find matching patch directory
        patch_dir = None
        for d in patch_dirs:
            if patch_name in d.name:
                patch_dir = d
                break

        if not patch_dir:
            continue

        patch_info = extract_patch_info(source, patch_dir.name)
        patch_info.update(patch_meta)

        scene = create_scene_from_patch(patch_info, source, dataset_origin)

        # Copy image to catalog image_root if needed
        config = get_m0_config()
        target_dir = Path(config.output.image_root) / dataset_origin
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / f"{scene.scene_id}.tif"
        if Path(scene.file_path).exists():
            import shutil
            shutil.copy2(scene.file_path, target_path)
            scene.file_path = str(target_path)

        catalog.insert_scene(scene, dataset_origin)
        ingested += 1

        if ingested % 100 == 0:
            print(f"Ingested {ingested} patches...")

    print(f"BigEarthNet ingestion complete: {ingested} scenes")
    return ingested


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest BigEarthNet into M0 catalog")
    parser.add_argument("--source", default="./data/raw/bigearthnet", help="BigEarthNet source path")
    parser.add_argument("--max-patches", type=int, help="Max patches to ingest")
    parser.add_argument("--db", default="./data/catalog.db", help="Catalog database path")

    args = parser.parse_args()

    catalog = Catalog(args.db, read_only=False)
    ingest_bigearthnet(catalog, args.source, max_patches=args.max_patches)
    catalog.close()