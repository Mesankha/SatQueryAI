"""
M0 GEE Assam Adapter
Ingests Earth Engine exports for Assam region into catalog.
Requires GEE authentication.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from shared.schemas import SceneMetadata
from .catalog import Catalog
from .config import get_m0_config


def ingest_gee_assam(
    catalog: Catalog,
    project_id: Optional[str] = None,
    region: str = "Assam, India",
    date_range: Optional[List[str]] = None,
    dataset_origin: str = "gee_assam",
    max_scenes: Optional[int] = None,
) -> int:
    """
    Ingest GEE Assam dataset into catalog.

    Requires Earth Engine Python API and authentication.
    If GEE auth fails, returns 0 (fallback per blueprint §2.4).
    """
    # Try to import Earth Engine
    try:
        import ee
    except ImportError:
        print("Earth Engine Python API not installed. Skipping GEE ingestion.")
        print("Install with: pip install earthengine-api")
        return 0

    # Initialize Earth Engine
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        print("Earth Engine initialized successfully")
    except Exception as e:
        print(f"GEE authentication failed: {e}")
        print("Falling back to BigEarthNet + CDVQA only (per blueprint §2.4)")
        return 0

    # Define Assam region geometry
    # This is a simplified bbox for Assam
    assam_geometry = ee.Geometry.Rectangle([89.5, 24.0, 96.0, 28.5])

    # Default date range
    if date_range is None:
        date_range = ["2023-01-01", "2024-12-31"]

    start_date, end_date = date_range

    # Query Sentinel-2 collection
    s2_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(assam_geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .limit(max_scenes or 100)
    )

    # Query Sentinel-1 collection
    s1_collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(assam_geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .limit(max_scenes or 100)
    )

    config = get_m0_config()
    target_dir = Path(config.output.image_root) / dataset_origin
    target_dir.mkdir(parents=True, exist_ok=True)

    ingested = 0

    # Process Sentinel-2
    print("Processing Sentinel-2 scenes...")
    try:
        s2_list = s2_collection.getInfo()["features"]
        for img_info in s2_list:
            if max_scenes and ingested >= max_scenes:
                break

            props = img_info["properties"]
            geom = img_info["geometry"]

            scene = SceneMetadata(
                scene_id=str(uuid.uuid4()),
                sensor="Sentinel-2",
                modality="multispectral",
                acquisition_time=props.get("system:time_start"),
                geometry_wkt=geom_to_wkt(geom),
                file_path=None,  # Would need to download
                cloud_cover=props.get("CLOUDY_PIXEL_PERCENTAGE"),
                object=None,
                paired_scene_id=None,
            )

            catalog.insert_scene(scene, dataset_origin)
            ingested += 1

        print(f"Ingested {ingested} Sentinel-2 scenes")
    except Exception as e:
        print(f"Sentinel-2 query failed: {e}")

    # Process Sentinel-1
    print("Processing Sentinel-1 scenes...")
    try:
        s1_list = s1_collection.getInfo()["features"]
        for img_info in s1_list:
            if max_scenes and ingested >= max_scenes:
                break

            props = img_info["properties"]
            geom = img_info["geometry"]

            scene = SceneMetadata(
                scene_id=str(uuid.uuid4()),
                sensor="Sentinel-1",
                modality="sar",
                acquisition_time=props.get("system:time_start"),
                geometry_wkt=geom_to_wkt(geom),
                file_path=None,
                cloud_cover=None,
                object=None,
                paired_scene_id=None,
            )

            catalog.insert_scene(scene, dataset_origin)
            ingested += 1

        print(f"Ingested {ingested} total scenes")
    except Exception as e:
        print(f"Sentinel-1 query failed: {e}")

    return ingested


def geom_to_wkt(geom: Dict[str, Any]) -> str:
    """Convert EE geometry to WKT (simplified)."""
    if geom["type"] == "Polygon":
        coords = geom["coordinates"][0]
        pts = ", ".join(f"{c[0]} {c[1]}" for c in coords)
        return f"POLYGON(({pts}))"
    return "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest GEE Assam into M0 catalog")
    parser.add_argument("--project-id", help="GEE project ID")
    parser.add_argument("--region", default="Assam, India", help="Region name")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date")
    parser.add_argument("--end-date", default="2024-12-31", help="End date")
    parser.add_argument("--max-scenes", type=int, default=100, help="Max scenes")
    parser.add_argument("--db", default="./data/catalog.db", help="Catalog database path")

    args = parser.parse_args()

    catalog = Catalog(args.db, read_only=False)
    ingest_gee_assam(
        catalog,
        project_id=args.project_id,
        region=args.region,
        date_range=[args.start_date, args.end_date],
        max_scenes=args.max_scenes,
    )
    catalog.close()