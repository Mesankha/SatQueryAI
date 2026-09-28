"""
M0 Ingestion - Microsoft Planetary Computer STAC ingestion.
Queries PC STAC for Sentinel-2 L2A and Sentinel-1 GRD, downloads chips, writes catalog.
"""
import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

import httpx
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
import numpy as np
import yaml

from shared.schemas import SceneMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# STAC API endpoints
STAC_ROOT = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTIONS = {
    "sentinel-2-l2a": {
        "sensor": "Sentinel-2",
        "modality": "multispectral",
        "bands": ["B02", "B03", "B04", "B08"],
        "cloud_cover_max": 40,
    },
    "sentinel-1-grd": {
        "sensor": "Sentinel-1",
        "modality": "sar",
        "bands": ["VV", "VH"],
        "cloud_cover_max": None,
    },
}

# Default wavelengths for Sentinel-2 (nm) - used if not in metadata
SENTINEL2_WAVELENGTHS = {
    "B02": 492,   # Blue
    "B03": 560,   # Green
    "B04": 665,   # Red
    "B08": 833,   # NIR
}

# Default wavelengths for Sentinel-1 - not applicable (SAR)
SENTINEL1_WAVELENGTHS = {}


class UniformError(Exception):
    """Uniform error for M0 module."""
    def __init__(self, module: str, error_type: str, message: str):
        self.module = module
        self.error_type = error_type
        self.message = message
        super().__init__(f"[{module}] {error_type}: {message}")


def load_aoi_config(aoi_path: str) -> list[dict[str, Any]]:
    """Load AOI configuration from YAML file."""
    with open(aoi_path) as f:
        config = yaml.safe_load(f)
    return config.get("aois", [])


def validate_aois(aois: list[dict[str, Any]]) -> None:
    """Validate that all AOIs have valid coordinates."""
    for aoi in aois:
        if aoi.get("lon") is None or aoi.get("lat") is None:
            raise UniformError(
                module="M0",
                error_type="invalid_input",
                message=f"AOI '{aoi['name']}' has null coordinates. Please provide valid lon/lat in {aoi_path}."
            )


def get_sas_token(collection: str, asset_href: str, client: httpx.Client) -> str:
    """Get SAS token for a Planetary Computer asset."""
    sas_url = f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"
    try:
        response = client.post(sas_url, json={"href": asset_href}, timeout=30)
        response.raise_for_status()
        return response.json()["token"]
    except Exception as e:
        logger.warning(f"Failed to get SAS token for {asset_href}: {e}")
        return ""


def search_stac(
    client: httpx.Client,
    collection: str,
    bbox: list[float],
    start_date: str,
    end_date: str,
    cloud_cover_max: Optional[float] = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search STAC API for items matching criteria."""
    search_url = f"{STAC_ROOT}/collections/{collection}/items"
    params = {
        "bbox": ",".join(map(str, bbox)),
        "datetime": f"{start_date}/{end_date}",
        "limit": limit,
    }
    if cloud_cover_max is not None:
        params["query"] = json.dumps({"eo:cloud_cover": {"lt": cloud_cover_max}})

    try:
        response = client.get(search_url, params=params, timeout=60)
        response.raise_for_status()
        return response.json().get("features", [])
    except Exception as e:
        logger.warning(f"STAC search failed for {collection}: {e}")
        return []


def download_chip(
    asset_href: str,
    bands: list[str],
    output_path: Path,
    size: int = 512,
    resolution: float = 10.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Download a chip from a COG asset.
    Returns (stacked_bands_array, wavelengths_dict).
    """
    with rasterio.open(asset_href) as src:
        # Calculate window for center of image at target resolution
        # Use the first band to get bounds
        bounds = src.bounds
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.top + bounds.bottom) / 2

        # Calculate window in pixel coordinates
        transform = src.transform
        col, row = ~transform * (center_x, center_y)
        col, row = int(col), int(row)
        half_size = size // 2
        window = Window(col - half_size, row - half_size, size, size)

        # Read bands
        band_data = []
        wavelengths = {}
        for i, band_name in enumerate(bands, 1):
            try:
                # Try to find band by name in descriptions
                band_idx = None
                for idx, desc in enumerate(src.descriptions, 1):
                    if desc and band_name.lower() in desc.lower():
                        band_idx = idx
                        break
                if band_idx is None:
                    band_idx = i  # fallback to index

                data = src.read(band_idx, window=window)
                band_data.append(data)

                # Get wavelength from metadata or use defaults
                if "sentinel-2" in asset_href.lower() or "s2" in asset_href.lower():
                    wavelengths[band_name] = SENTINEL2_WAVELENGTHS.get(band_name, 0)
                else:
                    wavelengths[band_name] = 0
            except Exception as e:
                logger.warning(f"Failed to read band {band_name}: {e}")
                band_data.append(np.zeros((size, size), dtype=np.float32))
                wavelengths[band_name] = 0

    if not band_data:
        raise UniformError(module="M0", error_type="invalid_input", message="No bands read successfully")

    stacked = np.stack(band_data, axis=0).astype(np.float32)

    # Write output GeoTIFF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=len(bands),
        dtype=stacked.dtype,
        crs=src.crs,
        transform=from_bounds(
            center_x - half_size * resolution,
            center_y - half_size * resolution,
            center_x + half_size * resolution,
            center_y + half_size * resolution,
            size,
            size,
        ),
    ) as dst:
        dst.write(stacked)
        dst.descriptions = tuple(bands)

    return stacked, wavelengths


def init_catalog_db(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite catalog database."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scenes (
            scene_id TEXT PRIMARY KEY,
            sensor TEXT NOT NULL,
            modality TEXT NOT NULL,
            acquisition_time TEXT NOT NULL,
            geometry_wkt TEXT NOT NULL,
            file_path TEXT,
            cloud_cover REAL,
            object TEXT,
            paired_scene_id TEXT,
            wavelengths_nm TEXT
        )
    """)
    conn.commit()
    return conn


def write_scene_to_db(conn: sqlite3.Connection, scene: SceneMetadata) -> None:
    """Write a scene to the catalog database."""
    conn.execute("""
        INSERT OR REPLACE INTO scenes
        (scene_id, sensor, modality, acquisition_time, geometry_wkt, file_path, cloud_cover, object, paired_scene_id, wavelengths_nm)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        scene.scene_id,
        scene.sensor,
        scene.modality,
        scene.acquisition_time.isoformat(),
        scene.geometry_wkt,
        scene.file_path,
        scene.cloud_cover,
        scene.object,
        scene.paired_scene_id,
        json.dumps(scene.wavelengths_nm) if scene.wavelengths_nm else None,
    ))
    conn.commit()


def create_bbox_from_point(lon: float, lat: float, buffer_deg: float = 0.05) -> list[float]:
    """Create a bounding box around a point."""
    return [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]


def geometry_to_wkt(bbox: list[float]) -> str:
    """Convert bbox to WKT polygon."""
    minx, miny, maxx, maxy = bbox
    return f"POLYGON(({minx} {miny}, {maxx} {miny}, {maxx} {maxy}, {minx} {maxy}, {minx} {miny}))"


def ingest_aois(
    aoi_path: str,
    db_path: str = "./data/catalog.db",
    scenes_dir: str = "./data/scenes",
    eval_splits_path: str = "./data/eval_splits.json",
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    max_per_aoi_per_sensor: int = 4,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Main ingestion function.
    """
    aois = load_aoi_config(aoi_path)
    validate_aois(aois)

    if dry_run:
        logger.info("DRY RUN: Would ingest the following AOIs:")
        for aoi in aois:
            logger.info(f"  {aoi['name']}: lon={aoi['lon']}, lat={aoi['lat']}")
        return {"status": "dry_run", "aois": len(aois)}

    # Initialize database
    conn = init_catalog_db(db_path)
    Path(scenes_dir).mkdir(parents=True, exist_ok=True)

    # HTTP client for STAC and SAS
    client = httpx.Client(timeout=60.0)

    eval_splits = {}
    total_scenes = 0

    try:
        for aoi_idx, aoi in enumerate(aois):
            aoi_name = aoi["name"]
            lon = aoi["lon"]
            lat = aoi["lat"]
            bbox = create_bbox_from_point(lon, lat)
            wkt = geometry_to_wkt(bbox)

            logger.info(f"Processing AOI {aoi_idx + 1}/{len(aois)}: {aoi_name} at ({lon}, {lat})")

            for collection_name, collection_info in COLLECTIONS.items():
                sensor = collection_info["sensor"]
                modality = collection_info["modality"]
                bands = collection_info["bands"]
                cloud_max = collection_info["cloud_cover_max"]

                items = search_stac(
                    client, collection_name, bbox, start_date, end_date,
                    cloud_cover_max=cloud_max, limit=max_per_aoi_per_sensor * 2
                )

                logger.info(f"  Found {len(items)} items in {collection_name}")

                count = 0
                for item in items:
                    if count >= max_per_aoi_per_sensor:
                        break

                    # Get asset hrefs for required bands
                    assets = item.get("assets", {})
                    band_hrefs = {}
                    for band in bands:
                        if band in assets:
                            band_hrefs[band] = assets[band]["href"]
                        else:
                            logger.warning(f"Band {band} not found in item {item['id']}")
                            break

                    if len(band_hrefs) != len(bands):
                        continue

                    # Get SAS tokens and build signed URLs
                    signed_hrefs = {}
                    for band, href in band_hrefs.items():
                        token = get_sas_token(collection_name, href, client)
                        if token:
                            signed_hrefs[band] = f"{href}?{token}"
                        else:
                            signed_hrefs[band] = href

                    # Download chip
                    scene_id = item["id"]
                    output_path = Path(scenes_dir) / f"{scene_id}.tif"

                    try:
                        _, wavelengths = download_chip(
                            list(signed_hrefs.values())[0],  # Use first band's href for metadata
                            bands,
                            output_path,
                        )

                        # Create SceneMetadata
                        acquisition_time = datetime.fromisoformat(
                            item["properties"]["datetime"].replace("Z", "+00:00")
                        )
                        cloud_cover = item["properties"].get("eo:cloud_cover")

                        scene = SceneMetadata(
                            scene_id=scene_id,
                            sensor=sensor,
                            modality=modality,
                            acquisition_time=acquisition_time,
                            geometry_wkt=wkt,
                            file_path=str(output_path),
                            cloud_cover=cloud_cover,
                            object=None,
                            wavelengths_nm=wavelengths,
                        )

                        write_scene_to_db(conn, scene)
                        total_scenes += 1
                        count += 1

                        # Assign eval split: hold out one AOI per sensor for test
                        # For simplicity, use AOI_1 as test for S2, AOI_2 as test for S1
                        if (sensor == "Sentinel-2" and aoi_idx == 0) or \
                           (sensor == "Sentinel-1" and aoi_idx == 1):
                            split = "test"
                        else:
                            split = "train"
                        eval_splits[scene_id] = {"aoi": aoi_name, "split": split}

                        logger.info(f"    Ingested {scene_id} ({split})")

                    except Exception as e:
                        logger.warning(f"Failed to ingest {scene_id}: {e}")

    finally:
        client.close()
        conn.close()

    # Write eval splits
    with open(eval_splits_path, "w") as f:
        json.dump(eval_splits, f, indent=2)

    logger.info(f"Ingestion complete. Total scenes: {total_scenes}")
    return {"status": "success", "total_scenes": total_scenes, "eval_splits": len(eval_splits)}


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="M0 Ingestion from Planetary Computer")
    parser.add_argument("--aois", required=True, help="Path to AOI config YAML")
    parser.add_argument("--db", default="./data/catalog.db", help="Output catalog database path")
    parser.add_argument("--scenes-dir", default="./data/scenes", help="Output scenes directory")
    parser.add_argument("--eval-splits", default="./data/eval_splits.json", help="Output eval splits path")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-per-aoi", type=int, default=4, help="Max scenes per AOI per sensor")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without downloading")

    args = parser.parse_args()

    try:
        result = ingest_aois(
            aoi_path=args.aois,
            db_path=args.db,
            scenes_dir=args.scenes_dir,
            eval_splits_path=args.eval_splits,
            start_date=args.start_date,
            end_date=args.end_date,
            max_per_aoi_per_sensor=args.max_per_aoi,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2))
    except UniformError as e:
        logger.error(e.message)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()