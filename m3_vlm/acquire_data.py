"""
M3 Data Acquisition — Complete real-data pipeline.

USAGE EXAMPLES:

  # For FINE-TUNING (one-time, on 3050 laptop):
  python -m m3_vlm.acquire_data train \\
      --source ./data/bigearthnet \\
      --output ./data/bigearthnet_lora \\
      --max-samples 2000

  # For DEMO (queries any time, no download):
  python -m m3_vlm.acquire_data demo \\
      --bbox 77.0 28.5 77.5 29.0 \\
      --start-date 2024-01-01 \\
      --end-date 2024-12-31 \\
      --output ./data/demo_delhi

  # For INDIAN CITIES (curated list, ready for demo):
  python -m m3_vlm.acquire_data cities \\
      --cities delhi mumbai bangalore \\
      --output ./data/indian_cities
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ===========================================================================
# INDIAN CITIES COORDINATES
# ===========================================================================
# Bounding boxes for major Indian cities (min_lon, min_lat, max_lon, max_lat)
# Coverage: ~50km x 50km around city center
INDIAN_CITIES = {
    "delhi": (77.05, 28.40, 77.35, 28.80),       # Delhi NCR
    "mumbai": (72.75, 18.85, 73.05, 19.30),      # Mumbai + Thane
    "bangalore": (77.45, 12.85, 77.75, 13.15),   # Bangalore urban
    "chennai": (80.15, 12.95, 80.40, 13.25),     # Chennai
    "kolkata": (88.25, 22.45, 88.55, 22.75),     # Kolkata
    "hyderabad": (78.30, 17.25, 78.60, 17.55),   # Hyderabad
    "pune": (73.70, 18.45, 74.00, 18.75),        # Pune
    "ahmedabad": (72.50, 22.95, 72.80, 23.25),   # Ahmedabad
    "jaipur": (75.65, 26.80, 75.95, 27.10),      # Jaipur
    "lucknow": (80.85, 26.75, 81.15, 27.05),     # Lucknow
    "guwahati": (91.55, 26.05, 91.85, 26.35),    # Guwahati (Assam)
    "kerala": (76.20, 9.95, 76.50, 10.50),       # Kerala (for floods)
    "assam": (90.50, 25.50, 95.00, 27.50),       # Assam state
    "rajasthan": (72.50, 26.50, 74.00, 28.00),   # Rajasthan desert
    "westbengal": (87.50, 22.50, 89.00, 24.00),  # West Bengal
}

# Default label per city (for synthetic VQA in absence of real labels)
CITY_LAND_COVER = {
    "delhi": "urban", "mumbai": "urban", "bangalore": "urban",
    "chennai": "coastal", "kolkata": "urban", "hyderabad": "urban",
    "pune": "urban", "ahmedabad": "urban", "jaipur": "urban",
    "lucknow": "urban", "guwahati": "urban", "kerala": "agriculture",
    "assam": "water", "rajasthan": "barren", "westbengal": "agriculture",
}


# ===========================================================================
# MODE 1: TRAINING DATA — BigEarthNet
# ===========================================================================
def acquire_bigearthnet(
    source_dir: Path,
    output_dir: Path,
    max_samples: int = 5000,
    min_label_quality: bool = True,
) -> int:
    """
    Convert a BigEarthNet directory into the LoRA training format.

    BigEarthNet directory structure (after download):
        bigearthnet/
            S2A_MSIL2A_20170613T101031_8_36/
                S2A_MSIL2A_20170613T101031_8_36_B02.tif
                S2A_MSIL2A_20170613T101031_8_36_B03.tif
                ... (12 bands total)
                S2A_MSIL2A_20170613T101031_8_36_labels_metadata.json

    HOW TO GET BigEarthNet:
        1. Register at http://bigearth.net/
        2. Download the "BigEarthNet-MM" (RGB only, ~15GB) or
           the full BigEarthNet-S2 (multi-band, ~68GB)
        3. Extract to ./data/raw/bigearthnet/
    """
    if not source_dir.exists():
        logger.error(f"BigEarthNet source not found: {source_dir}")
        logger.error("Download from: http://bigearth.net/")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Find all patch directories
    patch_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(patch_dirs)} patch directories in {source_dir}")
    random.shuffle(patch_dirs)
    if max_samples > 0:
        patch_dirs = patch_dirs[:max_samples]

    captions_path = output_dir / "captions.jsonl"
    vqa_path = output_dir / "vqa.jsonl"
    referring_path = output_dir / "referring.jsonl"

    # Land cover labels (BigEarthNet 19-class)
    BEN_LABELS = [
        "urban fabric", "industrial units", "arable land", "permanent crops",
        "pastures", "complex cultivation patterns", "agriculture with natural vegetation",
        "agro-forestry areas", "broad-leaved forest", "coniferous forest", "mixed forest",
        "natural grassland", "moors and heathland", "sclerophyllous vegetation",
        "transitional woodland-shrub", "beaches-dunes-sands", "inland wetlands",
        "coastal wetlands", "inland waters", "marine waters",
    ]

    # Map to simpler categories
    LABEL_MAP = {
        "urban fabric": "urban", "industrial units": "urban",
        "arable land": "agriculture", "permanent crops": "agriculture",
        "pastures": "agriculture", "complex cultivation patterns": "agriculture",
        "agriculture with natural vegetation": "agriculture",
        "agro-forestry areas": "forest", "broad-leaved forest": "forest",
        "coniferous forest": "forest", "mixed forest": "forest",
        "natural grassland": "grassland", "moors and heathland": "grassland",
        "sclerophyllous vegetation": "vegetation",
        "transitional woodland-shrub": "forest",
        "beaches-dunes-sands": "barren", "inland wetlands": "wetland",
        "coastal wetlands": "wetland", "inland waters": "water",
        "marine waters": "water",
    }

    n_processed = 0
    n_skipped = 0
    n_caption = n_vqa = n_ref = 0

    with open(captions_path, "w") as cap_f, \
         open(vqa_path, "w") as vqa_f, \
         open(referring_path, "w") as ref_f:

        for i, patch_dir in enumerate(patch_dirs):
            if (i + 1) % 200 == 0:
                logger.info(f"Processed {i+1}/{len(patch_dirs)} patches "
                            f"(kept={n_processed}, skipped={n_skipped})")

            # Read labels
            label_file = patch_dir / "labels_metadata.json"
            if not label_file.exists():
                label_file = patch_dir / f"{patch_dir.name}_labels_metadata.json"
            if not label_file.exists():
                n_skipped += 1
                continue

            try:
                with open(label_file) as f:
                    labels_data = json.load(f)
            except Exception as e:
                logger.debug(f"Skip {patch_dir.name}: {e}")
                n_skipped += 1
                continue

            raw_labels = labels_data.get("labels", [])
            if min_label_quality and len(raw_labels) < 1:
                n_skipped += 1
                continue
            primary_raw = raw_labels[0] if raw_labels else "unknown"
            primary = LABEL_MAP.get(primary_raw, primary_raw)

            # Find the RGB image (B04, B03, B02 for Sentinel-2)
            b04 = patch_dir / f"{patch_dir.name}_B04.tif"
            b03 = patch_dir / f"{patch_dir.name}_B03.tif"
            b02 = patch_dir / f"{patch_dir.name}_B02.tif"

            if not all(p.exists() for p in [b02, b03, b04]):
                # Fall back to .png if BigEarthNet-MM format
                rgb_png = patch_dir / f"{patch_dir.name}.png"
                if rgb_png.exists():
                    source_image = rgb_png
                else:
                    n_skipped += 1
                    continue
            else:
                # Combine B02, B03, B04 into RGB image
                source_image = None
                try:
                    import rasterio
                    import numpy as np
                    from PIL import Image
                    with rasterio.open(b04) as src:
                        r = src.read(1)
                    with rasterio.open(b03) as src:
                        g = src.read(1)
                    with rasterio.open(b02) as src:
                        b = src.read(1)
                    def _norm(x):
                        x = x.astype(np.float32)
                        mn, mx = np.percentile(x, [2, 98])
                        if mx - mn < 1e-6:
                            mx = mn + 1
                        return np.clip((x - mn) / (mx - mn) * 255, 0, 255).astype(np.uint8)
                    arr = np.stack([_norm(r), _norm(g), _norm(b)], axis=-1)
                    source_image = images_dir / f"{patch_dir.name}.png"
                    Image.fromarray(arr).save(source_image)
                except ImportError:
                    logger.error("Need rasterio + Pillow for multi-band GeoTIFFs")
                    n_skipped += 1
                    continue
                except Exception as e:
                    logger.debug(f"Conversion failed for {patch_dir.name}: {e}")
                    n_skipped += 1
                    continue

            if source_image is None or not source_image.exists():
                n_skipped += 1
                continue

            # Target filename (in output images/)
            target_filename = source_image.name
            target_path = images_dir / target_filename
            if not target_path.exists():
                if source_image != target_path:
                    try:
                        target_path.symlink_to(source_image.resolve())
                    except OSError:
                        import shutil
                        shutil.copy2(source_image, target_path)

            # 1. Caption
            caption = _make_caption(primary)
            cap_f.write(json.dumps({
                "image_id": target_filename,
                "caption": caption,
            }) + "\n")
            n_caption += 1

            # 2. VQA pairs
            for qa in _make_vqa(primary):
                qa["image_id"] = target_filename
                vqa_f.write(json.dumps(qa) + "\n")
                n_vqa += 1

            # 3. Referring expression
            ref = _make_referring(primary)
            ref["image_id"] = target_filename
            ref_f.write(json.dumps(ref) + "\n")
            n_ref += 1

            n_processed += 1

    logger.info(
        f"BigEarthNet done. Processed={n_processed}, Skipped={n_skipped}, "
        f"Captions={n_caption}, VQA={n_vqa}, Referring={n_ref}"
    )
    return n_processed


# ===========================================================================
# MODE 2: DEMO DATA — Microsoft Planetary Computer
# ===========================================================================
def acquire_planetary_computer(
    output_dir: Path,
    bbox: Tuple[float, float, float, float],
    date_range: Tuple[str, str],
    max_samples: int = 100,
    collections: Tuple[str, ...] = ("sentinel-2-l2a",),
    max_cloud_cover: int = 20,
    download_thumbnails: bool = True,
) -> int:
    """
    Query Microsoft Planetary Computer for any AOI and prepare data.

    Args:
        output_dir: Where to save (images + JSONL)
        bbox: (min_lon, min_lat, max_lon, max_lat)
        date_range: (start_date, end_date) in YYYY-MM-DD
        max_samples: Max scenes to process
        collections: STAC collections
        max_cloud_cover: Cloud cover threshold (0-100)
        download_thumbnails: Download small thumbnails for LoRA training

    Returns:
        Number of scenes processed
    """
    try:
        import planetary_computer
        import pystac_client
    except ImportError:
        logger.error("Install: pip install planetary-computer pystac-client")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    logger.info(
        f"Querying Planetary Computer: bbox={bbox}, dates={date_range}, "
        f"collections={collections}, max_cloud={max_cloud_cover}%"
    )

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1/"
    )

    # Build query
    query = None
    if "sentinel-2" in collections[0]:
        query = {"eo:cloud_cover": {"lt": max_cloud_cover}}

    search = catalog.search(
        collections=list(collections),
        bbox=list(bbox),
        datetime=f"{date_range[0]}/{date_range[1]}",
        query=query,
        limit=max_samples,
    )
    items = list(search.items())
    logger.info(f"Found {len(items)} STAC items")

    # Determine land cover hint based on bbox
    lc_hint = _bbox_to_land_cover(bbox)

    captions_path = output_dir / "captions.jsonl"
    vqa_path = output_dir / "vqa.jsonl"
    referring_path = output_dir / "referring.jsonl"
    scenes_meta_path = output_dir / "scenes_metadata.jsonl"

    n_processed = 0
    with open(captions_path, "w") as cap_f, \
         open(vqa_path, "w") as vqa_f, \
         open(referring_path, "w") as ref_f, \
         open(scenes_meta_path, "w") as meta_f:

        for i, item in enumerate(items):
            if n_processed >= max_samples:
                break
            try:
                # Get the best visual asset
                asset_key = None
                for k in ("visual", "thumbnail", "rendered_preview"):
                    if k in item.assets:
                        asset_key = k
                        break
                if asset_key is None:
                    logger.debug(f"Skip {item.id}: no visual asset")
                    continue
                asset = item.assets[asset_key]
                href = asset.href

                # Sign URL (free, no auth required)
                try:
                    signed_href = planetary_computer.sign(href)
                except Exception as e:
                    logger.debug(f"Skip {item.id}: sign failed: {e}")
                    continue

                # Download thumbnail
                image_filename = f"pc_{item.id}.png"
                target_image = images_dir / image_filename
                if download_thumbnails and not target_image.exists():
                    try:
                        _download_url(signed_href, target_image)
                    except Exception as e:
                        logger.debug(f"Download failed for {item.id}: {e}")
                        # Still record metadata even if download fails
                        target_image = None

                if target_image is not None and not target_image.exists():
                    target_image = None

                if target_image is not None:
                    # Write training JSONL entries
                    cap_f.write(json.dumps({
                        "image_id": image_filename,
                        "caption": _make_caption(lc_hint, region=item.properties.get("s2:mgrs_tile")),
                    }) + "\n")
                    for qa in _make_vqa(lc_hint):
                        qa["image_id"] = image_filename
                        vqa_f.write(json.dumps(qa) + "\n")
                    ref = _make_referring(lc_hint)
                    ref["image_id"] = image_filename
                    ref_f.write(json.dumps(ref) + "\n")

                # Always write scene metadata (for M0 catalog)
                meta_f.write(json.dumps({
                    "scene_id": item.id,
                    "collection": item.collection_id,
                    "datetime": str(item.datetime),
                    "bbox": list(bbox),
                    "geometry": item.geometry,
                    "assets": list(item.assets.keys()),
                    "signed_href": signed_href if not download_thumbnails else None,
                    "local_image": str(target_image) if target_image else None,
                    "properties": {k: v for k, v in item.properties.items()
                                   if isinstance(v, (str, int, float, bool))},
                }) + "\n")

                n_processed += 1
                if n_processed % 10 == 0:
                    logger.info(f"Processed {n_processed}/{len(items)}")

            except Exception as e:
                logger.debug(f"Skip {item.id}: {e}")
                continue

    logger.info(f"Planetary Computer done. Processed {n_processed} scenes")
    return n_processed


# ===========================================================================
# MODE 3: INDIAN CITIES — pre-curated bboxes
# ===========================================================================
def acquire_indian_cities(
    output_dir: Path,
    cities: List[str],
    max_per_city: int = 50,
    date_range: Tuple[str, str] = ("2024-01-01", "2024-12-31"),
    collections: Tuple[str, ...] = ("sentinel-2-l2a",),
    max_cloud_cover: int = 20,
) -> int:
    """Acquire demo data for a list of Indian cities."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for city in cities:
        if city not in INDIAN_CITIES:
            logger.warning(f"Unknown city: {city}. Available: {list(INDIAN_CITIES.keys())}")
            continue
        bbox = INDIAN_CITIES[city]
        city_dir = output_dir / city
        logger.info(f"=== {city.upper()} ===")
        n = acquire_planetary_computer(
            city_dir, bbox, date_range,
            max_samples=max_per_city,
            collections=collections,
            max_cloud_cover=max_cloud_cover,
        )
        total += n

    # Also generate a master catalog file
    catalog = {}
    for city in cities:
        city_dir = output_dir / city
        meta_file = city_dir / "scenes_metadata.jsonl"
        if meta_file.exists():
            catalog[city] = []
            with open(meta_file) as f:
                for line in f:
                    if line.strip():
                        catalog[city].append(json.loads(line))

    with open(output_dir / "master_catalog.json", "w") as f:
        json.dump(catalog, f, indent=2, default=str)
    logger.info(f"Total scenes across {len(cities)} cities: {total}")
    logger.info(f"Master catalog saved to {output_dir/'master_catalog.json'}")
    return total


# ===========================================================================
# M0 CATALOG POPULATION (from any source)
# ===========================================================================
def populate_m0_catalog(
    data_root: Path,
    db_path: str = "./data/catalog.db",
    origins: List[str] = None,
) -> int:
    """
    Walk a data directory and add all scenes to M0 catalog.

    Expected structure:
        data_root/
            city_name/
                images/      <- actual image files
                scenes_metadata.jsonl   <- metadata (optional)
    """
    try:
        from m0_catalog.catalog import Catalog
    except ImportError:
        logger.error("M0 catalog not available. Run from satquery/ directory.")
        return 0

    catalog = Catalog(db_path, read_only=False)
    count = 0

    for city_dir in data_root.iterdir():
        if not city_dir.is_dir():
            continue
        meta_file = city_dir / "scenes_metadata.jsonl"
        if not meta_file.exists():
            continue

        with open(meta_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    from shared.schemas import SceneMetadata
                    from datetime import datetime
                    scene = SceneMetadata(
                        scene_id=rec["scene_id"],
                        sensor="Sentinel-2" if "sentinel-2" in rec["collection"] else "Sentinel-1",
                        modality="optical" if "sentinel-2" in rec["collection"] else "sar",
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
                        object=CITY_LAND_COVER.get(city_dir.name, "unknown"),
                        paired_scene_id=None,
                    )
                    catalog.insert_scene(scene, dataset_origin=f"planetary_{city_dir.name}")
                    count += 1
                except Exception as e:
                    logger.debug(f"Insert failed: {e}")

    catalog.close()
    logger.info(f"Added {count} scenes to M0 catalog at {db_path}")
    return count


# ===========================================================================
# Helper functions
# ===========================================================================
def _download_url(url: str, target: Path) -> None:
    """Download a URL to a local file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url, headers={"User-Agent": "SatQuery-AI/1.0 (research)"}
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        with open(target, "wb") as f:
            f.write(response.read())


def _make_caption(label: str, region: Optional[str] = None) -> str:
    if region:
        return f"Sentinel-2 image of {label} land cover in tile {region}."
    templates = [
        f"Satellite image showing a {label} area.",
        f"Aerial view of {label} terrain with characteristic features.",
        f"Remote sensing scene depicting {label} landscape.",
        f"Overview of a {label} region captured by Sentinel-2 satellite.",
    ]
    return random.choice(templates)


def _make_vqa(label: str) -> List[Dict]:
    pairs = [
        {"question": "What is the land cover in this image?",
         "answer": f"The land cover is {label}."},
        {"question": "What type of terrain is shown?",
         "answer": f"This image shows {label} terrain."},
        {"question": "What is the dominant land cover class?",
         "answer": f"The dominant land cover is {label}."},
    ]
    if label == "urban":
        pairs.append({"question": "Is this an urban area?", "answer": "Yes"})
    elif label == "water":
        pairs.append({"question": "Is this area water?", "answer": "Yes"})
    elif label == "forest":
        pairs.append({"question": "Is this area forested?", "answer": "Yes"})
    elif label == "agriculture":
        pairs.append({"question": "Is this area agricultural?", "answer": "Yes"})
    return pairs


def _make_referring(label: str) -> Dict:
    return {
        "expression": f"the {label} area",
        "bbox": [0.2, 0.2, 0.8, 0.8],
    }


def _bbox_to_land_cover(bbox: Tuple[float, float, float, float]) -> str:
    """Infer land cover from bbox center coordinates (heuristic)."""
    center_lon = (bbox[0] + bbox[2]) / 2
    center_lat = (bbox[1] + bbox[3]) / 2

    # Match against known city bboxes
    for city, city_bbox in INDIAN_CITIES.items():
        if (city_bbox[0] <= center_lon <= city_bbox[2] and
            city_bbox[1] <= center_lat <= city_bbox[3]):
            return CITY_LAND_COVER.get(city, "mixed")

    # Geographic heuristics
    if center_lat > 28 and 75 < center_lon < 78:  # Indo-Gangetic plain
        return "agriculture"
    if center_lat < 10:  # Southern tip
        return "forest"
    return "mixed"


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Acquire real satellite data for SatQuery (training + demo)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train (BigEarthNet)
    train_p = subparsers.add_parser("train", help="Process BigEarthNet for LoRA training")
    train_p.add_argument("--source", required=True, help="BigEarthNet source directory")
    train_p.add_argument("--output", default="./data/bigearthnet_lora", help="Output directory")
    train_p.add_argument("--max-samples", type=int, default=5000)

    # demo (Planetary Computer, one bbox)
    demo_p = subparsers.add_parser("demo", help="Query Planetary Computer for one AOI")
    demo_p.add_argument("--bbox", nargs=4, type=float, required=True,
                       metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                       help="Bounding box, e.g. Delhi: 77.0 28.5 77.5 29.0")
    demo_p.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    demo_p.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    demo_p.add_argument("--output", default="./data/demo_query", help="Output directory")
    demo_p.add_argument("--max-samples", type=int, default=100)
    demo_p.add_argument("--collections", nargs="+", default=["sentinel-2-l2a"])
    demo_p.add_argument("--max-cloud", type=int, default=20)
    demo_p.add_argument("--no-download", action="store_true",
                        help="Skip downloading thumbnails (URLs only)")

    # cities (Planetary Computer, multiple Indian cities)
    cities_p = subparsers.add_parser("cities", help="Acquire data for multiple Indian cities")
    cities_p.add_argument("--cities", nargs="+", required=True,
                          help="City names: delhi mumbai bangalore chennai kolkata "
                               "hyderabad pune ahmedabad jaipur lucknow guwahati kerala "
                               "assam rajasthan westbengal")
    cities_p.add_argument("--output", default="./data/indian_cities", help="Output directory")
    cities_p.add_argument("--max-per-city", type=int, default=50)
    cities_p.add_argument("--start-date", default="2024-01-01")
    cities_p.add_argument("--end-date", default="2024-12-31")
    cities_p.add_argument("--max-cloud", type=int, default=20)

    # catalog (populate M0 from any acquired data)
    cat_p = subparsers.add_parser("catalog", help="Populate M0 catalog from data dir")
    cat_p.add_argument("--data-root", required=True, help="Root directory with city subdirs")
    cat_p.add_argument("--db", default="./data/catalog.db", help="M0 catalog database path")

    args = parser.parse_args()

    if args.command == "train":
        acquire_bigearthnet(
            Path(args.source), Path(args.output), args.max_samples
        )
    elif args.command == "demo":
        acquire_planetary_computer(
            Path(args.output), tuple(args.bbox),
            (args.start_date, args.end_date),
            max_samples=args.max_samples,
            collections=tuple(args.collections),
            max_cloud_cover=args.max_cloud,
            download_thumbnails=not args.no_download,
        )
    elif args.command == "cities":
        acquire_indian_cities(
            Path(args.output), args.cities,
            max_per_city=args.max_per_city,
            date_range=(args.start_date, args.end_date),
            max_cloud_cover=args.max_cloud,
        )
    elif args.command == "catalog":
        populate_m0_catalog(Path(args.data_root), args.db)


if __name__ == "__main__":
    main()