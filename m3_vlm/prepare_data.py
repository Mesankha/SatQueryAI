"""
Real data preparation for M3 LoRA fine-tuning.

This script converts REAL BigEarthNet / Sentinel-2 imagery into the JSONL
training format expected by train_lora.py. It supports:

1. BigEarthNet (19-class land cover) — patches are pre-cropped Sentinel-2
   tiles with labels. Download from: http://bigearth.net/
2. Microsoft Planetary Computer — query real Sentinel-2 L2A scenes and
   download cloud-free chips for any AOI in the world.
3. Generic folder of GeoTIFFs/JPGs with a labels CSV.

Output: data/bigearthnet_lora/{images,captions.jsonl,vqa.jsonl,referring.jsonl}
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BigEarthNet 19-class land cover labels
# ---------------------------------------------------------------------------
BEN_LABELS_19 = [
    "urban fabric", "industrial units", "arable land", "permanent crops",
    "pastures", "complex cultivation patterns", "agriculture with natural vegetation",
    "agro-forestry areas", "broad-leaved forest", "coniferous forest", "mixed forest",
    "natural grassland", "moors and heathland", "sclerophyllous vegetation",
    "transitional woodland-shrub", "beaches-dunes-sands", "inland wetlands",
    "coastal wetlands", "inland waters", "marine waters",
]

# Simplified single-label mapping (use primary class)
BEN_PRIMARY_LABELS = {
    "urban fabric": "urban",
    "industrial units": "urban",
    "arable land": "agriculture",
    "permanent crops": "agriculture",
    "pastures": "agriculture",
    "complex cultivation patterns": "agriculture",
    "agriculture with natural vegetation": "agriculture",
    "agro-forestry areas": "forest",
    "broad-leaved forest": "forest",
    "coniferous forest": "forest",
    "mixed forest": "forest",
    "natural grassland": "grassland",
    "moors and heathland": "grassland",
    "sclerophyllous vegetation": "vegetation",
    "transitional woodland-shrub": "forest",
    "beaches-dunes-sands": "barren",
    "inland wetlands": "wetland",
    "coastal wetlands": "wetland",
    "inland waters": "water",
    "marine waters": "water",
}

# Question templates for VQA
VQA_TEMPLATES = [
    ("What is the land cover in this image?", "The land cover is {label}."),
    ("What type of terrain is shown?", "This image shows {label} terrain."),
    ("What is the dominant land cover class?", "The dominant land cover is {label}."),
    ("Is this an urban area?", "{answer}"),
    ("Is this area water?", "{answer}"),
    ("Is this area forested?", "{answer}"),
    ("Is this area agricultural?", "{answer}"),
    ("Which satellite captured this image?", "This is a Sentinel-2 satellite image."),
    ("What surface type is visible?", "The surface appears to be {label}."),
]

# Referring expression templates
REFERRING_TEMPLATES = [
    "the {label} area",
    "the {label} region",
    "the {label} zone",
    "the main {label} patch",
    "{label} in the center",
]


# ---------------------------------------------------------------------------
# BigEarthNet ingestion
# ---------------------------------------------------------------------------
def process_bigearthnet(
    source_dir: Path,
    output_dir: Path,
    max_samples: int = 5000,
) -> int:
    """Process a BigEarthNet directory into training JSONL files.

    BigEarthNet directory structure (after download):
      bigearthnet/
        S2A_MSIL2A_20170613T101031_8_36/
        S2A_MSIL2A_20170613T101031_8_37/
        ...
        Each patch dir has 12 band .tif files + labels_metadata.json

    Or use the new (post-2020) "BigEarthNet-MM" structure with separate
    RGB images and label files.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"BigEarthNet source not found: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    captions_path = output_dir / "captions.jsonl"
    vqa_path = output_dir / "vqa.jsonl"
    referring_path = output_dir / "referring.jsonl"

    # Find patch directories
    patch_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(patch_dirs)} patch directories")

    random.shuffle(patch_dirs)
    patch_dirs = patch_dirs[:max_samples]

    n_caption = n_vqa = n_ref = 0
    skipped = 0

    with open(captions_path, "w") as cap_f, \
         open(vqa_path, "w") as vqa_f, \
         open(referring_path, "w") as ref_f:

        for i, patch_dir in enumerate(patch_dirs):
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{len(patch_dirs)} patches")

            # Read labels
            labels_file = patch_dir / "labels_metadata.json"
            if not labels_file.exists():
                # Try old format
                labels_file = patch_dir / "patch_labels.json"
            if not labels_file.exists():
                skipped += 1
                continue

            try:
                with open(labels_file) as f:
                    labels_data = json.load(f)
            except Exception as e:
                logger.debug(f"Skip {patch_dir.name}: {e}")
                skipped += 1
                continue

            # Extract primary label
            raw_labels = labels_data.get("labels", [])
            if not raw_labels:
                skipped += 1
                continue
            primary_raw = raw_labels[0]
            primary = BEN_PRIMARY_LABELS.get(primary_raw, primary_raw)

            # Find RGB image (use B04, B03, B02 = red, green, blue)
            image_filename = f"{patch_dir.name}.png"
            target_image = output_dir / "images" / image_filename

            # Check for existing RGB image
            rgb_path = patch_dir / f"{patch_dir.name}_RGB.tif"
            if not rgb_path.exists():
                # Look for any RGB or .tif files
                tif_files = list(patch_dir.glob("*.tif"))
                if not tif_files:
                    skipped += 1
                    continue
                rgb_path = tif_files[0]

            # Convert to PNG (smaller, PIL-readable)
            if not target_image.exists():
                try:
                    convert_tif_to_png(rgb_path, target_image, patch_dir)
                except Exception as e:
                    logger.debug(f"Skip {patch_dir.name}: conversion failed: {e}")
                    skipped += 1
                    continue

            if not target_image.exists():
                skipped += 1
                continue

            # 1. Caption
            caption = generate_caption(primary)
            cap_f.write(json.dumps({"image_id": image_filename, "caption": caption}) + "\n")
            n_caption += 1

            # 2. VQA pairs (3-4 per image)
            vqa_pairs = generate_vqa_pairs(primary)
            for qa in vqa_pairs:
                qa["image_id"] = image_filename
                vqa_f.write(json.dumps(qa) + "\n")
                n_vqa += 1

            # 3. Referring expression
            ref = generate_referring(primary)
            ref_f.write(json.dumps(ref) + "\n")
            n_ref += 1

    logger.info(f"Done. Captions: {n_caption}, VQA: {n_vqa}, Referring: {n_ref}, Skipped: {skipped}")
    return n_caption


def convert_tif_to_png(tif_path: Path, png_path: Path, patch_dir: Path) -> None:
    """Convert a BigEarthNet GeoTIFF to a PIL-readable PNG."""
    try:
        from PIL import Image
        img = Image.open(tif_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(png_path, "PNG")
    except Exception:
        # Fallback: use rasterio for multi-band
        try:
            import rasterio
            import numpy as np
            with rasterio.open(tif_path) as src:
                bands = src.count
                if bands == 1:
                    arr = src.read(1)
                elif bands >= 3:
                    # Try B04, B03, B02 (red, green, blue) for Sentinel-2
                    if bands >= 4:
                        r, g, b = src.read(4), src.read(3), src.read(2)
                    else:
                        r, g, b = src.read(1), src.read(2), src.read(3)
                    def _norm(x):
                        x = x.astype(np.float32)
                        mn, mx = np.percentile(x, [2, 98])
                        if mx - mn < 1e-6:
                            mx = mn + 1.0
                        x = np.clip((x - mn) / (mx - mn) * 255, 0, 255)
                        return x.astype(np.uint8)
                    arr = np.stack([_norm(r), _norm(g), _norm(b)], axis=-1)
                else:
                    arr = src.read(1)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] > 3:
                arr = arr[..., :3]
            from PIL import Image
            Image.fromarray(arr).save(png_path, "PNG")
        except ImportError:
            raise RuntimeError("PIL and rasterio both unavailable")


# ---------------------------------------------------------------------------
# Caption / VQA / Referring generation
# ---------------------------------------------------------------------------
def generate_caption(label: str) -> str:
    """Generate a natural-language caption for a given land cover class."""
    templates = [
        f"Satellite image showing a {label} area.",
        f"Aerial view of {label} terrain with characteristic features.",
        f"Remote sensing scene depicting {label} landscape.",
        f"Overview of a {label} region captured by Sentinel-2 satellite.",
        f"Sentinel-2 image of {label} land cover type.",
    ]
    return random.choice(templates)


def generate_vqa_pairs(label: str) -> List[Dict]:
    """Generate VQA pairs for a given land cover class."""
    pairs = []
    for q, a in VQA_TEMPLATES:
        if "{label}" in a:
            a = a.format(label=label)
        if "{answer}" in a:
            is_match = label in ("urban",)
            a = a.format(answer="Yes" if is_match else "No")
        pairs.append({"question": q, "answer": a})
    return pairs


def generate_referring(label: str) -> Dict:
    """Generate a referring expression with a synthetic centered bbox."""
    template = random.choice(REFERRING_TEMPLATES)
    expression = template.format(label=label)
    # Synthetic bbox: center 60% of image
    cx = random.uniform(0.3, 0.5)
    cy = random.uniform(0.3, 0.5)
    half_w = random.uniform(0.2, 0.3)
    half_h = random.uniform(0.2, 0.3)
    bbox = [
        max(0, cx - half_w),
        max(0, cy - half_h),
        min(1, cx + half_w),
        min(1, cy + half_h),
    ]
    return {
        "expression": expression,
        "bbox": bbox,
    }


# ---------------------------------------------------------------------------
# Microsoft Planetary Computer ingestion
# ---------------------------------------------------------------------------
def process_planetary_computer(
    output_dir: Path,
    bbox: Tuple[float, float, float, float],
    date_range: Tuple[str, str],
    max_samples: int = 5000,
    collections: Tuple[str, ...] = ("sentinel-2-l2a",),
    cloud_cover_max: int = 15,
) -> int:
    """Query Microsoft Planetary Computer and download chips for LoRA training.

    Args:
        output_dir: Where to save the dataset
        bbox: (min_lon, min_lat, max_lon, max_lat) — e.g. Delhi: (77.0, 28.5, 77.5, 29.0)
        date_range: (start_date, end_date) ISO format
        max_samples: Max number of scenes to download
        collections: STAC collections to query
        cloud_cover_max: Max cloud cover percentage

    Returns:
        Number of scenes processed
    """
    try:
        import planetary_computer
        import pystac_client
    except ImportError:
        logger.error(
            "Planetary Computer support requires: pip install planetary-computer pystac-client"
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    logger.info(
        f"Querying Planetary Computer: bbox={bbox}, dates={date_range}, "
        f"collections={collections}, max_cloud={cloud_cover_max}%"
    )

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1/"
    )

    search = catalog.search(
        collections=list(collections),
        bbox=list(bbox),
        datetime=f"{date_range[0]}/{date_range[1]}",
        query={"eo:cloud_cover": {"lt": cloud_cover_max}} if "sentinel-2" in collections[0] else None,
        limit=max_samples,
    )
    items = list(search.items())
    logger.info(f"Found {len(items)} items")

    captions_path = output_dir / "captions.jsonl"
    vqa_path = output_dir / "vqa.jsonl"
    referring_path = output_dir / "referring.jsonl"

    n_processed = 0

    with open(captions_path, "w") as cap_f, \
         open(vqa_path, "w") as vqa_f, \
         open(referring_path, "w") as ref_f:

        for i, item in enumerate(items):
            if n_processed >= max_samples:
                break

            # Get RGB thumbnail or visual asset
            asset_key = "visual" if "visual" in item.assets else "thumbnail"
            if asset_key not in item.assets:
                # Try common RGB keys
                for key in ("visual", "thumbnail", "rgb", "B04"):
                    if key in item.assets:
                        asset_key = key
                        break
                else:
                    logger.debug(f"Skip {item.id}: no visual asset")
                    continue

            asset = item.assets[asset_key]
            href = asset.href

            # Sign URL for free access
            try:
                signed_href = planetary_computer.sign(href)
            except Exception as e:
                logger.debug(f"Skip {item.id}: sign failed: {e}")
                continue

            # Download
            image_filename = f"pc_{item.id}.tif"
            target_image = output_dir / "images" / image_filename
            if not target_image.exists():
                try:
                    download_asset(signed_href, target_image)
                except Exception as e:
                    logger.debug(f"Skip {item.id}: download failed: {e}")
                    continue

            if not target_image.exists():
                continue

            # Derive land cover hint from collection
            coll = item.collection_id
            if "sentinel-2" in coll:
                lc_hint = "mixed"  # unknown without classification
            elif "sentinel-1" in coll:
                lc_hint = "sar"
            else:
                lc_hint = "natural"

            # Generate JSONL entries
            cap_f.write(json.dumps({
                "image_id": image_filename,
                "caption": generate_caption(lc_hint),
            }) + "\n")
            for qa in generate_vqa_pairs(lc_hint):
                qa["image_id"] = image_filename
                vqa_f.write(json.dumps(qa) + "\n")
            ref = generate_referring(lc_hint)
            ref["image_id"] = image_filename
            ref_f.write(json.dumps(ref) + "\n")

            n_processed += 1
            if n_processed % 10 == 0:
                logger.info(f"Processed {n_processed}/{len(items)}")

    logger.info(f"Done. Processed {n_processed} scenes")
    return n_processed


def download_asset(url: str, target: Path) -> None:
    """Download an asset from URL to target path."""
    import urllib.request
    import shutil
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        with open(target, "wb") as out:
            shutil.copyfileobj(response, out)


# ---------------------------------------------------------------------------
# Generic folder ingestion
# ---------------------------------------------------------------------------
def process_generic_folder(
    images_dir: Path,
    labels_csv: Optional[Path],
    output_dir: Path,
    max_samples: int = 5000,
) -> int:
    """Process a folder of images with optional labels CSV.

    CSV format: image_filename,label
    """
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    # Read labels
    labels = {}
    if labels_csv and labels_csv.exists():
        with open(labels_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["image_filename"]] = row["label"]

    # Find image files
    image_files = []
    for ext in ("*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"):
        image_files.extend(images_dir.glob(ext))
    image_files = image_files[:max_samples]
    logger.info(f"Found {len(image_files)} image files")

    captions_path = output_dir / "captions.jsonl"
    vqa_path = output_dir / "vqa.jsonl"
    referring_path = output_dir / "referring.jsonl"

    n_processed = 0
    with open(captions_path, "w") as cap_f, \
         open(vqa_path, "w") as vqa_f, \
         open(referring_path, "w") as ref_f:

        for img_path in image_files:
            label = labels.get(img_path.name, "unknown")

            # Copy or symlink to output images
            target = output_dir / "images" / img_path.name
            if not target.exists():
                if img_path.resolve() != target.resolve():
                    try:
                        target.symlink_to(img_path.resolve())
                    except OSError:
                        import shutil
                        shutil.copy2(img_path, target)

            # Generate JSONL entries
            cap_f.write(json.dumps({
                "image_id": img_path.name,
                "caption": generate_caption(label),
            }) + "\n")
            for qa in generate_vqa_pairs(label):
                qa["image_id"] = img_path.name
                vqa_f.write(json.dumps(qa) + "\n")
            ref = generate_referring(label)
            ref["image_id"] = img_path.name
            ref_f.write(json.dumps(ref) + "\n")
            n_processed += 1

    logger.info(f"Done. Processed {n_processed} images")
    return n_processed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare real satellite data for M3 LoRA fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # bigearthnet
    ben = subparsers.add_parser("bigearthnet", help="Ingest BigEarthNet patches")
    ben.add_argument("--source", required=True, help="BigEarthNet root directory")
    ben.add_argument("--output", default="./data/bigearthnet_lora")
    ben.add_argument("--max-samples", type=int, default=5000)

    # planetary_computer
    pc = subparsers.add_parser("planetary", help="Query Planetary Computer")
    pc.add_argument("--bbox", nargs=4, type=float, required=True,
                   metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                   help="Bounding box (e.g. Delhi: 77.0 28.5 77.5 29.0)")
    pc.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    pc.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    pc.add_argument("--output", default="./data/bigearthnet_lora")
    pc.add_argument("--max-samples", type=int, default=200)
    pc.add_argument("--collections", nargs="+", default=["sentinel-2-l2a"])
    pc.add_argument("--max-cloud", type=int, default=15)

    # generic
    gen = subparsers.add_parser("generic", help="Process generic image folder")
    gen.add_argument("--images-dir", required=True)
    gen.add_argument("--labels-csv", help="CSV with image_filename,label columns")
    gen.add_argument("--output", default="./data/bigearthnet_lora")
    gen.add_argument("--max-samples", type=int, default=5000)

    args = parser.parse_args()

    if args.command == "bigearthnet":
        process_bigearthnet(
            Path(args.source),
            Path(args.output),
            args.max_samples,
        )
    elif args.command == "planetary":
        process_planetary_computer(
            Path(args.output),
            tuple(args.bbox),
            (args.start_date, args.end_date),
            args.max_samples,
            tuple(args.collections),
            args.max_cloud,
        )
    elif args.command == "generic":
        process_generic_folder(
            Path(args.images_dir),
            Path(args.labels_csv) if args.labels_csv else None,
            Path(args.output),
            args.max_samples,
        )


if __name__ == "__main__":
    main()