"""
M3 SAR Data Acquisition — Get paired optical+SAR training data.

For fine-tuning GeoChat to understand SAR, we need:
  1. Sentinel-1 SAR (VV+VH) images with labels
  2. OPTIONAL but recommended: paired Sentinel-2 optical for the same AOI
  3. Captions/VQA/referring expressions for both modalities

BEST FREE SOURCES:
  - Microsoft Planetary Computer: Sentinel-1 GRD (VV+VH), Sentinel-2 L2A
  - Alaska Satellite Facility (ASF): Sentinel-1 archive
  - HuggingFace datasets: SARFish, SEN12MS, BigEarthNet-MM-SAR

STRATEGY:
  1. Query Planetary Computer for Sentinel-1 GRD over Indian cities
  2. Pair with Sentinel-2 L2A from same date (if available)
  3. Apply SAR preprocessing (log transform, speckle filter, pseudo-RGB)
  4. Generate training JSONL with SAR-specific prompts
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Indian cities (same as acquire_data.py)
INDIAN_CITIES = {
    "delhi": (77.05, 28.40, 77.35, 28.80),
    "mumbai": (72.75, 18.85, 73.05, 19.30),
    "bangalore": (77.45, 12.85, 77.75, 13.15),
    "kerala_flood": (76.20, 9.95, 76.50, 10.50),  # for SAR change detection
    "assam_flood": (90.50, 25.50, 95.00, 27.50),
    "rajasthan": (72.50, 26.50, 74.00, 28.00),
}

# SAR-specific question templates
SAR_VQA_TEMPLATES = [
    # Backscatter questions
    ("What is the dominant backscatter pattern in this image?",
     "{answer}"),
    ("Are there bright or dark areas in this SAR image?",
     "{answer}"),
    ("What does the bright area in this image likely represent?",
     "Bright areas in SAR indicate strong backscatter, typically from {label}."),
    ("What does the dark area in this image likely represent?",
     "Dark areas in SAR indicate low backscatter, typically from {label}."),

    # Land cover questions
    ("What is the land cover in this image?",
     "Based on radar backscatter, the land cover is {label}."),
    ("Is this area urban?",
     "{urban_answer}"),
    ("Is there water visible in this image?",
     "Water appears very dark in SAR due to specular reflection. {water_answer}"),
    ("Is this a vegetated area?",
     "Vegetation shows medium backscatter with characteristic texture. {veg_answer}"),
    ("Are there any linear features visible?",
     "Linear features in SAR can indicate roads, railways, or field boundaries. {linear_answer}"),

    # SAR-specific
    ("What polarization is this image?",
     "This is Sentinel-1 C-band SAR with VV and VH polarizations shown as RGB."),
    ("Is this image from an active or passive sensor?",
     "SAR is an active sensor, meaning it transmits its own radar signal and measures the backscatter."),
]

# SAR-specific caption templates
SAR_CAPTION_TEMPLATES = [
    "Sentinel-1 SAR image showing {label} with characteristic backscatter patterns.",
    "C-band radar image of {label}. Bright areas indicate strong backscatter, dark areas smooth surfaces.",
    "SAR acquisition over {label}. The {pattern} pattern suggests {interpretation}.",
    "Synthetic Aperture Radar view of {label} with VV and VH polarizations visible.",
]

# Referring expressions for SAR
SAR_REFERRING_TEMPLATES = [
    "the bright area (strong backscatter)",
    "the dark area (low backscatter)",
    "the {label} region",
    "the urban area (high backscatter)",
    "the water body (very low backscatter)",
    "the vegetated area (medium backscatter with texture)",
    "the linear structure (likely a road or boundary)",
]


def acquire_sar_data(
    output_dir: Path,
    bbox: Tuple[float, float, float, float],
    date_range: Tuple[str, str],
    max_samples: int = 100,
    include_optical: bool = True,
    max_cloud_cover: int = 25,
) -> int:
    """
    Acquire paired Sentinel-1 SAR and Sentinel-2 optical from Planetary Computer.
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

    logger.info(f"Querying Planetary Computer for SAR data over {bbox}")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1/"
    )

    # Query Sentinel-1 GRD
    sar_search = catalog.search(
        collections=["sentinel-1-grd"],
        bbox=list(bbox),
        datetime=f"{date_range[0]}/{date_range[1]}",
        query={
            "sar:frequency_band": {"eq": "C"},
            "sar:instrument_mode": {"eq": "IW"},  # Interferometric Wide (most common)
            "sar:product_type": {"eq": "GRD"},  # Ground Range Detected
            "polarization": {"in": ["VV", "VV+VH"]},  # At least VV
        },
        limit=max_samples,
    )
    sar_items = list(sar_search.items())
    logger.info(f"Found {len(sar_items)} Sentinel-1 GRD items")

    # Optionally query optical for pairing
    optical_items_by_date = {}
    if include_optical:
        logger.info("Querying Sentinel-2 for pairing...")
        opt_search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=list(bbox),
            datetime=f"{date_range[0]}/{date_range[1]}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            limit=max_samples * 2,
        )
        for item in opt_search.items():
            # Index by date (YYYY-MM-DD)
            date_str = item.datetime.strftime("%Y-%m-%d")
            optical_items_by_date[date_str] = item
        logger.info(f"Found {len(optical_items_by_date)} unique optical dates")

    n_processed = 0
    captions_path = output_dir / "captions.jsonl"
    vqa_path = output_dir / "vqa.jsonl"
    referring_path = output_dir / "referring.jsonl"
    optical_captions_path = output_dir / "optical_captions.jsonl"
    fusion_captions_path = output_dir / "fusion_captions.jsonl"

    with open(captions_path, "w") as sar_cap_f, \
         open(vqa_path, "w") as sar_vqa_f, \
         open(referring_path, "w") as sar_ref_f, \
         open(optical_captions_path, "w") as opt_cap_f, \
         open(fusion_captions_path, "w") as fusion_cap_f:

        for i, item in enumerate(sar_items):
            if n_processed >= max_samples:
                break

            try:
                # Download SAR (VV+VH)
                if "vv" in item.assets and "vh" in item.assets:
                    vv_href = planetary_computer.sign(item.assets["vv"].href)
                    vh_href = planetary_computer.sign(item.assets["vh"].href)

                    # Download both bands
                    sar_filename = f"pc_sar_{item.id}.tif"
                    sar_path = images_dir / sar_filename
                    if not sar_path.exists():
                        # Use a temporary directory to download both bands
                        tmp_vv = images_dir / f"_tmp_vv_{item.id}.tif"
                        tmp_vh = images_dir / f"_tmp_vh_{item.id}.tif"
                        try:
                            _download_url(vv_href, tmp_vv)
                            _download_url(vh_href, tmp_vh)
                            # Combine VV+VH into single multi-band TIFF
                            _combine_sar_bands(tmp_vv, tmp_vh, sar_path)
                        finally:
                            for p in (tmp_vv, tmp_vh):
                                if p.exists():
                                    p.unlink()

                    if not sar_path.exists():
                        continue

                    # Generate SAR training samples
                    label = _infer_land_cover(bbox)
                    sar_cap_f.write(json.dumps({
                        "image_id": sar_filename,
                        "caption": random.choice(SAR_CAPTION_TEMPLATES).format(
                            label=label, pattern="homogeneous", interpretation="uniform surface"
                        ),
                    }) + "\n")

                    for q, a in SAR_VQA_TEMPLATES:
                        answer = a.format(
                            answer=f"Predominantly {label}.",
                            label=label,
                            urban_answer="Yes" if label == "urban" else "No",
                            water_answer="Yes" if label == "water" else "No",
                            veg_answer="Yes" if label == "forest" else "No",
                            linear_answer="Yes, visible in the image" if label == "urban" else "Not clearly visible",
                        )
                        sar_vqa_f.write(json.dumps({
                            "image_id": sar_filename,
                            "question": q,
                            "answer": answer,
                        }) + "\n")

                    sar_ref_f.write(json.dumps({
                        "image_id": sar_filename,
                        "expression": random.choice(SAR_REFERRING_TEMPLATES).format(label=label),
                        "bbox": [0.2, 0.2, 0.8, 0.8],
                    }) + "\n")

                    # Try to find paired optical
                    if include_optical:
                        date_str = item.datetime.strftime("%Y-%m-%d")
                        # Look for optical within ±3 days
                        paired_opt = None
                        for offset in range(-3, 4):
                            try:
                                check_date = (item.datetime.replace(day=item.datetime.day + offset)).strftime("%Y-%m-%d")
                                if check_date in optical_items_by_date:
                                    paired_opt = optical_items_by_date[check_date]
                                    break
                            except Exception:
                                continue

                        if paired_opt:
                            opt_filename = f"pc_opt_{item.id}.tif"
                            opt_path = images_dir / opt_filename
                            if not opt_path.exists() and "visual" in paired_opt.assets:
                                _download_url(
                                    planetary_computer.sign(paired_opt.assets["visual"].href),
                                    opt_path
                                )

                            if opt_path.exists():
                                # Optical-only caption
                                opt_cap_f.write(json.dumps({
                                    "image_id": opt_filename,
                                    "caption": f"Sentinel-2 optical image showing {label} area.",
                                }) + "\n")

                                # Fusion caption (treats SAR and optical as same scene)
                                fusion_cap_f.write(json.dumps({
                                    "sar_image_id": sar_filename,
                                    "optical_image_id": opt_filename,
                                    "scene_id": item.id,
                                    "caption": _make_fusion_caption(label),
                                }) + "\n")

                    n_processed += 1
                    if n_processed % 10 == 0:
                        logger.info(f"Processed {n_processed}/{len(sar_items)}")

            except Exception as e:
                logger.debug(f"Skip {item.id}: {e}")
                continue

    logger.info(f"Done. Processed {n_processed} SAR scenes")
    return n_processed


def acquire_sar_from_hf_dataset(
    output_dir: Path,
    dataset_name: str = "blanchon/SEN12MS",
    max_samples: int = 1000,
) -> int:
    """
    Acquire SAR data from HuggingFace datasets.

    Available SAR datasets on HF:
      - blanchon/SEN12MS: Sentinel-1 + Sentinel-2 paired, 180K samples
      - Sentinel-1-SAR: Various smaller datasets
      - zclark/EuroSAT-SAR: SAR-only land cover
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install: pip install datasets")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    logger.info(f"Loading HuggingFace dataset: {dataset_name}")

    try:
        ds = load_dataset(dataset_name, split="train", streaming=True)
    except Exception as e:
        logger.error(f"Failed to load {dataset_name}: {e}")
        return 0

    n_processed = 0
    captions_path = output_dir / "captions.jsonl"
    vqa_path = output_dir / "vqa.jsonl"

    with open(captions_path, "w") as cap_f, open(vqa_path, "w") as vqa_f:
        for i, sample in enumerate(ds):
            if n_processed >= max_samples:
                break
            try:
                # SEN12MS has s1 (SAR) and s2 (optical) keys
                if "s1" in sample:
                    sar_img = sample["s1"]
                    sar_filename = f"hf_sar_{i:06d}.png"
                    sar_path = images_dir / sar_filename
                    sar_img.save(sar_path)

                    label = sample.get("label", sample.get("labels", "unknown"))
                    cap_f.write(json.dumps({
                        "image_id": sar_filename,
                        "caption": f"SAR image of {label} area.",
                    }) + "\n")

                    sar_vqa_f.write(json.dumps({
                        "image_id": sar_filename,
                        "question": "What is the land cover?",
                        "answer": f"The land cover is {label}.",
                    }) + "\n")

                    n_processed += 1
                    if n_processed % 50 == 0:
                        logger.info(f"Processed {n_processed}")
            except Exception as e:
                logger.debug(f"Skip sample {i}: {e}")
                continue

    return n_processed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_url(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url, headers={"User-Agent": "SatQuery-AI/1.0 (research)"}
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        with open(target, "wb") as f:
            f.write(response.read())


def _combine_sar_bands(vv_path: Path, vh_path: Path, output_path: Path) -> None:
    """Combine VV and VH into a 2-band GeoTIFF using rasterio."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
        import numpy as np

        with rasterio.open(vv_path) as vv_src:
            vv = vv_src.read(1)
            profile = vv_src.profile.copy()

        with rasterio.open(vh_path) as vh_src:
            vh = vh_src.read(1)

        # Stack as 2-band
        profile.update(count=2, dtype="float32")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(vv.astype(np.float32), 1)
            dst.write(vh.astype(np.float32), 2)

    except ImportError:
        # Fallback: just copy VV
        import shutil
        shutil.copy2(vv_path, output_path)


def _infer_land_cover(bbox: Tuple[float, float, float, float]) -> str:
    """Infer land cover from bbox (rough heuristic)."""
    center_lat = (bbox[1] + bbox[3]) / 2
    center_lon = (bbox[0] + bbox[2]) / 2
    # Match against known bboxes
    for city, city_bbox in INDIAN_CITIES.items():
        if (city_bbox[0] <= center_lon <= city_bbox[2] and
            city_bbox[1] <= center_lat <= city_bbox[3]):
            if "delhi" in city or "mumbai" in city or "bangalore" in city:
                return "urban"
            elif "flood" in city:
                return "water"
            elif "rajasthan" in city:
                return "barren"
    return "mixed"


def _make_fusion_caption(label: str) -> str:
    """Generate a fusion caption that uses both optical and SAR."""
    templates = [
        f"Combined Sentinel-1 SAR and Sentinel-2 optical view of {label}. "
        f"The optical image shows the visible appearance while the SAR image "
        f"reveals surface roughness and structural information.",

        f"Multi-modal analysis of {label} using both radar backscatter and "
        f"optical reflectance. The combination provides complementary "
        f"information about the scene.",

        f"SAR-optical fusion over {label}. Features that appear ambiguous "
        f"in one modality become clear when both are analyzed together.",
    ]
    return random.choice(templates)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Acquire SAR data for GeoChat fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # planetary
    pc = subparsers.add_parser("planetary", help="Query Planetary Computer for Sentinel-1")
    pc.add_argument("--bbox", nargs=4, type=float, required=True,
                   metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    pc.add_argument("--start-date", required=True)
    pc.add_argument("--end-date", required=True)
    pc.add_argument("--output", default="./data/sar_lora")
    pc.add_argument("--max-samples", type=int, default=200)
    pc.add_argument("--no-optical-pairing", action="store_true")

    # cities
    cities_p = subparsers.add_parser("cities", help="SAR for Indian cities")
    cities_p.add_argument("--cities", nargs="+",
                        default=["delhi", "mumbai", "bangalore", "kerala_flood", "rajasthan"])
    cities_p.add_argument("--output", default="./data/sar_lora")
    cities_p.add_argument("--max-per-city", type=int, default=100)
    cities_p.add_argument("--start-date", default="2024-01-01")
    cities_p.add_argument("--end-date", default="2024-12-31")

    # huggingface
    hf = subparsers.add_parser("huggingface", help="Pull SAR data from HF dataset")
    hf.add_argument("--dataset", default="blanchon/SEN12MS")
    hf.add_argument("--output", default="./data/sar_lora")
    hf.add_argument("--max-samples", type=int, default=1000)

    args = parser.parse_args()

    if args.command == "planetary":
        acquire_sar_data(
            Path(args.output), tuple(args.bbox),
            (args.start_date, args.end_date),
            max_samples=args.max_samples,
            include_optical=not args.no_optical_pairing,
        )
    elif args.command == "cities":
        for city in args.cities:
            if city not in INDIAN_CITIES:
                logger.warning(f"Unknown city: {city}")
                continue
            bbox = INDIAN_CITIES[city]
            city_dir = Path(args.output) / city
            logger.info(f"=== {city.upper()} ===")
            acquire_sar_data(
                city_dir, bbox,
                (args.start_date, args.end_date),
                max_samples=args.max_per_city,
            )
    elif args.command == "huggingface":
        acquire_sar_from_hf_dataset(
            Path(args.output), args.dataset, args.max_samples
        )


if __name__ == "__main__":
    main()