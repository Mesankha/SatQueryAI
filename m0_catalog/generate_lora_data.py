"""
M0 LoRA Training Dataset Generation
Extracts LoRA training data from BigEarthNet during ingestion.
Per plan §5.4: prepares data for m3_vlm/train_lora.py
"""
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.schemas import SceneMetadata
from .catalog import Catalog
from .config import get_m0_config


def generate_lora_training_data(
    catalog: Catalog,
    output_dir: str,
    max_samples: int = 5000,
    dataset_origin: str = "bigearthnet_txt",
) -> Dict[str, int]:
    """
    Generate LoRA training dataset from BigEarthNet scenes in catalog.
    Creates three supervision types: captions, VQA, referring expressions.
    """
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Get BigEarthNet scenes
    scenes = catalog.query_scenes(dataset_origin=dataset_origin)
    print(f"Found {len(scenes)} {dataset_origin} scenes for LoRA dataset")

    # Filter scenes with valid file_path and object
    valid_scenes = [s for s in scenes if s.file_path and s.object and Path(s.file_path).exists()]
    print(f"{len(valid_scenes)} scenes have valid images and labels")

    # Shuffle and limit
    random.shuffle(valid_scenes)
    valid_scenes = valid_scenes[:max_samples]

    # Output files
    captions_file = output_path / "captions.jsonl"
    vqa_file = output_path / "vqa.jsonl"
    referring_file = output_path / "referring.jsonl"

    caption_count = 0
    vqa_count = 0
    referring_count = 0

    with open(captions_file, "w") as cap_f, \
         open(vqa_file, "w") as vqa_f, \
         open(referring_file, "w") as ref_f:

        for scene in valid_scenes:
            # Copy image to LoRA dataset
            target_image = images_dir / f"{scene.scene_id}.tif"
            if not target_image.exists():
                shutil.copy2(scene.file_path, target_image)

            image_id = f"{scene.scene_id}.tif"

            # 1. Caption supervision (from ground_truth_text / object)
            caption = _generate_caption(scene)
            cap_f.write(json.dumps({"image_id": image_id, "caption": caption}) + "\n")
            caption_count += 1

            # 2. VQA supervision (template-based from object)
            vqa_pairs = _generate_vqa_pairs(scene)
            for qa in vqa_pairs:
                qa["image_id"] = image_id
                vqa_f.write(json.dumps(qa) + "\n")
                vqa_count += 1

            # 3. Referring expression supervision (synthetic bbox)
            referring_expr = _generate_referring_expression(scene)
            ref_f.write(json.dumps(referring_expr) + "\n")
            referring_count += 1

    print(f"LoRA dataset generated:")
    print(f"  Images: {len(valid_scenes)}")
    print(f"  Captions: {caption_count}")
    print(f"  VQA pairs: {vqa_count}")
    print(f"  Referring expressions: {referring_count}")

    return {
        "images": len(valid_scenes),
        "captions": caption_count,
        "vqa": vqa_count,
        "referring": referring_count,
    }


def _generate_caption(scene: SceneMetadata) -> str:
    """Generate caption from scene metadata."""
    obj = scene.object or "land cover"
    sensor = scene.sensor
    modality = scene.modality

    templates = [
        f"This {modality} image from {sensor} shows {obj} areas.",
        f"A {sensor} {modality} patch containing {obj}.",
        f"Remote sensing image of {obj} captured by {sensor}.",
        f"The scene depicts {obj} as observed by {sensor}.",
    ]
    return random.choice(templates)


def _generate_vqa_pairs(scene: SceneMetadata) -> List[Dict[str, str]]:
    """Generate VQA question-answer pairs from scene metadata."""
    obj = scene.object or "land cover"
    sensor = scene.sensor

    pairs = []

    # Template questions
    templates = [
        ("What is the land cover in this image?", f"The land cover is {obj}."),
        ("What type of terrain is shown?", f"This image shows {obj} terrain."),
        (f"Is there {obj} in this image?", f"Yes, {obj} is present in this image."),
        ("Which satellite captured this image?", f"This image was captured by {sensor}."),
        ("What is the dominant land cover class?", f"The dominant land cover is {obj}."),
    ]

    for question, answer in templates:
        pairs.append({"question": question, "answer": answer})

    return pairs


def _generate_referring_expression(scene: SceneMetadata) -> Dict[str, Any]:
    """Generate referring expression with synthetic bbox."""
    obj = scene.object or "area"

    # Synthetic bbox (center crop 0.2-0.8 normalized)
    # In practice, would use actual object detection
    bbox = [0.25, 0.25, 0.75, 0.75]

    expressions = [
        f"the {obj} area",
        f"the {obj} region",
        f"{obj} in the center",
        f"the main {obj} patch",
    ]

    return {
        "image_id": f"{scene.scene_id}.tif",
        "expression": random.choice(expressions),
        "bbox": bbox,
    }


def create_dummy_bigearthnet_lora(data_dir: str, num_samples: int = 100):
    """Create dummy BigEarthNet LoRA data for testing."""
    os.makedirs(os.path.join(data_dir, "images"), exist_ok=True)

    classes = ["urban", "agriculture", "forest", "water", "barren"]

    captions = []
    vqa = []
    referring = []

    for i in range(num_samples):
        cls = classes[i % len(classes)]
        image_id = f"dummy_{i:04d}.tif"

        captions.append({"image_id": image_id, "caption": f"Remote sensing image showing {cls} area."})
        vqa.append({"image_id": image_id, "question": "What is the land cover?", "answer": f"The land cover is {cls}."})
        referring.append({"image_id": image_id, "expression": f"the {cls} area", "bbox": [0.2, 0.2, 0.8, 0.8]})

    for fname, data in [("captions.jsonl", captions), ("vqa.jsonl", vqa), ("referring.jsonl", referring)]:
        with open(os.path.join(data_dir, fname), "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    print(f"Created dummy LoRA data in {data_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate LoRA training data from M0 catalog")
    parser.add_argument("--db", default="./data/catalog.db", help="Catalog database path")
    parser.add_argument("--output-dir", default="./data/bigearthnet_lora", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max samples")
    parser.add_argument("--origin", default="bigearthnet_txt", help="Dataset origin")
    parser.add_argument("--create-dummy", action="store_true", help="Create dummy data for testing")

    args = parser.parse_args()

    if args.create_dummy:
        create_dummy_bigearthnet_lora(args.output_dir, args.max_samples)
    else:
        catalog = Catalog(args.db, read_only=True)
        generate_lora_training_data(catalog, args.output_dir, args.max_samples, args.origin)
        catalog.close()