"""
Test a trained GeoChat LoRA adapter with real images.
This is a simple verification script you can use after training completes.

USAGE:
  python -m m3_vlm.test_inference --image ./path/to/test_image.png --question "What is this?"
  python -m m3_vlm.test_inference --dir ./data/test_images/
  python -m m3_vlm.test_inference --image ./sar_image.png --question "What is the land cover?"

The script auto-detects if the image is SAR or optical based on filename,
applies the right preprocessing, and uses the right prompt.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Prevent pytest from collecting this as a test file
__test__ = False

# Setup path
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger(__name__)


logger = setup_logging()


async def test_single_image(
    image_path: str,
    question: Optional[str] = None,
    adapter_path: Optional[str] = None,
    model_path: str = "mbzuai-oryx/GeoChat",
    verbose: bool = True,
) -> Dict:
    """
    Test inference on a single image.

    Auto-detects modality (SAR vs optical) from filename.
    Returns the result dict.
    """
    try:
        from m3_vlm.vlm_service import initialize_service
        from m3_vlm.api import run_vqa, run_caption, run_grounding
        from m3_vlm.sar_prompts import detect_modality  # type: ignore
    except ImportError as e:
        logger.error(f"Required module not available: {e}")
        return {"error": True, "error_type": "import_error", "message": str(e)}

    image_path = str(Path(image_path).resolve())
    if not Path(image_path).exists():
        return {"error": True, "error_type": "invalid_input", "message": f"Image not found: {image_path}"}

    modality = detect_modality(Path(image_path).name)
    if verbose:
        print(f"\n{'='*60}")
        print(f"Image: {image_path}")
        print(f"Modality: {modality.upper()}")
        if adapter_path:
            print(f"Adapter: {adapter_path}")
        else:
            print("Adapter: (none - using base GeoChat)")
        print(f"{'='*60}")

    # Initialize service
    config = {
        "model_path": model_path,
        "load_in_4bit": True,
        "lora_adapter_path": adapter_path,
    }
    ok = await initialize_service(config)
    if not ok:
        return {
            "error": True,
            "error_type": "model_unavailable",
            "message": "Failed to initialize GeoChat. Check torch/CUDA installation.",
        }

    results = {}
    start = time.time()

    # 1. Caption
    if verbose:
        print("\n[1/3] Generating caption...")
    caption_result = await run_caption(image_path)
    results["caption"] = caption_result
    if verbose:
        if caption_result.get("error"):
            print(f"  ERROR: {caption_result.get('message', 'unknown')}")
        else:
            warning = caption_result.get("warning", "")
            warning_str = f" [WARN: {warning}]" if warning else ""
            print(f"  Caption: {caption_result.get('caption', '')}{warning_str}")

    # 2. VQA
    if question:
        if verbose:
            print(f"\n[2/3] Answering question: '{question}'")
        vqa_result = await run_vqa(image_path, question)
        results["vqa"] = vqa_result
        if verbose:
            if vqa_result.get("error"):
                print(f"  ERROR: {vqa_result.get('message', 'unknown')}")
            else:
                warning = vqa_result.get("warning", "")
                warning_str = f" [WARN: {warning}]" if warning else ""
                print(f"  Answer: {vqa_result.get('answer', '')}{warning_str}")

    # 3. Grounding (use a default expression based on modality)
    if modality == "sar":
        expr = "the brightest area (strong backscatter)"
    else:
        expr = "the urban area"
    if verbose:
        print(f"\n[3/3] Grounding expression: '{expr}'")
    grounding_result = await run_grounding(image_path, expr)
    results["grounding"] = grounding_result
    if verbose:
        if grounding_result.get("error"):
            print(f"  ERROR: {grounding_result.get('message', 'unknown')}")
        else:
            print(f"  BBox: {grounding_result.get('bbox', [])}, "
                  f"Confidence: {grounding_result.get('confidence', 0):.2f}")

    elapsed = time.time() - start
    results["total_time_sec"] = elapsed
    results["modality"] = modality

    if verbose:
        print(f"\nTotal time: {elapsed:.1f}s")

    return results


async def test_directory(
    directory: str,
    adapter_path: Optional[str] = None,
    model_path: str = "mbzuai-oryx/GeoChat",
    max_images: int = 10,
):
    """Test inference on all images in a directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.error(f"Directory not found: {directory}")
        return

    # Find image files
    image_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        image_files.extend(dir_path.glob(ext))

    if not image_files:
        logger.error(f"No images found in {directory}")
        return

    image_files = image_files[:max_images]
    logger.info(f"Testing {len(image_files)} images from {directory}")

    all_results = []
    for img_path in image_files:
        result = await test_single_image(
            str(img_path), question="What is the land cover in this image?",
            adapter_path=adapter_path, model_path=model_path, verbose=True,
        )
        all_results.append({
            "image": str(img_path),
            "result": result,
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in all_results:
        img = Path(r["image"]).name
        modality = r["result"].get("modality", "?")
        caption = r["result"].get("caption", {})
        if caption.get("error"):
            status = "ERROR"
        else:
            status = caption.get("caption", "")[:50]
        print(f"  [{modality:8s}] {img}: {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Test a trained GeoChat LoRA adapter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single image")
    group.add_argument("--dir", help="Path to a directory of images")

    parser.add_argument("--question", default="What is the land cover in this image?",
                        help="Question to ask in VQA")
    parser.add_argument("--adapter", help="Path to LoRA adapter directory")
    parser.add_argument("--model", default="mbzuai-oryx/GeoChat",
                        help="Base model path")
    parser.add_argument("--max-images", type=int, default=10,
                        help="Max images to test in directory mode")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    if args.image:
        result = asyncio.run(test_single_image(
            args.image, question=args.question,
            adapter_path=args.adapter, model_path=args.model,
        ))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")
    else:
        asyncio.run(test_directory(
            args.dir, adapter_path=args.adapter,
            model_path=args.model, max_images=args.max_images,
        ))


if __name__ == "__main__":
    main()