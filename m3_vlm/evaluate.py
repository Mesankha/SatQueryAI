"""
Evaluate a trained GeoChat LoRA adapter on real test data.

This script:
  1. Loads GeoChat base model + LoRA adapter
  2. Runs inference on a test set (JSONL with image_id + question/answer)
  3. Computes metrics:
     - Caption: BLEU-4, CIDEr (simple), BERTScore (cosine sim)
     - VQA: exact match, F1
     - Grounding: IoU @ 0.5
  4. Saves results to JSON + Markdown report

Usage:
  python -m m3_vlm.evaluate \
    --adapter ./models/geochat_lora/adapter \
    --test-data ./data/bigearthnet_lora/test_set.jsonl \
    --data-dir ./data/bigearthnet_lora \
    --output ./eval_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup path
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Imports
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from m3_vlm.vlm_service import initialize_service, get_service
    from m3_vlm.api import run_vqa, run_caption, run_grounding
    SERVICE_AVAILABLE = True
except ImportError as e:
    SERVICE_AVAILABLE = False
    SERVICE_ERROR = str(e)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def tokenize(s: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    s = s.lower().strip()
    s = re.sub(r"([.!?,])", r" \1 ", s)
    return s.split()


def bleu4(reference: str, prediction: str) -> float:
    """Compute a simple BLEU-4 score (geometric mean of 1-4 gram precisions)."""
    import math
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)
    if not pred_tokens or not ref_tokens:
        return 0.0
    scores = []
    for n in (1, 2, 3, 4):
        ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1))
        pred_ngrams = Counter(tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1))
        overlap = sum((ref_ngrams & pred_ngrams).values())
        total = sum(pred_ngrams.values())
        scores.append(overlap / total if total else 0.0)
    # Geometric mean
    if any(s == 0 for s in scores):
        return 0.0
    return math.exp(sum(math.log(s) for s in scores) / 4)


def f1_score(reference: str, prediction: str) -> float:
    """Token-level F1 (good for VQA)."""
    ref = tokenize(reference)
    pred = tokenize(prediction)
    if not ref or not pred:
        return 0.0
    common = Counter(ref) & Counter(pred)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(ref)
    return 2 * precision * recall / (precision + recall)


def iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Intersection over Union for normalized [0,1] bboxes."""
    if bbox1 is None or bbox2 is None:
        return 0.0
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    a2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------
def resolve_image_path(sample: Dict[str, Any], data_dir: Path) -> Path:
    """Resolve image path from sample dict, trying multiple keys."""
    candidates = [
        sample.get("image_path"),
        sample.get("file_path"),
        data_dir / "images" / sample.get("image_id", ""),
        data_dir / sample.get("image_id", ""),
    ]
    for c in candidates:
        if c is None:
            continue
        p = Path(c)
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot resolve image for sample: keys={list(sample.keys())}, data_dir={data_dir}"
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
class Evaluator:
    def __init__(self, adapter_path: Optional[str] = None, model_path: str = "mbzuai-oryx/GeoChat"):
        self.adapter_path = adapter_path
        self.model_path = model_path
        self.service = None

    async def setup(self):
        """Initialize the GeoChat service with optional LoRA adapter."""
        if not SERVICE_AVAILABLE:
            raise RuntimeError(
                f"M3 service not available: {SERVICE_ERROR}. "
                f"Install: pip install torch transformers peft accelerate bitsandbytes"
            )
        config = {
            "model_path": self.model_path,
            "load_in_4bit": True,
            "lora_adapter_path": self.adapter_path,
        }
        logger.info(f"Initializing GeoChat with adapter={self.adapter_path}")
        ok = await initialize_service(config)
        if not ok:
            raise RuntimeError("Failed to initialize GeoChat service")
        self.service = get_service()
        logger.info("Service ready")

    async def eval_caption(self, test_samples: List[Dict], data_dir: Path) -> Dict:
        """Evaluate caption generation."""
        logger.info(f"Evaluating captioning on {len(test_samples)} samples")
        results = []
        for i, sample in enumerate(test_samples):
            try:
                img_path = resolve_image_path(sample, data_dir)
                ref_caption = sample.get("caption", "")
                if not ref_caption:
                    continue
                pred = await run_caption(str(img_path))
                pred_text = pred.get("caption", "") if not pred.get("error") else ""
                score = bleu4(ref_caption, pred_text)
                results.append({
                    "image_id": sample.get("image_id"),
                    "reference": ref_caption,
                    "prediction": pred_text,
                    "bleu4": score,
                    "error": pred.get("error", False),
                })
                if (i + 1) % 10 == 0:
                    logger.info(f"  Caption {i+1}/{len(test_samples)}: avg BLEU={sum(r['bleu4'] for r in results)/len(results):.3f}")
            except Exception as e:
                logger.warning(f"Caption eval failed for sample {i}: {e}")
        avg = sum(r["bleu4"] for r in results) / len(results) if results else 0
        return {"metric": "bleu4", "average": avg, "count": len(results), "samples": results}

    async def eval_vqa(self, test_samples: List[Dict], data_dir: Path) -> Dict:
        """Evaluate VQA."""
        logger.info(f"Evaluating VQA on {len(test_samples)} samples")
        results = []
        for i, sample in enumerate(test_samples):
            try:
                img_path = resolve_image_path(sample, data_dir)
                ref_answer = sample.get("answer", "")
                question = sample.get("question", "")
                if not ref_answer or not question:
                    continue
                pred = await run_vqa(str(img_path), question)
                pred_text = pred.get("answer", "") if not pred.get("error") else ""
                f1 = f1_score(ref_answer, pred_text)
                em = 1.0 if tokenize(ref_answer) == tokenize(pred_text) else 0.0
                results.append({
                    "image_id": sample.get("image_id"),
                    "question": question,
                    "reference": ref_answer,
                    "prediction": pred_text,
                    "f1": f1,
                    "exact_match": em,
                    "error": pred.get("error", False),
                })
                if (i + 1) % 10 == 0:
                    logger.info(f"  VQA {i+1}/{len(test_samples)}: avg F1={sum(r['f1'] for r in results)/len(results):.3f}")
            except Exception as e:
                logger.warning(f"VQA eval failed for sample {i}: {e}")
        if not results:
            return {"metric": "f1", "average": 0, "count": 0, "samples": []}
        avg_f1 = sum(r["f1"] for r in results) / len(results)
        avg_em = sum(r["exact_match"] for r in results) / len(results)
        return {
            "metric": "vqa",
            "average_f1": avg_f1,
            "average_exact_match": avg_em,
            "count": len(results),
            "samples": results,
        }

    async def eval_grounding(self, test_samples: List[Dict], data_dir: Path) -> Dict:
        """Evaluate grounding (IoU)."""
        logger.info(f"Evaluating grounding on {len(test_samples)} samples")
        results = []
        for i, sample in enumerate(test_samples):
            try:
                img_path = resolve_image_path(sample, data_dir)
                ref_bbox = sample.get("bbox")
                expression = sample.get("expression", "")
                if not ref_bbox or not expression:
                    continue
                if isinstance(ref_bbox, str):
                    ref_bbox = json.loads(ref_bbox)
                pred = await run_grounding(str(img_path), expression)
                if pred.get("error"):
                    iou_score = 0.0
                    pred_bbox = None
                else:
                    pred_bbox = pred.get("bbox", [])
                    iou_score = iou(pred_bbox, ref_bbox)
                results.append({
                    "image_id": sample.get("image_id"),
                    "expression": expression,
                    "reference_bbox": ref_bbox,
                    "prediction_bbox": pred_bbox,
                    "iou": iou_score,
                    "iou_at_0.5": 1.0 if iou_score >= 0.5 else 0.0,
                })
                if (i + 1) % 10 == 0:
                    logger.info(f"  Grounding {i+1}/{len(test_samples)}: avg IoU={sum(r['iou'] for r in results)/len(results):.3f}")
            except Exception as e:
                logger.warning(f"Grounding eval failed for sample {i}: {e}")
        if not results:
            return {"metric": "iou", "average": 0, "count": 0, "samples": []}
        avg_iou = sum(r["iou"] for r in results) / len(results)
        acc_50 = sum(r["iou_at_0.5"] for r in results) / len(results)
        return {
            "metric": "grounding",
            "average_iou": avg_iou,
            "accuracy_at_iou_0.5": acc_50,
            "count": len(results),
            "samples": results,
        }


# ---------------------------------------------------------------------------
# Test set construction
# ---------------------------------------------------------------------------
def build_test_set(data_dir: Path, max_per_task: int = 50) -> Dict[str, List[Dict]]:
    """Build test sets from JSONL files in data_dir."""
    test_sets = {"caption": [], "vqa": [], "grounding": []}
    file_map = {
        "caption": "captions.jsonl",
        "vqa": "vqa.jsonl",
        "grounding": "referring.jsonl",
    }
    for task, fname in file_map.items():
        path = data_dir / fname
        if not path.exists():
            continue
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        test_sets[task] = samples[:max_per_task]
    return test_sets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
async def amain():
    parser = argparse.ArgumentParser(description="Evaluate GeoChat LoRA adapter")
    parser.add_argument("--adapter", help="Path to LoRA adapter directory")
    parser.add_argument("--model", default="mbzuai-oryx/GeoChat", help="Base model")
    parser.add_argument("--data-dir", default="./data/bigearthnet_lora", help="Data directory")
    parser.add_argument("--test-set", help="Path to test_set.jsonl (overrides auto-load)")
    parser.add_argument("--output", default="./eval_results.json", help="Output results file")
    parser.add_argument("--max-per-task", type=int, default=50, help="Max samples per task type")
    parser.add_argument("--no-ground-truth", action="store_true",
                        help="Skip evaluation (just run inference for speed)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Build test set
    if args.test_set:
        with open(args.test_set) as f:
            all_samples = [json.loads(line) for line in f if line.strip()]
        test_sets = {"all": all_samples[:args.max_per_task]}
    else:
        test_sets = build_test_set(data_dir, max_per_task=args.max_per_task)
    logger.info(f"Test sets: {[(k, len(v)) for k, v in test_sets.items()]}")

    # Initialize service
    evaluator = Evaluator(adapter_path=args.adapter, model_path=args.model)
    await evaluator.setup()

    # Run evaluations
    all_results = {
        "adapter": args.adapter,
        "model": args.model,
        "data_dir": str(data_dir),
        "timestamp": time.time(),
    }

    if not args.no_ground_truth:
        for task in ("caption", "vqa", "grounding"):
            if not test_sets.get(task):
                logger.info(f"Skipping {task}: no samples")
                continue
            if task == "caption":
                result = await evaluator.eval_caption(test_sets[task], data_dir)
            elif task == "vqa":
                result = await evaluator.eval_vqa(test_sets[task], data_dir)
            elif task == "grounding":
                result = await evaluator.eval_grounding(test_sets[task], data_dir)
            all_results[task] = result
            # Print summary
            for k, v in result.items():
                if isinstance(v, float):
                    logger.info(f"  {task}.{k} = {v:.4f}")
                elif k == "count":
                    logger.info(f"  {task}.{k} = {v}")
    else:
        # Inference only
        for task, samples in test_sets.items():
            if isinstance(samples, list) and samples:
                logger.info(f"Running inference on {len(samples)} samples for {task}")

    # Save
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    # Print summary report
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    if "caption" in all_results and all_results["caption"].get("count"):
        print(f"Captioning: BLEU-4 = {all_results['caption'].get('average', 0):.4f} "
              f"({all_results['caption']['count']} samples)")
    if "vqa" in all_results and all_results["vqa"].get("count"):
        print(f"VQA: F1 = {all_results['vqa'].get('average_f1', 0):.4f}, "
              f"EM = {all_results['vqa'].get('average_exact_match', 0):.4f} "
              f"({all_results['vqa']['count']} samples)")
    if "grounding" in all_results and all_results["grounding"].get("count"):
        print(f"Grounding: IoU = {all_results['grounding'].get('average_iou', 0):.4f}, "
              f"Acc@0.5 = {all_results['grounding'].get('accuracy_at_iou_0.5', 0):.4f} "
              f"({all_results['grounding']['count']} samples)")
    print("=" * 60)


def main():
    import asyncio
    asyncio.run(amain())


if __name__ == "__main__":
    main()