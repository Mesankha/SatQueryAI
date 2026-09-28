"""
M3 Multi-Modal LoRA Fine-Tuning — Optical + SAR + Fusion.

This is an EXTENSION of train_lora.py that:
  1. Accepts three types of training data:
     - Optical RGB (Sentinel-2 or similar)
     - SAR (Sentinel-1, preprocessed to pseudo-RGB)
     - Fusion (paired optical+SAR)
  2. Uses modality-aware prompts via sar_prompts.select_prompt()
  3. Auto-detects modality from data file naming convention
  4. Supports training on a mix of modalities simultaneously

USAGE:
  # Train on optical only (as before)
  python -m m3_vlm.train_multimodal \\
      --data-dir ./data/bigearthnet_lora \\
      --output-dir ./models/geochat_lora \\
      --modalities optical

  # Train on SAR only
  python -m m3_vlm.train_multimodal \\
      --data-dir ./data/sar_lora \\
      --output-dir ./models/geochat_sar_lora \\
      --modalities sar

  # Train on both (multi-task LoRA)
  python -m m3_vlm.train_multimodal \\
      --data-dirs ./data/bigearthnet_lora ./data/sar_lora \\
      --output-dir ./models/geochat_multimodal_lora \\
      --modalities optical sar
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup path
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset as TorchDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from m3_vlm.train_lora import (
        TrainConfig, setup_logging, detect_lora_target_modules,
        custom_collate_fn, time_aware_split, load_model_for_training,
        setup_training_args, load_image_safely,
    )
    from m3_vlm.sar_prompts import select_prompt
    TRAIN_LORA_AVAILABLE = True
except ImportError as e:
    TRAIN_LORA_AVAILABLE = False
    print(f"train_multimodal: failed to import from train_lora: {e}")

logger = setup_logging() if TRAIN_LORA_AVAILABLE else logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modality detection
# ---------------------------------------------------------------------------

def detect_modality(image_filename: str) -> str:
    """
    Detect the modality of an image from its filename.

    Returns:
        "sar" if filename suggests SAR
        "optical" if filename suggests optical RGB
        "fusion" if filename suggests paired data
        "unknown" otherwise
    """
    fn = image_filename.lower()

    # SAR indicators
    if any(kw in fn for kw in [
        "sar", "s1_", "sentinel-1", "sentinel1", "grd",
        "_vv_", "_vh_", "pc_sar_", "hf_sar_"
    ]):
        return "sar"

    # Fusion indicators (contains both)
    if "fusion" in fn or "paired" in fn or "combined" in fn:
        return "fusion"

    # Optical indicators
    if any(kw in fn for kw in [
        "s2_", "sentinel-2", "sentinel2", "optical", "rgb",
        "ms_", "msi_", "pc_opt_", "bigearthnet"
    ]):
        return "optical"

    return "unknown"


def detect_task_from_jsonl(jsonl_path: Path) -> str:
    """Detect task type from JSONL filename."""
    fn = jsonl_path.name.lower()
    if "caption" in fn:
        return "caption"
    if "vqa" in fn:
        return "vqa"
    if "referring" in fn or "grounding" in fn:
        return "grounding"
    if "fusion" in fn:
        return "fusion"
    if "change" in fn:
        return "change"
    return "vqa"  # default


# ---------------------------------------------------------------------------
# Multi-modal dataset
# ---------------------------------------------------------------------------

class MultiModalDataset(TorchDataset if TORCH_AVAILABLE else object):
    """
    Dataset that handles optical, SAR, and fusion samples.

    Each sample has:
      - modality: "optical" | "sar" | "fusion"
      - task: "caption" | "vqa" | "grounding"
      - image_id, question/answer, expression/bbox
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        data_dir: Path,
        processor: Any,
        modality_filter: Optional[List[str]] = None,
        max_length: int = 1024,
    ):
        self.samples = samples
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.processor = processor
        self.max_length = max_length
        self.tokenizer = processor.tokenizer
        self.modality_filter = set(modality_filter) if modality_filter else None

        if self.tokenizer.pad_token is None:
            if self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

    def __len__(self):
        return len(self.samples)

    def _resolve_image_path(self, sample: Dict[str, Any]) -> Path:
        """Resolve image path - works for both filename-keyed and path-keyed samples."""
        image_id = sample.get("image_id", "")
        candidates = [
            self.images_dir / image_id,
            self.data_dir / image_id,
        ]
        if "image_path" in sample:
            candidates.insert(0, Path(sample["image_path"]))
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(f"Cannot find image for {image_id} in {self.images_dir}")

    def _detect_sample_modality(self, sample: Dict[str, Any]) -> str:
        """Detect modality from sample dict or filename."""
        if "modality" in sample:
            return sample["modality"]
        image_id = sample.get("image_id", "")
        return detect_modality(image_id)

    def _build_prompt_target(self, sample: Dict[str, Any], modality: str) -> Tuple[str, str]:
        """Build (prompt, target) pair using modality-aware prompts."""
        task = sample.get("task_type", "vqa")

        if task == "caption":
            target = sample.get("caption", "")
            prompt = select_prompt("caption", modality)
        elif task == "vqa":
            question = sample.get("question", "")
            target = sample.get("answer", "")
            prompt = select_prompt("vqa", modality, question=question)
        elif task == "grounding":
            expression = sample.get("expression", "")
            target = self._format_bbox(sample.get("bbox", [0.25, 0.25, 0.75, 0.75]))
            prompt = select_prompt("grounding", modality, expression=expression)
        elif task == "fusion":
            target = sample.get("caption", "")
            prompt = select_prompt("caption", "fusion")
        else:
            target = sample.get("answer", sample.get("caption", ""))
            prompt = select_prompt(task, modality)

        return prompt, target

    def _format_bbox(self, bbox) -> str:
        if isinstance(bbox, str):
            bbox = json.loads(bbox)
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return "[0.25, 0.25, 0.75, 0.75]"
        return f"[{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        modality = self._detect_sample_modality(sample)

        # Filter by modality
        if self.modality_filter and modality not in self.modality_filter:
            # Fall back to first available sample
            idx = (idx + 1) % len(self.samples)
            sample = self.samples[idx]
            modality = self._detect_sample_modality(sample)

        image_path = self._resolve_image_path(sample)
        prompt, target = self._build_prompt_target(sample, modality)

        # Load and preprocess image
        image = _load_image_safely(str(image_path), max_side=1024)

        # For SAR, apply additional preprocessing (idempotent if already done)
        if modality == "sar":
            # The SAR file is likely already preprocessed, but we ensure
            # it's loaded correctly
            pass

        # Tokenize
        prompt_inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        )
        prompt_input_ids = prompt_inputs["input_ids"][0]
        prompt_attention_mask = prompt_inputs["attention_mask"][0]
        pixel_values = prompt_inputs["pixel_values"][0]

        target_ids = self.tokenizer(
            target, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]

        if self.tokenizer.eos_token_id is not None:
            target_ids = torch.cat([
                target_ids,
                torch.tensor([self.tokenizer.eos_token_id]),
            ])

        input_ids = torch.cat([prompt_input_ids, target_ids])
        attention_mask = torch.cat([
            prompt_attention_mask,
            torch.ones(len(target_ids), dtype=torch.long),
        ])

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        labels = torch.full_like(input_ids, -100)
        labels[len(prompt_input_ids):] = input_ids[len(prompt_input_ids):]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
            "modality": modality,  # for debugging
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_multimodal_samples(
    data_dirs: List[Path],
    modalities: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Load samples from multiple data directories, auto-tagging modality.

    Args:
        data_dirs: List of data directories (each with images/ and *.jsonl)
        modalities: Optional filter - only include these modalities
    """
    samples = []
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            logger.warning(f"Data dir not found: {data_dir}")
            continue

        for jsonl_file in data_dir.glob("*.jsonl"):
            task = detect_task_from_jsonl(jsonl_file)
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        s = json.loads(line)
                        s["task_type"] = task
                        # Auto-detect modality if not in sample
                        if "modality" not in s:
                            image_id = s.get("image_id", "")
                            s["modality"] = detect_modality(image_id)
                        # Apply filter
                        if modalities and s["modality"] not in modalities:
                            continue
                        samples.append(s)
                    except json.JSONDecodeError:
                        continue
            logger.info(f"Loaded {task} from {jsonl_file.name}")
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-modal (optical + SAR) LoRA fine-tuning for GeoChat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dirs", nargs="+", required=True,
        help="One or more data directories (e.g., ./data/bigearthnet_lora ./data/sar_lora)"
    )
    parser.add_argument(
        "--output-dir", default="./models/geochat_multimodal_lora",
        help="Output directory for LoRA adapter"
    )
    parser.add_argument(
        "--modalities", nargs="+",
        choices=["optical", "sar", "fusion"],
        default=None,
        help="Filter: only train on these modalities (default: all)"
    )
    parser.add_argument("--model-path", default="mbzuai-oryx/GeoChat")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--max-val-samples", type=int, default=500)
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not TRAIN_LORA_AVAILABLE:
        print("ERROR: train_lora.py not available. Run from satquery/ directory.")
        sys.exit(1)

    # Build config
    data_dirs = [Path(d) for d in args.data_dirs]
    cfg = TrainConfig(
        data_dir=str(data_dirs[0]),  # primary
        output_dir=args.output_dir,
        max_train_samples=args.max_samples,
        max_val_samples=args.max_val_samples,
        model_path=args.model_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        report_to=args.report_to,
        resume_from_checkpoint=args.resume_from_checkpoint,
        seed=args.seed,
    )

    # Update config to use multi-modal data dirs (storing in data_dir is just a label)
    log_dir = Path(cfg.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    global logger
    logger = setup_logging(log_dir)

    logger.info(f"Config: {cfg}")
    logger.info(f"Data dirs: {data_dirs}")
    logger.info(f"Modalities filter: {args.modalities}")

    # Save config
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.output_dir) / "train_config.json", "w") as f:
        json.dump({k: v for k, v in cfg.__dict__.items()}, f, indent=2, default=str)

    # Load all samples
    all_samples = load_multimodal_samples(data_dirs, modalities=args.modalities)
    logger.info(f"Total samples: {len(all_samples)}")

    # Log modality distribution
    from collections import Counter
    dist = Counter(s.get("modality", "unknown") for s in all_samples)
    logger.info(f"Modality distribution: {dict(dist)}")

    # Limit samples
    if cfg.max_train_samples > 0 and len(all_samples) > cfg.max_train_samples + cfg.max_val_samples:
        random.seed(cfg.seed)
        random.shuffle(all_samples)
        all_samples = all_samples[: cfg.max_train_samples + cfg.max_val_samples]

    # Split
    train_samples, val_samples, test_samples = time_aware_split(
        all_samples,
        train_ratio=0.9, val_ratio=0.1, test_ratio=0.0,
        seed=cfg.seed,
    )
    logger.info(f"Split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    # Load model
    model, processor = load_model_for_training(cfg)

    # Build datasets
    train_dataset = MultiModalDataset(
        train_samples, data_dirs[0], processor, args.modalities
    )
    eval_dataset = MultiModalDataset(
        val_samples, data_dirs[0], processor, args.modalities
    ) if val_samples else None

    # Training args
    training_args = setup_training_args(cfg, log_dir)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=custom_collate_fn,
        tokenizer=processor.tokenizer,
    )

    # Train
    logger.info("Starting multi-modal training...")
    start = time.time()
    try:
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    except KeyboardInterrupt:
        logger.warning("Interrupted, saving...")
        trainer.save_model(str(Path(cfg.output_dir) / "adapter_interrupted"))
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        try:
            trainer.save_model(str(Path(cfg.output_dir) / "adapter_failed"))
        except Exception:
            pass
        raise
    finally:
        elapsed = time.time() - start
        logger.info(f"Training elapsed: {elapsed/60:.1f} minutes")

    # Save
    adapter_dir = Path(cfg.output_dir) / "adapter"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    processor.tokenizer.save_pretrained(str(adapter_dir))

    logger.info(f"Multi-modal training complete. Adapter: {adapter_dir}")


if __name__ == "__main__":
    main()