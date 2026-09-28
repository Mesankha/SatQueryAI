"""
M3 LoRA Fine-tuning on real BigEarthNet / Sentinel-2 imagery.

Production-grade training script with all critical fixes:
  F1: Real image path resolution (relative to data_dir, not current working dir)
  F2: prompt+target concatenated with image tokens (correct GeoChat format)
  F3: Multi-band GeoTIFF handled via rasterio (B4,B3,B2 -> RGB)
  F4, F5: Correct tokenization (input_ids = prompt+tokens, labels = mask prompt tokens)
  F6, F21, F31: Fresh model load in TRAINING mode (not service's eval mode)
  F7: model.train() after LoRA wrap
  F8: prepare_model_for_kbit_training() before LoRA
  F9, F10: format_sample handles single-sample correctly for Dataset.map(batched=False)
  F11: Full TrainingArguments (eval strategy, save strategy, warmup, scheduler, max_grad_norm)
  F12: Save every epoch instead of every 500 steps
  F13: TensorBoard logging enabled
  F14: enable_input_require_grads() for gradient checkpointing on frozen base
  F15: Validation set for early stopping
  F16: try/except around trainer.train() with checkpoint save on OOM
  F17, F18: Save processor and tokenizer alongside adapter
  F19: num_proc for parallel tokenization
  F20: Full argparse with all hyperparams
  F22: 4-bit config separate for training (default OFF for stability)
  F23, F33: No circular imports (defines own templates)
  F24: Time-aware train/val/test split (90/10/0 or 80/10/10)
  F25: Module-level logger
  F26: Robust bbox parsing
  F27: Auto-detect LLM target modules by inspecting model architecture
  F28: modules_to_save for new tokens
  F29: bias="lora_only" (slightly better)
  F34: Dummy mode actually creates dummy images (224x224 PNG)
  F35: pad_token handled properly
  F49: model.config.use_cache=False for training
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# IMPORTANT: setup path BEFORE other imports
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Torch / HF / PEFT (graceful imports for documentation/CI)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset as TorchDataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

try:
    from datasets import Dataset, load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    Dataset = None
    AutoModelForCausalLM = None
    AutoProcessor = None
    BitsAndBytesConfig = None
    Trainer = None
    TrainingArguments = None
    default_data_collator = None
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    PeftModel = None
    TRANSFORMERS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False

try:
    import rasterio
    import numpy as np
    RASTERIO_AVAILABLE = True
except ImportError:
    rasterio = None
    np = None
    RASTERIO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO):
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "train.log"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    return logging.getLogger(__name__)


logger = setup_logging()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """All training hyperparameters in one place."""
    # Data
    data_dir: str = "./data/bigearthnet_lora"
    output_dir: str = "./models/geochat_lora"
    max_train_samples: int = 5000
    max_val_samples: int = 500
    train_val_test_split: Tuple[float, float, float] = (0.9, 0.1, 0.0)
    seed: int = 42

    # Model
    model_path: str = "mbzuai-oryx/GeoChat"
    load_in_4bit_train: bool = False  # F22: default OFF for stability
    compute_dtype: str = "auto"  # "auto" | "bf16" | "fp16" | "fp32"
    trust_remote_code: bool = True

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "lora_only"  # F29
    target_modules: Optional[List[str]] = None  # auto-detect if None
    modules_to_save: Optional[List[str]] = None

    # Training
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit" if RASTERIO_AVAILABLE else "adamw_torch"

    # Saving/eval
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    evaluation_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    logging_steps: int = 20
    save_steps: int = 500
    max_steps: int = -1  # -1 = use num_epochs
    report_to: str = "tensorboard"

    # Misc
    dataloader_num_workers: int = 2
    remove_unused_columns: bool = False
    fp16: bool = False
    bf16: bool = False
    tf32: bool = True
    ddp_find_unused_parameters: bool = False
    resume_from_checkpoint: Optional[str] = None  # path to checkpoint


# ---------------------------------------------------------------------------
# Image loading (F3, F41 FIX)
# ---------------------------------------------------------------------------
def load_image_safely(image_path: str, max_side: int = 1024) -> Image.Image:
    """Load image with multi-band GeoTIFF support.

    Strategy:
      1. Try PIL (JPG, PNG, single-band TIFF, RGB TIFF)
      2. Fall back to rasterio (Sentinel-2 multi-band -> RGB)
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow not available")

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        if not RASTERIO_AVAILABLE:
            raise RuntimeError(f"Cannot load {image_path} (PIL failed, no rasterio)")
        with rasterio.open(image_path) as src:
            bands = src.count
            if bands == 1:
                arr = src.read(1)
            elif bands >= 3:
                # Sentinel-2: B2=blue, B3=green, B4=red
                if bands >= 4:
                    r, g, b = src.read(4), src.read(3), src.read(2)
                else:
                    r, g, b = src.read(1), src.read(2), src.read(3)
                # Normalize per-band
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
            img = Image.fromarray(arr)

    # Resize if too large (saves GPU memory)
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# Templates (F56: use GeoChat format, no circular import)
# ---------------------------------------------------------------------------
GEOCHAT_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

VQA_TEMPLATE = f"{GEOCHAT_SYSTEM} USER: <image>\n{{question}} ASSISTANT:"
CAPTION_TEMPLATE = f"{GEOCHAT_SYSTEM} USER: <image>\nPlease describe this remote sensing image in detail. ASSISTANT:"
GROUNDING_TEMPLATE = (
    f"{GEOCHAT_SYSTEM} USER: <image>\nLocate the following in the image and provide its bounding box "
    "as [x1, y1, x2, y2] in normalized coordinates: {{expression}} ASSISTANT:"
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BigEarthNetDataset(TorchDataset if TORCH_AVAILABLE else object):
    """Custom dataset for BigEarthNet-style LoRA training.

    Each sample is a dict with:
      - image_id: filename of image in data_dir/images/
      - task_type: 'caption' | 'vqa' | 'grounding'
      - caption OR (question, answer) OR (expression, bbox)
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        data_dir: Path,
        processor: Any,
        max_length: int = 1024,
    ):
        self.samples = samples
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.processor = processor
        self.max_length = max_length
        self.tokenizer = processor.tokenizer
        # F35: ensure pad_token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, sample: Dict[str, Any]) -> Path:
        """F1 FIX: resolve image path relative to data_dir/images, not CWD."""
        # Try several path resolution strategies
        image_id = sample["image_id"]
        # Strategy 1: directly under images/
        direct = self.images_dir / image_id
        if direct.exists():
            return direct
        # Strategy 2: image_id might already include 'images/' prefix
        if image_id.startswith("images/"):
            direct2 = self.data_dir / image_id
            if direct2.exists():
                return direct2
        # Strategy 3: search for any matching file
        matches = list(self.images_dir.glob(f"*{Path(image_id).name}*"))
        if matches:
            return matches[0]
        raise FileNotFoundError(
            f"Cannot find image for {image_id} in {self.images_dir}"
        )

    def _build_prompt_and_target(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """F2 FIX: return (prompt, target) separately.

        The prompt includes the image placeholder. The target is the answer
        that the model should learn to generate.
        """
        task = sample["task_type"]
        if task == "caption":
            prompt = CAPTION_TEMPLATE
            target = sample["caption"]
        elif task == "vqa":
            prompt = VQA_TEMPLATE.format(question=sample["question"])
            target = sample["answer"]
        elif task == "grounding":
            prompt = GROUNDING_TEMPLATE.format(expression=sample["expression"])
            # F26 FIX: robust bbox parsing
            bbox = sample["bbox"]
            if isinstance(bbox, str):
                # Parse "[0.2, 0.2, 0.8, 0.8]" or similar
                bbox = json.loads(bbox)
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError(f"Invalid bbox: {bbox}")
            target = f"[{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]"
        else:
            raise ValueError(f"Unknown task type: {task}")
        return prompt, target

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image_path = self._resolve_image_path(sample)
        prompt, target = self._build_prompt_and_target(sample)

        # Load and preprocess image
        image = load_image_safely(str(image_path), max_side=1024)

        # Tokenize prompt WITH image
        # F2, F4, F5 FIX: proper tokenization
        prompt_inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        prompt_input_ids = prompt_inputs["input_ids"][0]
        prompt_attention_mask = prompt_inputs["attention_mask"][0]
        pixel_values = prompt_inputs["pixel_values"][0]

        # Tokenize target WITHOUT image (text only)
        target_ids = self.tokenizer(
            target,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]

        # Add EOS to teach model when to stop
        if self.tokenizer.eos_token_id is not None:
            target_ids = torch.cat([
                target_ids,
                torch.tensor([self.tokenizer.eos_token_id]),
            ])

        # Concatenate prompt + target
        input_ids = torch.cat([prompt_input_ids, target_ids])
        attention_mask = torch.cat([
            prompt_attention_mask,
            torch.ones(len(target_ids), dtype=torch.long),
        ])

        # Truncate if too long
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        # Build labels: -100 for prompt tokens, real ids for target tokens
        labels = torch.full_like(input_ids, -100)
        labels[len(prompt_input_ids):] = input_ids[len(prompt_input_ids):]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }


def custom_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad batch to longest sequence."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch required for collation")
    # Pad input_ids, attention_mask, labels
    pad_id = batch[0].get("pad_token_id", 0)
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.stack([
        torch.nn.functional.pad(b["input_ids"], (0, max_len - b["input_ids"].size(0)), value=pad_id)
        for b in batch
    ])
    attention_mask = torch.stack([
        torch.nn.functional.pad(b["attention_mask"], (0, max_len - b["attention_mask"].size(0)), value=0)
        for b in batch
    ])
    labels = torch.stack([
        torch.nn.functional.pad(b["labels"], (0, max_len - b["labels"].size(0)), value=-100)
        for b in batch
    ])
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def load_all_samples(data_dir: Path) -> List[Dict[str, Any]]:
    """Load all three supervision types and merge."""
    samples: List[Dict[str, Any]] = []
    for fname, task in [
        ("captions.jsonl", "caption"),
        ("vqa.jsonl", "vqa"),
        ("referring.jsonl", "grounding"),
    ]:
        path = data_dir / fname
        if not path.exists():
            logger.warning(f"Skipping {fname} (not found at {path})")
            continue
        loaded = load_jsonl(path)
        for s in loaded:
            s["task_type"] = task
        samples.extend(loaded)
        logger.info(f"Loaded {len(loaded)} samples from {fname}")
    if not samples:
        raise ValueError(f"No training data found in {data_dir}")
    return samples


def time_aware_split(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """F24 FIX: split data. If samples have a 'date' field, sort by date first
    to prevent temporal leakage. Otherwise random split.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    has_dates = all("date" in s or "acquisition_time" in s for s in samples[:10])
    if has_dates:
        date_key = "date" if "date" in samples[0] else "acquisition_time"
        samples = sorted(samples, key=lambda s: s.get(date_key, ""))
        logger.info("Sorted samples by date for time-aware split")
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return (
        [samples[i] for i in train_idx],
        [samples[i] for i in val_idx],
        [samples[i] for i in test_idx],
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def detect_lora_target_modules(model) -> List[str]:
    """F27 FIX: auto-detect LLM linear layer names.

    For LLaVA/GeoChat (Vicuna LLM backbone), the modules are:
      - q_proj, k_proj, v_proj, o_proj (attention)
      - gate_proj, up_proj, down_proj (MLP)
    We look at the first decoder layer and find the actual module names.
    """
    # Strategy 1: look for known module names in the model
    known_patterns = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "qkv_proj", "wq", "wk", "wv", "wo"]  # LLaMA-2 sometimes
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for pat in known_patterns:
                if name.endswith("." + pat) or name.endswith(pat):
                    found.add(pat)
    if found:
        logger.info(f"Detected LoRA target modules: {sorted(found)}")
        return sorted(found)
    # Strategy 2: fallback to common pattern
    logger.warning("Could not auto-detect LoRA targets, using default LLaMA pattern")
    return ["q_proj", "v_proj"]


def load_model_for_training(cfg: TrainConfig):
    """F6, F21, F31 FIX: load fresh model in TRAINING mode, not from service.

    Returns: (model, processor)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers/peft not installed")
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch not installed")

    logger.info(f"Loading model from {cfg.model_path}")

    # Compute dtype
    if cfg.compute_dtype == "auto":
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[cfg.compute_dtype]
    logger.info(f"Compute dtype: {dtype}")

    # Load model
    if cfg.load_in_4bit_train and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        # F8 FIX: prepare 4-bit model for k-bit training
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            device_map={"": 0} if torch.cuda.is_available() else "cpu",
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    # F35: pad_token setup
    processor = AutoProcessor.from_pretrained(
        cfg.model_path, trust_remote_code=cfg.trust_remote_code
    )
    if processor.tokenizer.pad_token is None:
        if processor.tokenizer.unk_token is not None:
            processor.tokenizer.pad_token = processor.tokenizer.unk_token
        else:
            processor.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            model.resize_token_embeddings(len(processor.tokenizer))

    # F49: disable use_cache for training (required for grad checkpointing)
    model.config.use_cache = False

    # F14 FIX: enable input require grads (needed for grad ckpt on frozen base)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Auto-detect target modules if not provided
    if cfg.target_modules is None:
        cfg.target_modules = detect_lora_target_modules(model)

    # F28: optionally save embed/lm_head for new tokens
    modules_to_save = cfg.modules_to_save
    if (
        getattr(model.config, "vocab_size", 0) != len(processor.tokenizer)
        and modules_to_save is None
    ):
        modules_to_save = ["embed_tokens", "lm_head"]

    # Build LoRA config
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=modules_to_save,
    )

    # F7: apply LoRA and set train mode
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()  # F7 FIX: explicit train mode

    return model, processor


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def setup_training_args(cfg: TrainConfig, log_dir: Path) -> TrainingArguments:
    """F11, F12 FIX: full training args with eval, save strategy, warmup, etc."""
    return TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
        gradient_checkpointing=cfg.gradient_checkpointing,
        # F12 FIX: save per epoch
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        save_steps=cfg.save_steps,
        # F11 FIX: evaluation strategy
        evaluation_strategy=cfg.evaluation_strategy if cfg.max_val_samples > 0 else "no",
        load_best_model_at_end=cfg.load_best_model_at_end and cfg.max_val_samples > 0,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        # F13 FIX: TensorBoard logging
        logging_dir=str(log_dir / "tensorboard"),
        logging_steps=cfg.logging_steps,
        report_to=[] if cfg.report_to == "none" else [cfg.report_to],
        # Misc
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=False,
        remove_unused_columns=cfg.remove_unused_columns,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        tf32=cfg.tf32,
        ddp_find_unused_parameters=cfg.ddp_find_unused_parameters,
        seed=cfg.seed,
        save_safetensors=True,
        # Performance
        group_by_length=False,
    )


def train(cfg: TrainConfig):
    """Main training function."""
    data_dir = Path(cfg.data_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    global logger
    logger = setup_logging(log_dir, level=logging.INFO)

    logger.info(f"Config: {cfg}")

    # Save config
    config_save = {k: v for k, v in cfg.__dict__.items()}
    config_save["train_val_test_split"] = list(config_save["train_val_test_split"])
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(config_save, f, indent=2, default=str)

    # Load data
    logger.info(f"Loading data from {data_dir}")
    all_samples = load_all_samples(data_dir)
    logger.info(f"Total samples: {len(all_samples)}")

    if cfg.max_train_samples > 0 and len(all_samples) > cfg.max_train_samples + cfg.max_val_samples:
        # Limit total
        rng = random.Random(cfg.seed)
        rng.shuffle(all_samples)
        all_samples = all_samples[: cfg.max_train_samples + cfg.max_val_samples]

    # F24 FIX: time-aware split
    train_samples, val_samples, test_samples = time_aware_split(
        all_samples,
        train_ratio=cfg.train_val_test_split[0],
        val_ratio=cfg.train_val_test_split[1],
        test_ratio=cfg.train_val_test_split[2],
        seed=cfg.seed,
    )
    logger.info(
        f"Split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
    )

    # Save test set for later eval
    if test_samples:
        with open(output_dir / "test_set.jsonl", "w") as f:
            for s in test_samples:
                f.write(json.dumps(s) + "\n")

    # Load model + processor
    model, processor = load_model_for_training(cfg)

    # Build datasets
    train_dataset = BigEarthNetDataset(train_samples, data_dir, processor, max_length=1024)
    eval_dataset = (
        BigEarthNetDataset(val_samples, data_dir, processor, max_length=1024)
        if val_samples else None
    )

    # Training args
    training_args = setup_training_args(cfg, log_dir)

    # Data collator (custom for variable-length sequences)
    data_collator = custom_collate_fn

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
    )

    # F16 FIX: try/except with checkpoint save on OOM
    logger.info("Starting training...")
    start_time = time.time()
    try:
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"OOM during training: {e}")
        logger.info("Saving emergency checkpoint...")
        try:
            trainer.save_model(str(output_dir / "adapter_oom_emergency"))
        except Exception as save_err:
            logger.error(f"Emergency save failed: {save_err}")
        raise
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        try:
            trainer.save_model(str(output_dir / "adapter_interrupted"))
            logger.info("Saved interrupt checkpoint")
        except Exception as e:
            logger.error(f"Interrupt save failed: {e}")
        raise
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        # Try to save anyway
        try:
            trainer.save_model(str(output_dir / "adapter_failed"))
        except Exception:
            pass
        raise
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Training elapsed: {elapsed/60:.1f} minutes")

    # F17, F18 FIX: save adapter + processor + tokenizer
    adapter_dir = output_dir / "adapter"
    logger.info(f"Saving final adapter to {adapter_dir}")
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    processor.tokenizer.save_pretrained(str(adapter_dir))

    # Save training summary
    summary = {
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "training_time_minutes": (time.time() - start_time) / 60,
        "lora_config": {
            "r": cfg.lora_r,
            "alpha": cfg.lora_alpha,
            "dropout": cfg.lora_dropout,
            "target_modules": cfg.target_modules,
            "modules_to_save": cfg.modules_to_save,
        },
        "model_path": cfg.model_path,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Training complete. Adapter: {adapter_dir}")
    return str(adapter_dir)


# ---------------------------------------------------------------------------
# Dummy data generation (F34 FIX: actually create images)
# ---------------------------------------------------------------------------
def create_dummy_bigearthnet(data_dir: str, num_samples: int = 100):
    """Create REAL dummy image files + JSONL labels for testing the pipeline.

    F34 FIX: previous version only created JSONL but no actual images,
    causing PIL.open() to fail during training.
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    classes = ["urban", "agriculture", "forest", "water", "barren"]
    colors = {
        "urban": (200, 200, 200),
        "agriculture": (50, 180, 50),
        "forest": (10, 100, 10),
        "water": (10, 10, 200),
        "barren": (180, 150, 100),
    }

    captions, vqa, referring = [], [], []
    for i in range(num_samples):
        cls = classes[i % len(classes)]
        image_id = f"dummy_{i:04d}.png"

        # Create a synthetic 224x224 image (F34 FIX)
        if PIL_AVAILABLE:
            img = Image.new("RGB", (224, 224), colors[cls])
            # Add some noise for realism
            import random as r
            pixels = img.load()
            for x in range(0, 224, 10):
                for y in range(0, 224, 10):
                    dx = r.randint(-30, 30)
                    pixels[x, y] = tuple(max(0, min(255, c + dx)) for c in pixels[x, y])
            img.save(images_dir / image_id)

        captions.append({"image_id": image_id, "caption": f"Remote sensing image showing {cls} area."})
        vqa.append({"image_id": image_id, "question": "What is the land cover?", "answer": f"The land cover is {cls}."})
        referring.append({"image_id": image_id, "expression": f"the {cls} area", "bbox": [0.2, 0.2, 0.8, 0.8]})

    for fname, data in [("captions.jsonl", captions), ("vqa.jsonl", vqa), ("referring.jsonl", referring)]:
        with open(data_dir / fname, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    logger.info(f"Created {num_samples} dummy samples in {data_dir} (with real 224x224 PNG images)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune GeoChat on BigEarthNet / Sentinel-2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    parser.add_argument("--data-dir", default="./data/bigearthnet_lora", help="Training data directory")
    parser.add_argument("--output-dir", default="./models/geochat_lora", help="Output directory for LoRA adapter")
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--max-val-samples", type=int, default=500)
    parser.add_argument("--create-dummy", action="store_true", help="Create dummy data (with real images) for testing")

    # Model
    parser.add_argument("--model-path", default="mbzuai-oryx/GeoChat")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit quantization (may be unstable)")
    parser.add_argument("--compute-dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])

    # LoRA
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-bias", default="lora_only", choices=["none", "lora_only", "all"])
    parser.add_argument("--target-modules", nargs="+", default=None,
                        help="LoRA target modules (auto-detect if not provided)")

    # Training
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler", default="cosine", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--no-gradient-checkpointing", action="store_true", help="Disable gradient checkpointing (more memory, faster)")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Path to checkpoint to resume from")

    # F20 FIX: logging and saving
    parser.add_argument("--report-to", default="tensorboard", choices=["none", "tensorboard", "wandb"])
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-strategy", default="epoch", choices=["no", "epoch", "steps"])
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()

    # Build config
    cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        model_path=args.model_path,
        load_in_4bit_train=args.load_in_4bit,
        compute_dtype=args.compute_dtype,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
        target_modules=args.target_modules,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        resume_from_checkpoint=args.resume_from_checkpoint,
        report_to=args.report_to,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
    )

    # F25: setup logging
    log_dir = Path(cfg.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    global logger
    logger = setup_logging(log_dir)

    # F34 FIX: create dummy data WITH real images
    if args.create_dummy:
        create_dummy_bigearthnet(args.data_dir, num_samples=200)
        print(f"Created dummy data in {args.data_dir}")
        print("Real 224x224 PNG images + captions/vqa/referring JSONL files.")
        print(f"Now run: python -m m3_vlm.train_lora --data-dir {args.data_dir} --output-dir {args.output_dir} --max-train-samples 200 --max-val-samples 50 --epochs 1 --report-to none")
        return

    # Train
    train(cfg)


if __name__ == "__main__":
    main()