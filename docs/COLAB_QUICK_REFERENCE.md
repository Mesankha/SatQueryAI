# 🚀 SatQuery AI - Google Colab Quick Reference Card

## ⚡ TL;DR - The Only Command That Matters

```python
!python -m m3_vlm.train_multimodal \
    --data-dirs ./data/sar_lora ./data/optical_lora \
    --output-dir ./models/geochat_india_sar \
    --modalities sar optical \
    --model-path mbzuai-oryx/GeoChat \
    --epochs 3
```

## 📋 Colab Cell Order (Copy-Paste)

| # | What | Time | Skip? |
|---|------|------|-------|
| 1 | Install packages | 3-5 min | Required |
| 2 | Mount Google Drive | 30 sec | Required |
| 3 | Copy code to Drive | 1 min | Required |
| 4 | Get SAR + optical data | 10-20 min | **NO BigEarthNet needed!** |
| 5 | Sanity check | 3-5 min | Optional but recommended |
| 6 | **Real training** | **4-6 hours** | This is the main step |
| 7 | TensorBoard | 0 min (parallel) | Optional |
| 8 | Test model | 1 min | Required |
| 9 | Evaluate metrics | 2 min | Optional |
| 10 | Backup to Drive | 1 min | Required |
| 11 | Deploy with ngrok | 2 min | Optional |

## 💡 Why No BigEarthNet?

| BigEarthNet | Planetary Computer |
|-------------|-------------------|
| 15-68GB download | 0 GB |
| European only | Any AOI (Indian cities!) |
| RGB only | SAR + Optical |
| 2017-2018 | 2024+ |
| Registration needed | No auth |

**For your use case: skip BigEarthNet, use Planetary Computer.**

## 🎯 The Critical Path

```
Cell 1 (install) → Cell 2 (drive) → Cell 3 (code) → 
Cell 4 (data, 20 min) → Cell 6 (train, 4-6 hrs) → DONE!
```

## 📁 Where Things Go in Google Drive

```
/content/drive/MyDrive/satquery/
├── data/
│   ├── sar_lora/         # SAR images (auto-downloaded)
│   │   ├── images/
│   │   ├── captions.jsonl
│   │   ├── vqa.jsonl
│   │   └── referring.jsonl
│   └── optical_lora/     # Optical images (auto-downloaded)
│       └── ...
├── models/
│   └── geochat_india_sar/  # Trained adapter (200MB)
│       ├── adapter/        # ← THIS is what you download
│       ├── logs/
│       └── train_config.json
└── m3_vlm/                # All the code
```

## ⚠️ Common Issues & Fixes

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: m3_vlm` | Add `sys.path.insert(0, '/content/satquery')` |
| `CUDA out of memory` | Add `--max-train-samples 1500 --lora-r 8` |
| `Disconnected after 12 hrs` | Re-run Cell 6 with `--resume-from-checkpoint` |
| `Planetary Computer down` | Use Cell 4 OPTION B: HuggingFace SEN12MS |
| `No module named 'flask'` | Skip Cell 11, just use local M3 service |
| `FileNotFoundError: data/sar_lora` | Re-run Cell 4 |

## 🎛️ Training Config Presets

### Free T4 GPU (Colab default)
```python
!python -m m3_vlm.train_multimodal \
    --data-dirs ./data/sar_lora ./data/optical_lora \
    --output-dir ./models/geochat_india_sar \
    --modalities sar optical \
    --epochs 3 \
    --batch-size 1 --grad-accum 8 \
    --lora-r 16 --lora-alpha 32
```

### Colab Pro T4 (more VRAM available)
```python
!python -m m3_vlm.train_multimodal \
    --data-dirs ./data/sar_lora ./data/optical_lora \
    --output-dir ./models/geochat_india_sar \
    --modalities sar optical \
    --epochs 5 \
    --batch-size 2 --grad-accum 4 \
    --lora-r 32 --lora-alpha 64 \
    --max-train-samples 5000
```

### Colab Pro A100 (fastest)
```python
!python -m m3_vlm.train_multimodal \
    --data-dirs ./data/sar_lora ./data/optical_lora \
    --output-dir ./models/geochat_india_sar \
    --modalities sar optical \
    --epochs 3 \
    --batch-size 4 --grad-accum 2 \
    --lora-r 32 --lora-alpha 64 \
    --max-train-samples 5000
```

### SAR-only (specialized)
```python
!python -m m3_vlm.train_multimodal \
    --data-dirs ./data/sar_lora \
    --output-dir ./models/geochat_sar_only \
    --modalities sar \
    --epochs 5 \
    --lora-r 16
```

## 🔄 Resuming After Disconnect

```python
# List available checkpoints
!ls /content/satquery/models/geochat_india_sar/

# Resume from latest
!python -m m3_vlm.train_multimodal \
    --data-dirs ./data/sar_lora ./data/optical_lora \
    --output-dir ./models/geochat_india_sar \
    --resume-from-checkpoint ./models/geochat_india_sar/checkpoint-500 \
    --epochs 3
```

## 📥 Download Trained Model

After training:
```python
# Zip the adapter
!cd /content/satquery/models/geochat_india_sar && zip -r adapter.zip adapter/

# Option A: Download directly
from google.colab import files
files.download('adapter.zip')

# Option B: Already in Drive at:
# /content/drive/MyDrive/satquery/models/geochat_india_sar/adapter/
```

## 🚀 Use on Local Machine

```powershell
# 1. Download adapter from Drive to local machine
# 2. Set environment variable
$env:LORA_ADAPTER_PATH = "D:\path\to\adapter"

# 3. Start M3 service
cd D:\VxKex\c\Sat-query-AI\satquery
python -m m3_vlm.service --port 8001

# 4. Test
curl http://localhost:8001/health
```

## 🎯 The Bottom Line

> **You do NOT need BigEarthNet.**
> Use Planetary Computer to get real Indian city SAR data for free.
> Train on Colab free T4 for 4-6 hours.
> Get a 200MB LoRA adapter that understands both optical and SAR.

That's it. Everything else is optimization.