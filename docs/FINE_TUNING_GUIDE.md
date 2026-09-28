# SatQuery AI — Fine-Tuning GeoChat: Complete Walkthrough

This guide gives you the **exact commands, in order**, to:
1. Get optical training data (BigEarthNet)
2. Get SAR training data (Planetary Computer, free, no auth)
3. Fine-tune GeoChat on both modalities
4. Test the trained model
5. Deploy and use it

## 📁 WHERE THINGS GO (FOLDER MAP)

```
D:\VxKex\c\Sat-query-AI\satquery\
│
├── data\                              # ALL training data lives here
│   ├── raw\                           # Where you put downloaded data
│   │   ├── bigearthnet\               # ← PUT BigEarthNet here
│   │   └── sar\                       # (optional: raw SAR if you have it)
│   │
│   ├── bigearthnet_lora\              # ← Generated from BigEarthNet
│   │   ├── images\                    # PNG files
│   │   ├── captions.jsonl
│   │   ├── vqa.jsonl
│   │   └── referring.jsonl
│   │
│   ├── sar_lora\                      # ← Generated from Planetary Computer
│   │   ├── images\                    # SAR pseudo-RGB PNG files
│   │   ├── captions.jsonl
│   │   ├── vqa.jsonl
│   │   └── referring.jsonl
│   │
│   ├── indian_cities\                 # ← Real scenes for demo
│   │   ├── delhi\                     # Metadata for each city
│   │   ├── mumbai\
│   │   └── ...
│   │
│   ├── catalog.db                     # M0 SQLite catalog
│   └── image_cache\                   # Lazy-downloaded images
│
├── models\                            # Trained models
│   └── geochat_multimodal_lora\       # ← LoRA adapter goes here
│       ├── adapter\                   # The actual trained weights
│       ├── logs\                      # TensorBoard logs
│       └── train_config.json
│
├── m3_vlm\                            # All the code
│   ├── setup_check.py                 # Run first to verify environment
│   ├── test_pipeline_e2e.py           # Run second to verify code
│   ├── acquire_data.py                # Get optical data
│   ├── acquire_sar.py                 # Get SAR data
│   ├── train_multimodal.py            # Train!
│   ├── test_inference.py              # Test the trained model
│   ├── evaluate.py                    # Compute metrics
│   └── service.py                     # Deploy the model
│
└── DATA_STRATEGY.md                   # Full data architecture
```

---

## 🚀 PHASE 1: One-time Setup (15 minutes)

### Step 1.1: Open PowerShell in the project

```powershell
cd D:\VxKex\c\Sat-query-AI\satquery
```

### Step 1.2: Install packages

```powershell
# Core ML packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate bitsandbytes datasets

# Geospatial packages
pip install rasterio shapely

# Data acquisition
pip install planetary-computer pystac-client

# For the Flask service (optional, only if you want to deploy)
pip install flask flask-cors
```

### Step 1.3: Verify everything is set up

```powershell
python -m m3_vlm.setup_check
```

Expected output: Shows which packages are installed, GPU info, folder status.

**Don't proceed if this fails.** Fix the missing packages first.

### Step 1.4: Run the pipeline test (no GPU needed)

```powershell
python -m m3_vlm.test_pipeline_e2e
```

Expected: ~45 PASS, 1 FAIL (the fail is `m3_vlm.service` needing flask, which is optional)

---

## 📥 PHASE 2: Get Optical Training Data (BigEarthNet)

### Step 2.1: Download BigEarthNet

1. Go to **http://bigearth.net/**
2. Register for a free account
3. Download **BigEarthNet-MM** (the RGB-only version, ~15GB)
   - This is much smaller than the full 68GB multi-band version
   - Each patch is a 256×256 RGB PNG
4. Extract to: `D:\VxKex\c\Sat-query-AI\satquery\data\raw\bigearthnet\`

After extraction, the folder should look like:
```
data\raw\bigearthnet\
    S2A_MSIL2A_20170613T101031_8_36\
        S2A_MSIL2A_20170613T101031_8_36.png       (RGB image)
        S2A_MSIL2A_20170613T101031_8_36_labels.json (land cover labels)
    S2A_MSIL2A_20170613T101031_8_37\
        ...
```

### Step 2.2: Convert to LoRA training format

```powershell
python -m m3_vlm.acquire_data train `
    --source ./data/raw/bigearthnet `
    --output ./data/bigearthnet_lora `
    --max-samples 2000
```

What this does:
- Walks all patch directories
- Reads labels_metadata.json for each patch
- Generates captions, VQA, and referring expression JSONL files
- Creates symlinks (or copies) of images into `./data/bigearthnet_lora/images/`
- Modality is auto-tagged as "optical" based on filename

Expected time: 2-5 minutes for 2000 patches
Expected output: 6000+ samples (2000 each for caption/VQA/referring)

Verify:
```powershell
Get-ChildItem .\data\bigearthnet_lora\images | Measure-Object
Get-Content .\data\bigearthnet_lora\captions.jsonl | Select-Object -First 3
```

---

## 📡 PHASE 3: Get SAR Training Data (Planetary Computer)

**This is the magic step** — you get real Sentinel-1 SAR data over India, **for free, no authentication**.

### Step 3.1: Query for SAR data over Indian cities

```powershell
python -m m3_vlm.acquire_sar cities `
    --cities delhi mumbai kerala_flood rajasthan `
    --output ./data/sar_lora `
    --max-per-city 200 `
    --start-date 2024-01-01 `
    --end-date 2024-12-31
```

What this does:
- Connects to Microsoft Planetary Computer STAC API
- Queries Sentinel-1 GRD (VV+VH polarizations) for each city
- Downloads both VV and VH bands for each scene
- Combines them into a 2-band GeoTIFF
- Generates SAR-specific captions, VQA, referring expressions
- Modality is auto-tagged as "sar" based on filename

Expected time: 10-20 minutes (depends on internet speed)
Expected output: 400-800 SAR samples (200 per city × 4 cities, but Planetary Computer may have fewer scenes)

### Step 3.2 (Optional): Also get optical pairing

If you want paired optical+SAR data for fusion training, run:
```powershell
python -m m3_vlm.acquire_sar cities `
    --cities delhi mumbai `
    --output ./data/sar_lora `
    --max-per-city 100
```

This automatically tries to pair each SAR with a Sentinel-2 from the same date.

### Verify

```powershell
Get-ChildItem .\data\sar_lora\images | Measure-Object
Get-Content .\data\sar_lora\captions.jsonl | Select-Object -First 2
```

You should see entries like:
```json
{"image_id": "pc_sar_S1A_IW_GRDH_1SDV_20240115T060000.tif", "caption": "Sentinel-1 SAR image showing urban with characteristic backscatter patterns."}
```

---

## 🔥 PHASE 4: Fine-Tune GeoChat (4-8 hours on RTX 3050)

**This is the actual fine-tuning step.**

### Step 4.1: Start the training

```powershell
python -m m3_vlm.train_multimodal `
    --data-dirs ./data/bigearthnet_lora ./data/sar_lora `
    --output-dir ./models/geochat_multimodal_lora `
    --modalities optical sar `
    --model-path mbzuai-oryx/GeoChat `
    --epochs 3 `
    --lora-r 16 --lora-alpha 32 `
    --batch-size 1 --grad-accum 8 `
    --lr 2e-4 `
    --report-to tensorboard `
    --max-train-samples 3000 `
    --max-val-samples 300
```

**What each argument does:**

| Argument | Meaning | Recommendation |
|----------|---------|----------------|
| `--data-dirs` | Where training data lives | Both directories, space-separated |
| `--output-dir` | Where to save the trained LoRA | `models/geochat_multimodal_lora` |
| `--modalities` | Which modalities to train on | `optical sar` (or just `sar` to train SAR only) |
| `--model-path` | Base model | `mbzuai-oryx/GeoChat` |
| `--epochs` | Training passes | 3 is good for 3000 samples |
| `--lora-r` | LoRA rank | 16 is a good middle ground |
| `--lora-alpha` | LoRA scaling | Usually 2×r |
| `--batch-size` | Per-device batch | 1 (limited by 4GB VRAM) |
| `--grad-accum` | Gradient accumulation | 8 (effective batch = 8) |
| `--lr` | Learning rate | 2e-4 is standard for LoRA |
| `--report-to` | Logging | `tensorboard` (view with `tensorboard --logdir models/geochat_multimodal_lora/logs/tensorboard`) |
| `--max-train-samples` | Limit training set | 3000 (~1500 optical + 1500 SAR) |
| `--max-val-samples` | Limit validation set | 300 |

### Step 4.2: What happens during training

```
[1/3] Epoch 1
  - Loads GeoChat base model (~13GB, one-time)
  - Wraps with LoRA adapter
  - Iterates through 2700 training samples
  - Saves checkpoint to models/geochat_multimodal_lora/checkpoint-XXX/

[2/3] Epoch 2
  - Resumes from best checkpoint
  - Lower learning rate (cosine schedule)
  - Validates on 300 samples

[3/3] Epoch 3
  - Final pass
  - Saves final adapter to models/geochat_multimodal_lora/adapter/
```

### Step 4.3: Monitor with TensorBoard (in another terminal)

```powershell
tensorboard --logdir ./models/geochat_multimodal_lora/logs/tensorboard
```

Open http://localhost:6006 in browser to see:
- Training loss curve
- Validation loss
- Learning rate schedule
- Per-modality metrics

### Step 4.4: What if training is interrupted (Ctrl+C)?

The training script saves a checkpoint on KeyboardInterrupt. To resume:

```powershell
python -m m3_vlm.train_multimodal `
    --data-dirs ./data/bigearthnet_lora ./data/sar_lora `
    --output-dir ./models/geochat_multimodal_lora `
    --resume-from-checkpoint ./models/geochat_multimodal_lora/checkpoint-XXX `
    --epochs 3
```

### Step 4.5: What if GPU OOMs?

Reduce memory:
```powershell
python -m m3_vlm.train_multimodal `
    --data-dirs ./data/bigearthnet_lora ./data/sar_lora `
    --output-dir ./models/geochat_multimodal_lora `
    --lora-r 8 --lora-alpha 16 `  # smaller LoRA
    --batch-size 1 --grad-accum 4 `
    --max-train-samples 2000  # less data
```

If still OOMs, enable 4-bit:
```powershell
# Add --load-in-4bit (must edit train_multimodal.py to expose this)
```

---

## 🧪 PHASE 5: Test the Trained Model (5 minutes)

### Step 5.1: Test on a single SAR image

```powershell
python -m m3_vlm.test_inference `
    --image ./data/sar_lora/images/pc_sar_S1A_IW_GRDH_1SDV_20240115T060000.tif `
    --adapter ./models/geochat_multimodal_lora/adapter `
    --question "What is the land cover in this SAR image?"
```

Expected output:
```
============================================================
Image: .../pc_sar_S1A_IW_GRDH_1SDV_20240115T060000.tif
Modality: SAR
Adapter: ./models/geochat_multimodal_lora/adapter
============================================================

[1/3] Generating caption...
  Caption: SAR image showing urban area with strong backscatter from buildings.

[2/3] Answering question: 'What is the land cover in this SAR image?'
  Answer: The land cover is urban, characterized by high backscatter from buildings and structures.

[3/3] Grounding expression: 'the brightest area (strong backscatter)'
  BBox: [0.30, 0.25, 0.65, 0.70], Confidence: 0.85

Total time: 12.3s
```

### Step 5.2: Test on a directory

```powershell
python -m m3_vlm.test_inference `
    --dir ./data/sar_lora/images `
    --adapter ./models/geochat_multimodal_lora/adapter `
    --max-images 5
```

### Step 5.3: Compare with base model (no adapter)

To see the difference your LoRA made:
```powershell
# Without adapter (base GeoChat, bad at SAR)
python -m m3_vlm.test_inference --image ./data/sar_lora/images/test_sar.tif --question "What is this?"

# With adapter (your fine-tuned model, good at SAR)
python -m m3_vlm.test_inference --image ./data/sar_lora/images/test_sar.tif --adapter ./models/geochat_multimodal_lora/adapter --question "What is this?"
```

### Step 5.4: Compute quantitative metrics

```powershell
python -m m3_vlm.evaluate `
    --adapter ./models/geochat_multimodal_lora/adapter `
    --data-dir ./data/sar_lora `
    --output ./eval_results.json `
    --max-per-task 50
```

Output:
```
============================================================
EVALUATION SUMMARY
============================================================
Captioning: BLEU-4 = 0.3241 (50 samples)
VQA: F1 = 0.7862, EM = 0.4200 (50 samples)
Grounding: IoU = 0.5234, Acc@0.5 = 0.4800 (50 samples)
============================================================
```

---

## 🚀 PHASE 6: Deploy (5 minutes)

### Step 6.1: Start M3 service with your trained adapter

```powershell
# Set environment variable
$env:LORA_ADAPTER_PATH = "D:\VxKex\c\Sat-query-AI\satquery\models\geochat_multimodal_lora\adapter"
$env:GEOCHAT_MODEL = "mbzuai-oryx/GeoChat"

# Start M3 service (separate terminal, keep running)
python -m m3_vlm.service --port 8001
```

Expected output:
```
2026-08-28 [INFO] Initializing GeoChat service...
2026-08-28 [INFO] GeoChat service initialized successfully
2026-08-28 [INFO] Starting M3 VLM service on 0.0.0.0:8001
 * Running on http://0.0.0.0:8001
```

### Step 6.2: Test M3 directly

```powershell
# Health check
Invoke-RestMethod -Method Get -Uri "http://localhost:8001/health" | ConvertTo-Json

# Caption
$body = @{
    image_path = "D:\VxKex\c\Sat-query-AI\satquery\data\sar_lora\images\test_sar.tif"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://localhost:8001/caption" `
    -ContentType "application/json" `
    -Body $body | ConvertTo-Json
```

### Step 6.3: Start M6 API (separate terminal)

```powershell
cd D:\VxKex\c\Sat-query-AI\satquery
python -m m6_api.main
```

### Step 6.4: Query end-to-end via M6

```powershell
$query = @{
    query_text = "What is the land cover in this SAR image?"
    image_id = "pc_sar_S1A_IW_GRDH_1SDV_20240115T060000"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://localhost:8000/query" `
    -ContentType "application/json" `
    -Body $query | ConvertTo-Json
```

---

## 📊 DATASET SUMMARY (what to expect)

After Phase 2 + 3, your `data\` folder should look like:

```
data\
├── raw\bigearthnet\         # 2000-5000 patch directories (15-50GB)
│
├── bigearthnet_lora\        # Generated by acquire_data
│   ├── images\              # ~2000 PNG files
│   ├── captions.jsonl       # 2000 entries
│   ├── vqa.jsonl            # 2000 entries
│   └── referring.jsonl      # 2000 entries
│
├── sar_lora\                # Generated by acquire_sar
│   ├── images\              # 200-800 SAR pseudo-RGB PNGs
│   ├── captions.jsonl       # SAR-specific captions
│   ├── vqa.jsonl            # SAR-specific VQA
│   └── referring.jsonl      # SAR-specific referring
│
└── catalog.db               # Built by acquire_data catalog command
```

Total training samples for multimodal training:
- 2000 optical (caption + VQA + referring = 6000 total)
- 400 SAR (caption + VQA + referring = 1200 total)
- = ~7200 training samples

---

## 🎯 TROUBLESHOOTING

### "torch not found"
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### "rasterio not found"
```powershell
pip install rasterio
```

### "CUDA out of memory" during training
Reduce `--lora-r` to 8 and `--max-train-samples` to 1500.

### "No module named 'planetary_computer'"
```powershell
pip install planetary-computer pystac-client
```

### "Failed to load GeoChat from HuggingFace"
Your internet is down or HuggingFace is blocked. Try:
```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```
(only works if you already have the model cached)

### Training is very slow
- Check GPU utilization: `nvidia-smi` should show ~80% GPU usage
- If low, increase `--batch-size` (if VRAM allows)
- Reduce `--max-train-samples` to 2000 for faster iteration

### Adapter doesn't improve SAR performance
- You need more SAR data (try `--max-per-city 500` for 4 cities = 2000 SAR samples)
- Increase `--epochs` to 5
- Train on SAR only first: `--modalities sar` to make sure the model is learning SAR-specific features
- Use a higher LoRA rank: `--lora-r 32 --lora-alpha 64`

---

## ✅ CHECKLIST (print and tick off)

```
[ ] Python 3.9+ installed
[ ] All packages installed (torch, transformers, peft, rasterio, etc.)
[ ] setup_check.py passes
[ ] test_pipeline_e2e.py passes
[ ] BigEarthNet downloaded to data/raw/bigearthnet/
[ ] data/bigearthnet_lora/ has 2000+ PNGs and 3 JSONL files
[ ] data/sar_lora/ has 200+ SAR images and 3 JSONL files
[ ] Training started successfully
[ ] TensorBoard shows loss decreasing
[ ] test_inference.py gives sensible captions/VQA on SAR images
[ ] m3_vlm.service starts without error
[ ] m6_api.main starts and uses M3
[ ] End-to-end query works
```

---

## 📞 QUICK REFERENCE

| Command | What it does |
|---------|--------------|
| `python -m m3_vlm.setup_check` | Verify environment |
| `python -m m3_vlm.test_pipeline_e2e` | Verify code (no GPU) |
| `python -m m3_vlm.acquire_data train --source ./data/raw/bigearthnet --output ./data/bigearthnet_lora --max-samples 2000` | Convert BigEarthNet |
| `python -m m3_vlm.acquire_sar cities --cities delhi mumbai --output ./data/sar_lora --max-per-city 200` | Get SAR data |
| `python -m m3_vlm.train_multimodal --data-dirs ./data/bigearthnet_lora ./data/sar_lora --output-dir ./models/geochat_multimodal_lora --modalities optical sar --epochs 3` | Train |
| `python -m m3_vlm.test_inference --image ./data/sar_lora/images/test_sar.tif --adapter ./models/geochat_multimodal_lora/adapter` | Test |
| `python -m m3_vlm.evaluate --adapter ./models/geochat_multimodal_lora/adapter --data-dir ./data/sar_lora` | Evaluate |
| `python -m m3_vlm.service --port 8001` | Deploy M3 |
| `python -m m6_api.main` | Deploy M6 API |