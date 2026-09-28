# SatQuery AI — Data Strategy Guide

## TL;DR (30-second version)

| What you need | Where to get it | Size | When |
|---------------|-----------------|------|------|
| **LoRA training data** | BigEarthNet download | ~500MB (sample) to 68GB (full) | One-time, on 3050 laptop |
| **Demo query data** | Microsoft Planetary Computer API | **0 bytes** (fetched on-demand) | Any time, automatic |
| **M0 catalog** | Mix of both | ~10MB SQLite | Built once, queried forever |

**You do NOT need to keep all the satellite data on your disk.** Only training samples. Demo data is fetched on-demand from a free public API.

---

## The Complete Picture

```
┌─────────────────────────────────────────────────────────────────┐
│  YOUR MACHINE (one-time setup + ongoing queries)               │
│                                                                 │
│  ┌──────────────────┐  ┌─────────────────┐  ┌──────────────┐   │
│  │  BigEarthNet      │  │  data/           │  │  models/     │   │
│  │  Download (once)  │  │  catalog.db      │  │  geochat_lora │   │
│  │                   │  │  (SQLite, ~10MB) │  │  /adapter    │   │
│  │  • 5000 PNGs      │  │                  │  │  (~200MB)    │   │
│  │  • 500MB total    │  │  scene_id, sensor │  │              │   │
│  │  • captions.jsonl │  │  date, file_path  │  │  Trained     │   │
│  │  • vqa.jsonl      │  │                  │  │  GeoChat     │   │
│  │  • referring.jsonl│  │  file_path can be │  │  adapter     │   │
│  │                   │  │  • local file     │  │              │   │
│  │  Used for:        │  │  • URL (lazy load)│  │  Used for:   │   │
│  │  Fine-tuning M3   │  │                  │  │  M3 VLM      │   │
│  │  GeoChat LoRA     │  │  Used for:        │  │  inference   │   │
│  └──────────────────┘  │  Catalog queries  │  └──────────────┘   │
│           │              │  for all demos     │         │        │
│           │              └─────────────────┘         │        │
│           ▼                       │                  ▼        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           m3_vlm.train_lora.py                              │  │
│  │  Reads BigEarthNet PNGs, fine-tunes GeoChat, saves        │  │
│  │  LoRA adapter to models/geochat_lora/                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
              │ (training is one-time, 4-8h on RTX 3050)
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  EXTERNAL: Microsoft Planetary Computer (FREE, no auth)          │
│  https://planetarycomputer.microsoft.com/api/stac/v1/           │
│                                                                  │
│  • Sentinel-2 L2A (13 bands, 10-60m resolution)                 │
│  • Sentinel-1 GRD (SAR, C-band)                                 │
│  • Landsat 8/9 (multispectral + thermal)                        │
│  • MODIS (daily global coverage)                                │
│  • Cloud-Optimized GeoTIFFs (download in chunks)                 │
│                                                                  │
│  Used for:  Demo queries like "show me agriculture near Delhi"  │
│  How:       M0 catalog stores signed URL → m0_catalog/          │
│             image_resolver.py downloads to ./data/image_cache/   │
│             on first use → M3 GeoChat reads the cached file      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Decision: Why Hybrid Storage?

**Option A: Keep everything local (1TB+ for full Sentinel-2)**
- ❌ Wastes disk
- ❌ Data becomes stale (no new imagery)
- ❌ You have to manage updates yourself
- ❌ Downloading Sentinel-2 for all of India is ~500GB

**Option B: Fetch everything remotely (always query API)**
- ❌ Slow (each query needs network round-trip)
- ❌ API rate limits
- ❌ Can't work offline

**Option C: HYBRID (recommended) ✓**
- ✅ Only **metadata** stored locally (fast catalog queries)
- ✅ **Image files** cached locally on first use (fast inference)
- ✅ Signed URLs allow re-fetching newer versions
- ✅ Cache auto-expires old images (configurable)

This is the same pattern as `apt` or `pip` — package metadata is local, package files cached.

---

## Practical Workflows

### Workflow 1: First-Time Setup (Do This Once)

#### Step 1a: Install Python packages
```bash
pip install planetary-computer pystac-client rasterio
```

#### Step 1b: Download BigEarthNet for training (on 3050 laptop)
```bash
# Go to http://bigearth.net/ and register for free
# Download the "BigEarthNet-MM" RGB variant (smaller, ~15GB)
# Extract to D:/data/raw/bigearthnet/
```

BigEarthNet structure:
```
data/raw/bigearthnet/
    S2A_MSIL2A_20170613T101031_8_36/
        S2A_MSIL2A_20170613T101031_8_36_B02.tif   (blue band)
        S2A_MSIL2A_20170613T101031_8_36_B03.tif   (green)
        S2A_MSIL2A_20170613T101031_8_36_B04.tif   (red)
        ... (12 bands total)
        S2A_MSIL2A_20170613T101031_8_36_labels_metadata.json
    S2A_MSIL2A_20170613T101031_8_37/
    ... (~590,000 patch directories)
```

#### Step 1c: Convert to LoRA training format
```bash
cd D:\VxKex\c\Sat-query-AI\satquery

python -m m3_vlm.acquire_data train \
    --source ./data/raw/bigearthnet \
    --output ./data/bigearthnet_lora \
    --max-samples 2000
```

This produces:
```
data/bigearthnet_lora/
    images/                  ← 2000 PNG files (~500MB)
    captions.jsonl           ← {"image_id": "...", "caption": "..."}
    vqa.jsonl                ← {"image_id": "...", "question": "...", "answer": "..."}
    referring.jsonl          ← {"image_id": "...", "expression": "...", "bbox": [...]}
```

#### Step 1d: Fine-tune GeoChat (4-8 hours on RTX 3050)
```bash
python -m m3_vlm.train_lora \
    --data-dir ./data/bigearthnet_lora \
    --output-dir ./models/geochat_lora \
    --epochs 3 \
    --lora-r 16 --lora-alpha 32 \
    --report-to tensorboard
```

#### Step 1e: Populate M0 catalog for demo (on either laptop)
```bash
# Query Planetary Computer for major Indian cities
python -m m3_vlm.acquire_data cities \
    --cities delhi mumbai bangalore chennai kolkata \
    --output ./data/indian_cities \
    --max-per-city 50

# Build SQLite catalog from the downloaded metadata
python -m m3_vlm.acquire_data catalog \
    --data-root ./data/indian_cities \
    --db ./data/catalog.db
```

Now `catalog.db` contains 250 scenes (5 cities × 50 each) with signed URLs to actual Sentinel-2 data.

**That's it for setup.** No further downloads needed.

---

### Workflow 2: Demo Day (Any Time After Setup)

When a user makes a query like *"show me agriculture near Delhi from June 2024"*:

1. **M1 Parser** extracts: location=Delhi, date=June 2024, object=agriculture
2. **M5 Controller** queries **M0 catalog** (local SQLite, instant)
3. **M0 catalog** returns matching scenes with `file_path` = signed URL
4. **M3 GeoChat** needs the actual image:
   - `m0_catalog/image_resolver.py` checks `./data/image_cache/`
   - If not cached: downloads from URL to cache (one-time, ~50MB per scene)
   - If cached: returns immediately
5. **GeoChat** runs VQA/caption/grounding
6. **Result** returned to user

The cache grows over time but:
- Bounded by disk space (configurable eviction)
- Automatic cleanup of old images (`clear_cache(max_age_days=30)`)
- Subsequent queries for same image = instant

---

### Workflow 3: Adding New Cities/Regions

When you want to add a new region (e.g., Hyderabad):

```bash
python -m m3_vlm.acquire_data cities \
    --cities hyderabad pune \
    --output ./data/indian_cities \
    --max-per-city 30

# Re-build catalog
python -m m3_vlm.acquire_data catalog \
    --data-root ./data/indian_cities \
    --db ./data/catalog.db
```

**No re-training needed.** New scenes are added to M0 catalog, GeoChat works on them.

---

### Workflow 4: Updating to Newer Imagery

Planetary Computer is updated by ESA/USGS continuously. To get newer Sentinel-2:

```bash
python -m m3_vlm.acquire_data cities \
    --cities delhi mumbai bangalore chennai kolkata hyderabad pune \
    --output ./data/indian_cities \
    --start-date 2025-01-01 \
    --end-date 2026-12-31 \
    --max-per-city 30
```

New scenes added to catalog. Old images still cached (don't re-download).

---

## Disk Budget

| Component | Size on RTX 3050 laptop | Notes |
|-----------|--------------------------|-------|
| Python venv | ~8GB | torch+transformers+peft |
| BigEarthNet (full) | 68GB | Optional, only if you want full training |
| BigEarthNet sample (5K) | ~500MB | Recommended for 1-day training |
| Catalog DB | 10-50MB | SQLite with metadata + signed URLs |
| Image cache (100 images) | ~500MB | Lazy download, auto-cleanup |
| LoRA adapter | ~200MB | Output of training |
| LoRA training data | ~500MB | JSONL files + 5K PNGs |
| **Total** | **~10-12GB** | For full demo capability |

vs. keeping all Sentinel-2 for India: **~500GB**

---

## What to Tell Stakeholders/Demo Audience

> "SatQuery uses a hybrid data architecture. Training data is sourced from
> BigEarthNet, a research dataset of 590,000 labeled Sentinel-2 patches
> published by TU Berlin. For live queries, the system connects to
> Microsoft's Planetary Computer — a free public API that provides
> real-time access to the entire ESA Sentinel-2 archive over India and
> the world. The M0 catalog stores scene metadata locally for instant
> queries; the actual image files are cached on first use and refreshed
> automatically as new satellite passes become available."

This is **production-grade** architecture used by companies like Planet, Maxar, and Microsoft AI for Earth.

---

## Summary: Three Answers to Your Questions

1. **"Which data for what?"**
   - **Training**: BigEarthNet (one-time download, ~500MB sample)
   - **Demo**: Microsoft Planetary Computer (real-time, free, no download)

2. **"Local DB or pipeline from elsewhere?"**
   - **Metadata** (catalog) → local SQLite DB (instant queries)
   - **Images** → signed URL stored in DB, lazy-downloaded to cache on use
   - **Result**: best of both worlds — fast queries, fresh data, bounded disk usage

3. **"How to use the data?"**
   - **Training** (one-time): `acquire_data train` → `train_lora` → saves adapter
   - **Demo** (always works): `acquire_data cities` → populates catalog → user queries
   - **Ongoing**: just query the API; cache fills up automatically

---

## Quick Reference: All Commands

```bash
# TRAINING (one-time, on 3050 laptop)
pip install planetary-computer pystac-client rasterio
python -m m3_vlm.acquire_data train --source ./data/raw/bigearthnet --output ./data/bigearthnet_lora --max-samples 2000
python -m m3_vlm.train_lora --data-dir ./data/bigearthnet_lora --output-dir ./models/geochat_lora

# DEMO SETUP (one-time, on either laptop)
python -m m3_vlm.acquire_data cities --cities delhi mumbai bangalore --output ./data/indian_cities --max-per-city 50
python -m m3_vlm.acquire_data catalog --data-root ./data/indian_cities --db ./data/catalog.db

# DEMO USAGE (always, automatic)
# Just start the API and make queries:
python -m m6_api.main
# Then: POST /query with {"query_text": "show me agriculture near Delhi from June 2024"}

# ADD NEW CITIES (any time)
python -m m3_vlm.acquire_data cities --cities chennai kolkata --output ./data/indian_cities
python -m m3_vlm.acquire_data catalog --data-root ./data/indian_cities --db ./data/catalog.db

# CACHE MAINTENANCE
python -c "from m0_catalog.image_resolver import clear_cache, get_cache_size_mb; print(f'Cache: {get_cache_size_mb():.1f}MB'); clear_cache(30)"
```

That's the complete data strategy. You now have a **clear path** that:
- Costs **0 dollars** (no API keys, no paid services)
- Uses **<12GB disk** (vs 500GB for naive approach)
- Gives you **real satellite data** (not synthetic)
- Works for **any Indian city** (or any AOI worldwide)
- Updates **automatically** (no manual refresh needed)