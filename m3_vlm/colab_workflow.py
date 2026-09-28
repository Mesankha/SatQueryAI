"""
Google Colab Training Script for SatQuery GeoChat Fine-Tuning.

This file provides ready-to-paste Colab cells for fine-tuning GeoChat on
real SAR + optical satellite data from Microsoft Planetary Computer.

Key insight: You do NOT need BigEarthNet. Planetary Computer provides
real Sentinel-1 SAR + Sentinel-2 optical data for any AOI worldwide,
including all Indian cities - for free, with no authentication.

Usage: python -m m3_vlm.colab_workflow (prints all cells)
Or: read this file and copy each cell block into Colab
"""

CELL_1_INSTALL = """\
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers peft accelerate bitsandbytes datasets
!pip install -q rasterio planetary-computer pystac-client scipy
!pip install -q fastapi uvicorn httpx pydantic pydantic-settings

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
"""


CELL_2_MOUNT = """\
from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT_DIR = '/content/drive/MyDrive/satquery'
os.makedirs(f'{PROJECT_DIR}/data/sar_lora/images', exist_ok=True)
os.makedirs(f'{PROJECT_DIR}/data/optical_lora/images', exist_ok=True)
os.makedirs(f'{PROJECT_DIR}/models', exist_ok=True)
os.makedirs(f'{PROJECT_DIR}/m3_vlm', exist_ok=True)

!ln -sf {PROJECT_DIR} /content/satquery
print(f"Project at: {PROJECT_DIR}")
"""


CELL_3_COPY_CODE = """\
# OPTION A: Clone from GitHub (if you pushed your code)
# !git clone https://github.com/YOUR_USER/Sat-query-AI /content/satquery-src
# !cp -r /content/satquery-src/satquery/m3_vlm/* /content/satquery/m3_vlm/
# !cp -r /content/satquery-src/satquery/m0_catalog /content/satquery/
# !cp -r /content/satquery-src/satquery/m5_controller /content/satquery/
# !cp -r /content/satquery-src/satquery/m6_api /content/satquery/
# !cp -r /content/satquery-src/satquery/shared /content/satquery/
# !cp -r /content/satquery-src/satquery/schemas /content/satquery/
# !cp -r /content/satquery-src/satquery/config.yaml /content/satquery/

# OPTION B: Upload as ZIP from local
# 1. Zip satquery folder on Windows: right-click -> Compressed
# 2. Upload ZIP to Google Drive at: /content/drive/MyDrive/satquery/satquery.zip
# 3. Then run:
# !unzip -q -o /content/drive/MyDrive/satquery/satquery.zip -d /content/
# !mv /content/satquery/* /content/satquery/ 2>/dev/null || true

import sys
sys.path.insert(0, '/content/satquery')
print("[OK] Code ready")
"""


CELL_4_GET_DATA = """\
import os, sys
os.chdir('/content/satquery')
sys.path.insert(0, '/content/satquery')

# KEY STEP: Real Sentinel-1 SAR over Indian cities
# - 5 cities
# - 100 scenes per city
# - 2024 full year
# - NO authentication, NO download limit, completely FREE

!python -m m3_vlm.acquire_sar cities \\
    --cities delhi mumbai kerala_flood rajasthan guwahati \\
    --output ./data/sar_lora \\
    --max-per-city 100 \\
    --start-date 2024-01-01 \\
    --end-date 2024-12-31

# Also get optical (Sentinel-2) data for paired/fusion training
!python -m m3_vlm.acquire_sar cities \\
    --cities delhi mumbai kerala_flood \\
    --output ./data/optical_lora \\
    --max-per-city 50 \\
    --start-date 2024-01-01 \\
    --end-date 2024-12-31

# Verify
import os
sar_count = len([f for f in os.listdir('./data/sar_lora/images/') if f.endswith(('.tif', '.png'))])
opt_count = len([f for f in os.listdir('./data/optical_lora/images/') if f.endswith(('.tif', '.png'))])
print(f"SAR images: {sar_count}")
print(f"Optical images: {opt_count}")
"""


CELL_5_SANITY_CHECK = """\
# Run this BEFORE real training to validate the pipeline
# Takes 3-5 min, uses no real data

!python -m m3_vlm.train_lora --create-dummy --data-dir ./data/dummy_test
!python -m m3_vlm.train_lora \\
    --data-dir ./data/dummy_test \\
    --output-dir ./models/dummy_test \\
    --max-train-samples 50 \\
    --max-val-samples 10 \\
    --epochs 1 \\
    --report-to none

print("[OK] Pipeline works! Proceed to real training.")
"""


CELL_6_TRAIN = """\
# Main training command
# T4 GPU (Colab free tier): ~4-6 hours for 3 epochs
# If you have Colab Pro/A100: ~30-60 min

!python -m m3_vlm.train_multimodal \\
    --data-dirs ./data/sar_lora ./data/optical_lora \\
    --output-dir ./models/geochat_india_sar \\
    --modalities sar optical \\
    --model-path mbzuai-oryx/GeoChat \\
    --epochs 3 \\
    --lora-r 16 --lora-alpha 32 \\
    --batch-size 1 --grad-accum 8 \\
    --lr 2e-4 \\
    --report-to tensorboard \\
    --max-train-samples 3000 \\
    --max-val-samples 300

# Trained adapter will be at:
# /content/drive/MyDrive/satquery/models/geochat_india_sar/adapter/
"""


CELL_7_TENSORBOARD = """\
%load_ext tensorboard
%tensorboard --logdir /content/satquery/models/geochat_india_sar/logs/tensorboard
"""


CELL_8_TEST = """\
import os, sys
os.chdir('/content/satquery')
sys.path.insert(0, '/content/satquery')

# Test on a SAR image
sar_images = [f for f in os.listdir('./data/sar_lora/images/') if f.endswith(('.tif', '.png'))]
if sar_images:
    test_image = f'./data/sar_lora/images/{sar_images[0]}'
    !python -m m3_vlm.test_inference \\
        --image {test_image} \\
        --adapter ./models/geochat_india_sar/adapter \\
        --question "What is the land cover in this SAR image?"
else:
    print("No SAR images found. Did Step 4 complete?")
"""


CELL_9_EVALUATE = """\
!python -m m3_vlm.evaluate \\
    --adapter ./models/geochat_india_sar/adapter \\
    --data-dir ./data/sar_lora \\
    --output ./eval_results.json \\
    --max-per-task 30
"""


CELL_10_BACKUP = """\
import os, datetime
adapter_path = '/content/satquery/models/geochat_india_sar/adapter'
if os.path.exists(adapter_path):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_path = f'/content/drive/MyDrive/satquery/geochat_india_sar_{timestamp}.zip'
    !cd /content/satquery/models/geochat_india_sar && zip -r {zip_path} adapter/
    !ls -lh {zip_path}
    print(f"[OK] Backup: {zip_path}")
"""


CELL_11_DEPLOY = """\
!pip install -q pyngrok flask flask-cors
import os
os.chdir('/content/satquery')
os.environ['LORA_ADAPTER_PATH'] = '/content/satquery/models/geochat_india_sar/adapter'

# Get token from https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN = ''  # <-- PASTE YOUR TOKEN HERE

if NGROK_AUTH_TOKEN:
    from pyngrok import ngrok
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    !nohup python -m m3_vlm.service --port 8001 > /content/m3_service.log 2>&1 &
    import time; time.sleep(10)
    public_url = ngrok.connect(8001)
    print(f"Public URL: {public_url}")
    print(f"Test: curl {public_url}/health")
"""


def print_workflow():
    """Print all cells in order with clear formatting."""
    cells = [
        ("CELL 1 (Install dependencies, 3-5 min)", CELL_1_INSTALL),
        ("CELL 2 (Mount Google Drive)", CELL_2_MOUNT),
        ("CELL 3 (Copy code to Colab)", CELL_3_COPY_CODE),
        ("CELL 4 (Get real SAR + optical data - NO BigEarthNet needed!)", CELL_4_GET_DATA),
        ("CELL 5 (Sanity check, optional but recommended)", CELL_5_SANITY_CHECK),
        ("CELL 6 (Real training, 4-6 hours)", CELL_6_TRAIN),
        ("CELL 7 (TensorBoard, run in parallel cell)", CELL_7_TENSORBOARD),
        ("CELL 8 (Test the trained model)", CELL_8_TEST),
        ("CELL 9 (Quantitative evaluation)", CELL_9_EVALUATE),
        ("CELL 10 (Backup to Drive)", CELL_10_BACKUP),
        ("CELL 11 (Deploy with ngrok, optional)", CELL_11_DEPLOY),
    ]

    print("=" * 70)
    print("COLAB WORKFLOW - Copy each cell below into a separate Colab cell")
    print("=" * 70)
    print()
    print("Total time: 4-6 hours (mostly training)")
    print("Final adapter size: ~200MB")
    print("Storage needed: ~5GB for data + 13GB for base model")
    print()
    print("=" * 70)
    print()

    for i, (title, content) in enumerate(cells, 1):
        print(f"\n{'#' * 70}")
        print(f"# {title}")
        print(f"{'#' * 70}")
        print(content)
        print()


if __name__ == "__main__":
    print_workflow()