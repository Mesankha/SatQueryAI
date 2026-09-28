"""
Colab Notebook Cells - for copy-paste into Google Colab.

This is a cleaner version of the colab workflow with each cell as a string
constant. Use this if you want to print and copy each cell individually.

To see all cells: python -m m3_vlm.colab_notebook
"""

CELL_1 = """\
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

CELL_2 = """\
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

CELL_3 = """\
# OPTION A: Clone from GitHub
# !git clone https://github.com/YOUR_USER/Sat-query-AI /content/satquery-src
# !cp -r /content/satquery-src/satquery/m3_vlm/* /content/satquery/m3_vlm/
# !cp -r /content/satquery-src/satquery/m0_catalog /content/satquery/
# !cp -r /content/satquery-src/satquery/m5_controller /content/satquery/
# !cp -r /content/satquery-src/satquery/m6_api /content/satquery/
# !cp -r /content/satquery-src/satquery/shared /content/satquery/
# !cp -r /content/satquery-src/satquery/schemas /content/satquery/
# !cp -r /content/satquery-src/satquery/config.yaml /content/satquery/

# OPTION B: Upload ZIP from local
# 1. Zip satquery folder: right-click -> Compressed
# 2. Upload to Google Drive at: /content/drive/MyDrive/satquery/satquery.zip
# 3. Then:
# !unzip -q -o /content/drive/MyDrive/satquery/satquery.zip -d /content/

import sys
sys.path.insert(0, '/content/satquery')
print("[OK] Code ready")
"""

CELL_4 = """\
import os, sys
os.chdir('/content/satquery')
sys.path.insert(0, '/content/satquery')

# Real Sentinel-1 SAR over Indian cities (NO BigEarthNet needed!)
!python -m m3_vlm.acquire_sar cities \\
    --cities delhi mumbai kerala_flood rajasthan guwahati \\
    --output ./data/sar_lora \\
    --max-per-city 100 \\
    --start-date 2024-01-01 \\
    --end-date 2024-12-31

# Real Sentinel-2 optical over Indian cities
!python -m m3_vlm.acquire_sar cities \\
    --cities delhi mumbai kerala_flood \\
    --output ./data/optical_lora \\
    --max-per-city 50 \\
    --start-date 2024-01-01 \\
    --end-date 2024-12-31

import os
sar_count = len([f for f in os.listdir('./data/sar_lora/images/') if f.endswith(('.tif', '.png'))])
opt_count = len([f for f in os.listdir('./data/optical_lora/images/') if f.endswith(('.tif', '.png'))])
print(f"SAR images: {sar_count}")
print(f"Optical images: {opt_count}")
"""

CELL_5 = """\
# Validate pipeline before real training (5 min)
!python -m m3_vlm.train_lora --create-dummy --data-dir ./data/dummy_test
!python -m m3_vlm.train_lora \\
    --data-dir ./data/dummy_test \\
    --output-dir ./models/dummy_test \\
    --max-train-samples 50 \\
    --max-val-samples 10 \\
    --epochs 1 \\
    --report-to none
print("[OK] Pipeline works!")
"""

CELL_6 = """\
# Real training: 4-6 hours on free T4 GPU
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
"""

CELL_7 = """\
# Run in a separate cell while training is going
%load_ext tensorboard
%tensorboard --logdir /content/satquery/models/geochat_india_sar/logs/tensorboard
"""

CELL_8 = """\
import os, sys
os.chdir('/content/satquery')
sys.path.insert(0, '/content/satquery')

sar_images = [f for f in os.listdir('./data/sar_lora/images/') if f.endswith(('.tif', '.png'))]
if sar_images:
    test_image = f'./data/sar_lora/images/{sar_images[0]}'
    !python -m m3_vlm.test_inference \\
        --image {test_image} \\
        --adapter ./models/geochat_india_sar/adapter \\
        --question "What is the land cover in this SAR image?"
else:
    print("No SAR images found.")
"""

CELL_9 = """\
!python -m m3_vlm.evaluate \\
    --adapter ./models/geochat_india_sar/adapter \\
    --data-dir ./data/sar_lora \\
    --output ./eval_results.json \\
    --max-per-task 30
"""

CELL_10 = """\
import os, datetime
adapter_path = '/content/satquery/models/geochat_india_sar/adapter'
if os.path.exists(adapter_path):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_path = f'/content/drive/MyDrive/satquery/geochat_india_sar_{timestamp}.zip'
    !cd /content/satquery/models/geochat_india_sar && zip -r {zip_path} adapter/
    !ls -lh {zip_path}
    print(f"Backup: {zip_path}")
"""

CELL_11 = """\
!pip install -q pyngrok flask flask-cors
import os
os.chdir('/content/satquery')
os.environ['LORA_ADAPTER_PATH'] = '/content/satquery/models/geochat_india_sar/adapter'

NGROK_AUTH_TOKEN = ''  # Get from https://dashboard.ngrok.com/

if NGROK_AUTH_TOKEN:
    from pyngrok import ngrok
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    !nohup python -m m3_vlm.service --port 8001 > /content/m3_service.log 2>&1 &
    import time; time.sleep(10)
    public_url = ngrok.connect(8001)
    print(f"Public URL: {public_url}")
"""


def main():
    """Print all cells."""
    cells = [
        ("Install dependencies", CELL_1),
        ("Mount Google Drive", CELL_2),
        ("Copy code to Drive", CELL_3),
        ("Get SAR + optical data (NO BigEarthNet!)", CELL_4),
        ("Sanity check (optional)", CELL_5),
        ("REAL TRAINING (4-6 hours)", CELL_6),
        ("TensorBoard (run in parallel)", CELL_7),
        ("Test trained model", CELL_8),
        ("Evaluate metrics", CELL_9),
        ("Backup to Drive", CELL_10),
        ("Deploy with ngrok (optional)", CELL_11),
    ]

    print("=" * 70)
    print("COLAB CELLS - Copy each into a separate Colab cell")
    print("=" * 70)
    print()
    for i, (title, content) in enumerate(cells, 1):
        print(f"\n{'#' * 70}")
        print(f"# CELL {i}: {title}")
        print(f"{'#' * 70}")
        print(content)
        print()


if __name__ == "__main__":
    main()