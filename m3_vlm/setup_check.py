"""
SETUP SCRIPT - Run this ONCE before any training/eval.

This script:
  1. Verifies Python version and packages
  2. Checks if model weights are cached
  3. Creates necessary directories
  4. Provides an interactive walkthrough of next steps

Run: python -m m3_vlm.setup_check
"""
import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path


REQUIRED_PACKAGES = {
    # Core
    "torch": "torch>=2.0",
    "transformers": "transformers>=4.30",
    "peft": "peft>=0.5",
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "datasets": "datasets",

    # Geospatial
    "rasterio": "rasterio",
    "shapely": "shapely",

    # Data acquisition
    "planetary_computer": "planetary-computer",
    "pystac_client": "pystac-client",
    "scipy": "scipy",
    "PIL": "Pillow",

    # API
    "fastapi": "fastapi",
    "uvicorn": "uvicorn[standard]",
    "httpx": "httpx",

    # Utils
    "pydantic": "pydantic>=2.0",
    "pydantic_settings": "pydantic-settings",
}


def check_python_version():
    """Check Python version is 3.9+."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"[FAIL] Python {version.major}.{version.minor} too old. Need 3.9+")
        return False
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_package(pkg_name, pip_name=None):
    """Check if a package is importable."""
    pip_name = pip_name or pkg_name
    try:
        importlib.import_module(pkg_name)
        return True, None
    except ImportError as e:
        return False, str(e)


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[OK] GPU: {gpu_name} ({vram:.1f} GB)")
            return True
        else:
            print("[WARN] No CUDA GPU detected")
            return False
    except ImportError:
        print("[WARN] PyTorch not installed, can't check GPU")
        return False


def check_cuda_compatibility():
    """Warn if RTX 5070 (sm_120) with old PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            # RTX 5070 has compute capability 12.0 (sm_120)
            # PyTorch 2.4+ supports it. Older versions won't work.
            torch_ver = tuple(int(x) for x in torch.__version__.split(".")[:2])
            if cap[0] >= 12 and torch_ver < (2, 5):
                print(f"[WARN] GPU sm_{cap[0]}{cap[1]} detected but PyTorch {torch.__version__} may not support it.")
                print(f"       Recommended: PyTorch 2.5+ with CUDA 12.4+")
                print(f"       Install: pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu124")
                return False
        return True
    except Exception:
        return True


def check_hf_cache():
    """Check if model weights are cached locally."""
    cache_dir = Path.home() / ".cache" / "huggingface"
    if not cache_dir.exists():
        return False
    # Look for geochat
    for p in cache_dir.rglob("*.bin"):
        if "geochat" in str(p).lower():
            print(f"[OK] GeoChat weights found: {p.parent}")
            return True
    return False


def check_folders():
    """Verify all required folders exist."""
    required = [
        "data/raw/bigearthnet",
        "data/raw/sar",
        "data/bigearthnet_lora/images",
        "data/sar_lora/images",
        "data/indian_cities",
        "models/geochat_multimodal_lora",
    ]
    all_ok = True
    for rel_path in required:
        p = Path(rel_path)
        if p.exists():
            print(f"[OK] {rel_path}/")
        else:
            print(f"[--] {rel_path}/  (will be created)")
            p.mkdir(parents=True, exist_ok=True)
            all_ok = False
    return True


def print_workflow():
    """Print the recommended workflow."""
    print()
    print("=" * 70)
    print("RECOMMENDED WORKFLOW")
    print("=" * 70)
    print()
    print("STEP 1: Get optical training data (BigEarthNet)")
    print("  1a. Download BigEarthNet-MM from http://bigearth.net/")
    print("      Extract to: data/raw/bigearthnet/")
    print("      (Each patch = a directory with 12 .tif band files + 1 .json)")
    print()
    print("  1b. Convert to LoRA training format:")
    print("      python -m m3_vlm.acquire_data train \\")
    print("          --source ./data/raw/bigearthnet \\")
    print("          --output ./data/bigearthnet_lora \\")
    print("          --max-samples 2000")
    print()
    print("STEP 2: Get SAR training data (Planetary Computer, free, no auth)")
    print("      python -m m3_vlm.acquire_sar cities \\")
    print("          --cities delhi mumbai kerala_flood rajasthan \\")
    print("          --output ./data/sar_lora \\")
    print("          --max-per-city 200")
    print()
    print("STEP 3: Fine-tune GeoChat on both modalities")
    print("      python -m m3_vlm.train_multimodal \\")
    print("          --data-dirs ./data/bigearthnet_lora ./data/sar_lora \\")
    print("          --output-dir ./models/geochat_multimodal_lora \\")
    print("          --modalities optical sar \\")
    print("          --epochs 3 \\")
    print("          --lora-r 16 --lora-alpha 32")
    print()
    print("STEP 4: Evaluate the trained model")
    print("      python -m m3_vlm.evaluate \\")
    print("          --adapter ./models/geochat_multimodal_lora/adapter \\")
    print("          --data-dir ./data/sar_lora \\")
    print("          --output ./eval_results.json")
    print()
    print("STEP 5: Start M3 service with trained adapter")
    print("      set LORA_ADAPTER_PATH=./models/geochat_multimodal_lora/adapter")
    print("      python -m m3_vlm.service --port 8001")
    print()
    print("STEP 6: Start M6 API (uses M3 automatically)")
    print("      python -m m6_api.main")
    print()
    print("STEP 7: Test from PowerShell")
    print('      Invoke-RestMethod -Method Post -Uri "http://localhost:8000/query" \\')
    print('          -ContentType "application/json" \\')
    print('          -Body \'{"query_text": "What is the land cover in this image?", "image_id": "user_upload_001"}\'')
    print()
    print("=" * 70)


def main():
    print("=" * 70)
    print("SatQuery AI - Setup Check")
    print("=" * 70)
    print()

    # Python
    print("--- Python ---")
    check_python_version()
    print()

    # Packages
    print("--- Required packages ---")
    missing = []
    for pkg, spec in REQUIRED_PACKAGES.items():
        ok, err = check_package(pkg)
        if ok:
            print(f"[OK] {spec}")
        else:
            print(f"[--] {spec}  (not installed: {err})")
            missing.append(spec)
    print()
    if missing:
        print("To install missing packages:")
        print(f"  pip install {' '.join(missing)}")
        print()

    # GPU
    print("--- Hardware ---")
    check_gpu()
    check_cuda_compatibility()
    print()

    # Cache
    print("--- Model cache ---")
    if not check_hf_cache():
        print("[--] GeoChat not yet downloaded. Will be auto-downloaded on first run (~13GB)")
    print()

    # Folders
    print("--- Folder structure ---")
    check_folders()
    print()

    # Workflow
    print_workflow()


if __name__ == "__main__":
    main()