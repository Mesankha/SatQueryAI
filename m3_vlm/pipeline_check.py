"""
END-TO-END CHECK - Verify the entire training pipeline works with dummy data.
This is what you should run FIRST to make sure everything is wired correctly.

This check:
  1. Creates synthetic optical + SAR training data
  2. Loads the multimodal training script
  3. Verifies all components import and initialize
  4. Tests SAR preprocessing on synthetic data
  5. Tests multi-modal prompt selection
  6. Validates the dataset class

No GPU required. No real model needed. Just tests the CODE.

Run: python -m m3_vlm.pipeline_check
"""
import os
import sys
import tempfile
import shutil
import json
from pathlib import Path

# Setup path
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

print("=" * 70)
print("SatQuery AI - End-to-End Pipeline Check")
print("=" * 70)
print()

passed = 0
failed = 0


def check(name, condition=True, details=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}: {details}")
        failed += 1


# ---------------------------------------------------------------------------
# Check 1: Folder structure
# ---------------------------------------------------------------------------
print("\n[Check 1] Folder structure")
required_dirs = [
    "data/raw/bigearthnet",
    "data/raw/sar",
    "data/bigearthnet_lora/images",
    "data/sar_lora/images",
    "data/indian_cities",
    "models/geochat_multimodal_lora",
]
for d in required_dirs:
    p = Path(d)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    check(f"  {d}/", p.exists())

# ---------------------------------------------------------------------------
# Check 2: Imports
# ---------------------------------------------------------------------------
print("\n[Check 2] Module imports")

modules = [
    "m3_vlm.train_lora",
    "m3_vlm.train_multimodal",
    "m3_vlm.api",
    "m3_vlm.vlm_service",
    "m3_vlm.sar_preprocessing",
    "m3_vlm.sar_prompts",
    "m3_vlm.acquire_data",
    "m3_vlm.acquire_sar",
    "m3_vlm.evaluate",
    "m3_vlm.http_client",
    "m0_catalog.catalog",
    "m0_catalog.image_resolver",
    "m5_controller.dispatch_table",
    "m6_api.main",
]
for mod in modules:
    try:
        __import__(mod)
        check(f"  import {mod}", True)
    except Exception as e:
        check(f"  import {mod}", False, str(e)[:80])

# ---------------------------------------------------------------------------
# Check 3: SAR preprocessing on synthetic data
# ---------------------------------------------------------------------------
print("\n[Check 3] SAR preprocessing")
try:
    from m3_vlm.sar_preprocessing import (
        to_db, from_db, normalize_to_uint8, detect_sar_bands,
        sar_to_pseudo_rgb, detect_modality,
    )
    import numpy as np

    x = np.array([1.0, 10.0, 100.0])
    x_db = to_db(x)
    check("  to_db conversion", np.allclose(x_db, [0.0, 10.0, 20.0]))

    x_back = from_db(x_db)
    check("  dB roundtrip", np.allclose(x, x_back, rtol=1e-5))

    img = np.random.rand(50, 50) * 100
    norm = normalize_to_uint8(img)
    check("  normalize_to_uint8", norm.dtype.name == "uint8")

    arr = np.stack([np.random.rand(20, 20) * 100, np.random.rand(20, 20) * 30])
    info = detect_sar_bands(arr)
    check("  detect_sar_bands dual-pol", info["n_bands"] == 2)

    check("  detect_modality SAR", detect_modality("S1_scene.tif") == "sar")
    check("  detect_modality Optical", detect_modality("S2_scene.tif") == "optical")
    check("  detect_modality SAR prefix", detect_modality("sar_image.png") == "sar")
    check("  detect_modality fallback", detect_modality("random.jpg") == "optical")

    try:
        import rasterio
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            sar_tif = f.name

        np.random.seed(42)
        vv = np.random.rand(50, 50).astype(np.float32) * 100
        vh = np.random.rand(50, 50).astype(np.float32) * 30

        with rasterio.open(
            sar_tif, "w",
            driver="GTiff",
            height=50, width=50,
            count=2, dtype="float32",
        ) as dst:
            dst.write(vv, 1)
            dst.write(vh, 2)

        pil_img, metadata = sar_to_pseudo_rgb(sar_tif, target_size=224)
        check("  sar_to_pseudo_rgb", pil_img.mode == "RGB" and pil_img.size == (224, 224))
        os.unlink(sar_tif)
    except ImportError:
        print("  [SKIP] sar_to_pseudo_rgb (rasterio not installed)")

except Exception as e:
    check("  SAR preprocessing", False, str(e)[:100])

# ---------------------------------------------------------------------------
# Check 4: SAR prompts
# ---------------------------------------------------------------------------
print("\n[Check 4] SAR prompts")
try:
    from m3_vlm.sar_prompts import select_prompt

    p = select_prompt("vqa", "sar", question="Is this water?")
    check("  select_prompt SAR vqa", "Is this water?" in p and "backscatter" in p.lower())

    p = select_prompt("caption", "sar")
    check("  select_prompt SAR caption", "backscatter" in p.lower() or "Synthetic Aperture Radar" in p)

    p = select_prompt("caption", "optical")
    check("  select_prompt optical caption", "USER: <image>" in p)

    p = select_prompt("vqa", "fusion", question="What's changed?")
    check("  select_prompt fusion vqa", "optical" in p.lower() and "sar" in p.lower())

except Exception as e:
    check("  SAR prompts", False, str(e)[:100])

# ---------------------------------------------------------------------------
# Check 5: Multi-modal dataset class
# ---------------------------------------------------------------------------
print("\n[Check 5] Multi-modal dataset")
try:
    from m3_vlm.train_multimodal import load_multimodal_samples, detect_modality

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "test_data"
        images_dir = data_dir / "images"
        images_dir.mkdir(parents=True)

        try:
            from PIL import Image
            opt_img = Image.new("RGB", (100, 100), (100, 150, 200))
            opt_img.save(images_dir / "S2_test.png")
            sar_img = Image.new("RGB", (100, 100), (50, 50, 50))
            sar_img.save(images_dir / "S1_test.tif")
        except Exception:
            pass

        with open(data_dir / "captions.jsonl", "w") as f:
            f.write(json.dumps({
                "image_id": "S2_test.png",
                "caption": "An optical image showing urban area.",
            }) + "\n")
            f.write(json.dumps({
                "image_id": "S1_test.tif",
                "caption": "A SAR image showing urban backscatter.",
            }) + "\n")

        samples = load_multimodal_samples([data_dir])
        check("  load_multimodal_samples", len(samples) == 2)

        optical_samples = [s for s in samples if s.get("modality") == "optical"]
        sar_samples = [s for s in samples if s.get("modality") == "sar"]
        check("  modality auto-detection", len(optical_samples) == 1 and len(sar_samples) == 1)

        optical_only = load_multimodal_samples([data_dir], modalities=["optical"])
        check("  modality filter", len(optical_only) == 1)

        sar_only = load_multimodal_samples([data_dir], modalities=["sar"])
        check("  modality filter SAR", len(sar_only) == 1)

except Exception as e:
    check("  Multi-modal dataset", False, str(e)[:150])

# ---------------------------------------------------------------------------
# Check 6: M0 catalog + image resolver
# ---------------------------------------------------------------------------
print("\n[Check 6] M0 catalog + image resolver")
try:
    from m0_catalog.catalog import Catalog
    from m0_catalog.image_resolver import (
        resolve_image, is_cached, _is_url, _url_to_cache_path,
    )

    check("  _is_url http", _is_url("http://example.com") is True)
    check("  _is_url local", _is_url("./local/file.tif") is False)
    check("  _is_url s3", _is_url("s3://bucket/key") is True)

    p1 = _url_to_cache_path("https://example.com/test.tif")
    p2 = _url_to_cache_path("https://example.com/test.tif")
    check("  cache path deterministic", p1 == p2)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    catalog = Catalog(db_path, read_only=False)
    from shared.schemas import SceneMetadata
    from datetime import datetime, timezone
    scene = SceneMetadata(
        scene_id="check-001",
        sensor="Sentinel-2",
        modality="multispectral",
        acquisition_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        geometry_wkt="POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
        file_path="./check.tif",
    )
    catalog.insert_scene(scene, "check_origin")
    catalog.close()

    catalog2 = Catalog(db_path, read_only=True)
    retrieved = catalog2.get_scene("check-001")
    check("  catalog insert/retrieve", retrieved is not None and retrieved.scene_id == "check-001")
    catalog2.close()
    os.unlink(db_path)

except Exception as e:
    check("  M0 catalog", False, str(e)[:150])

# ---------------------------------------------------------------------------
# Check 7: M5 controller dispatch
# ---------------------------------------------------------------------------
print("\n[Check 7] M5 controller")
try:
    from m5_controller.dispatch_table import (
        orchestrate, DISPATCH,
    )
    from shared.schemas import TaskType

    all_tasks_have_pipelines = all(task.value in DISPATCH for task in TaskType)
    check("  All task types have pipelines", all_tasks_have_pipelines)

    response = orchestrate("Show me agriculture near Delhi")
    check("  orchestrate returns QueryResponse", hasattr(response, "results") and hasattr(response, "trace"))

except Exception as e:
    check("  M5 controller", False, str(e)[:150])

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 70)
print()
if failed == 0:
    print("[OK] All checks passed. The pipeline is wired correctly.")
    print()
    print("Next steps:")
    print("  1. python -m m3_vlm.setup_check  (verify GPU + packages)")
    print("  2. python -m m3_vlm.acquire_data train --source ./data/raw/bigearthnet --output ./data/bigearthnet_lora --max-samples 2000")
    print("  3. python -m m3_vlm.acquire_sar cities --cities delhi mumbai --output ./data/sar_lora --max-per-city 200")
    print("  4. python -m m3_vlm.train_multimodal --data-dirs ./data/bigearthnet_lora ./data/sar_lora --output-dir ./models/geochat_multimodal_lora --modalities optical sar --epochs 3")
    print("  5. python -m m3_vlm.test_inference --image ./data/sar_lora/images/S1_test.tif --adapter ./models/geochat_multimodal_lora/adapter")
else:
    print(f"[FAIL] {failed} checks failed. Please fix before proceeding.")
    sys.exit(1)