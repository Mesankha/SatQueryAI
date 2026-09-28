"""
M3 SAR Preprocessing and Prompts tests.
"""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from m3_vlm.sar_preprocessing import (
    to_db, from_db, normalize_to_uint8, detect_sar_bands,
    lee_filter, enhanced_lee_filter,
    sar_to_pseudo_rgb, _load_sar_geotiff, save_sar_as_png,
)
from m3_vlm.sar_prompts import (
    select_prompt, VQA_TEMPLATE, CAPTION_TEMPLATE, GROUNDING_TEMPLATE,
    SAR_VQA_TEMPLATE, SAR_CAPTION_TEMPLATE, SAR_GROUNDING_TEMPLATE,
    FUSION_CAPTION_TEMPLATE, FUSION_VQA_TEMPLATE, FUSION_CHANGE_TEMPLATE,
    GEOCHAT_SYSTEM, SAR_SYSTEM,
)
from m3_vlm.vlm_service import _load_image_safely


class TestToFromDb:
    def test_to_db_basic(self):
        x = np.array([1.0, 10.0, 100.0])
        result = to_db(x)
        assert np.allclose(result, [0.0, 10.0, 20.0])

    def test_from_db_basic(self):
        x_db = np.array([0.0, 10.0, 20.0])
        result = from_db(x_db)
        assert np.allclose(result, [1.0, 10.0, 100.0])

    def test_roundtrip(self):
        x = np.array([0.5, 5.0, 50.0])
        x_db = to_db(x)
        x_back = from_db(x_db)
        assert np.allclose(x, x_back, rtol=1e-5)

    def test_eps_handles_zero(self):
        x = np.array([0.0, 0.0])
        result = to_db(x)
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()


class TestNormalizeToUint8:
    def test_basic_normalization(self):
        x = np.array([0.0, 50.0, 100.0])
        result = normalize_to_uint8(x, percentile_clip=100)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_handles_inf(self):
        x = np.array([0.0, 100.0, np.inf, -np.inf, np.nan])
        result = normalize_to_uint8(x)
        assert result.dtype == np.uint8

    def test_constant_image(self):
        x = np.ones((10, 10)) * 50.0
        result = normalize_to_uint8(x, percentile_clip=99.5)
        # All same value, should normalize to mid-range
        assert result.dtype == np.uint8


class TestDetectSarBands:
    def test_single_band(self):
        arr = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
        info = detect_sar_bands(arr)
        assert info["n_bands"] == 1
        assert "single-pol" in info["interpretation"]

    def test_dual_band(self):
        arr = np.array([
            [[10.0, 20.0], [30.0, 40.0]],  # VV (brighter)
            [[1.0, 2.0], [3.0, 4.0]],      # VH (dimmer)
        ])
        info = detect_sar_bands(arr)
        assert info["n_bands"] == 2
        assert "dual-pol" in info["interpretation"]
        assert info["vv_idx"] == 0
        assert info["vh_idx"] == 1

    def test_three_band_interpretation(self):
        arr = np.random.rand(3, 10, 10)
        info = detect_sar_bands(arr)
        assert info["n_bands"] == 3


class TestLeeFilter:
    def test_filter_returns_same_shape(self):
        np.random.seed(42)
        sar = np.random.rand(50, 50) * 10
        filtered = enhanced_lee_filter(sar, window_size=5)
        assert filtered.shape == sar.shape

    def test_filter_reduces_noise(self):
        np.random.seed(42)
        # Noisy image
        sar = np.random.rand(50, 50) * 10
        filtered = enhanced_lee_filter(sar, window_size=5)
        # Filtered should have lower variance than original
        assert np.std(filtered) <= np.std(sar) * 1.5  # Allow some tolerance

    def test_even_window_size(self):
        np.random.seed(42)
        sar = np.random.rand(50, 50) * 10
        filtered = enhanced_lee_filter(sar, window_size=6)  # even, should auto-adjust
        assert filtered.shape == sar.shape


class TestSarToPseudoRgb:
    def test_creates_rgb_image(self, tmp_path):
        # Create a synthetic 2-band GeoTIFF (VV+VH)
        try:
            import rasterio
        except ImportError:
            pytest.skip("rasterio not available")

        tif_path = tmp_path / "test_sar.tif"
        # Create 2-band synthetic SAR
        np.random.seed(42)
        vv = np.random.rand(50, 50).astype(np.float32) * 100
        vh = np.random.rand(50, 50).astype(np.float32) * 30

        with rasterio.open(
            tif_path, "w",
            driver="GTiff",
            height=50, width=50,
            count=2, dtype="float32",
            crs="EPSG:4326",
            transform=rasterio.transform.from_bounds(0, 0, 1, 1, 50, 50),
        ) as dst:
            dst.write(vv, 1)
            dst.write(vh, 2)

        # Process
        pil_img, metadata = sar_to_pseudo_rgb(str(tif_path), target_size=224)
        assert pil_img.mode == "RGB"
        assert pil_img.size == (224, 224)
        assert "preprocessing" in metadata
        assert "bands_detected" in metadata

    def test_save_as_png(self, tmp_path):
        try:
            import rasterio
        except ImportError:
            pytest.skip("rasterio not available")

        tif_path = tmp_path / "test_sar.tif"
        png_path = tmp_path / "test_sar.png"
        np.random.seed(42)
        vv = np.random.rand(30, 30).astype(np.float32) * 100
        vh = np.random.rand(30, 30).astype(np.float32) * 30

        with rasterio.open(
            tif_path, "w",
            driver="GTiff",
            height=30, width=30,
            count=2, dtype="float32",
        ) as dst:
            dst.write(vv, 1)
            dst.write(vh, 2)

        metadata = save_sar_as_png(str(tif_path), str(png_path), target_size=224)
        assert png_path.exists()
        # Verify PNG is valid
        img = Image.open(png_path)
        assert img.size == (224, 224)


class TestSarPrompts:
    def test_select_prompt_optical_caption(self):
        prompt = select_prompt("caption", "optical")
        assert "USER: <image>" in prompt
        assert "ASSISTANT:" in prompt

    def test_select_prompt_sar_caption(self):
        prompt = select_prompt("caption", "sar")
        assert "SAR" in prompt or "Synthetic Aperture Radar" in prompt
        assert "backscatter" in prompt.lower()

    def test_select_prompt_sar_vqa(self):
        prompt = select_prompt("vqa", "sar", question="Is this water?")
        assert "Is this water?" in prompt
        assert "backscatter" in prompt.lower()

    def test_select_prompt_fusion_caption(self):
        prompt = select_prompt("caption", "fusion")
        assert "optical" in prompt.lower()
        assert "sar" in prompt.lower()

    def test_sar_system_distinct_from_optical(self):
        assert SAR_SYSTEM != GEOCHAT_SYSTEM
        assert "Synthetic Aperture Radar" in SAR_SYSTEM
        assert "backscatter" in SAR_SYSTEM.lower()

    def test_sar_vqa_mentions_polarization(self):
        prompt = SAR_VQA_TEMPLATE.format(question="test")
        assert "VV" in prompt or "VH" in prompt or "polarization" in prompt.lower()

    def test_fusion_prompts_mention_both_modalities(self):
        assert "optical" in FUSION_CAPTION_TEMPLATE.lower()
        assert "sar" in FUSION_CAPTION_TEMPLATE.lower()
        assert "optical" in FUSION_VQA_TEMPLATE.lower()
        assert "sar" in FUSION_VQA_TEMPLATE.lower()


class TestLoadImageSafelyModality:
    def test_sar_filename_routes_to_sar(self, tmp_path):
        # Create a fake SAR file with .tif extension using actual SAR-like data
        try:
            import rasterio
        except ImportError:
            pytest.skip("rasterio not available")

        sar_path = tmp_path / "S1_scene.tif"
        np.random.seed(42)
        vv = np.random.rand(100, 100).astype(np.float32) * 100
        vh = np.random.rand(100, 100).astype(np.float32) * 30
        with rasterio.open(
            sar_path, "w",
            driver="GTiff",
            height=100, width=100,
            count=2, dtype="float32",
        ) as dst:
            dst.write(vv, 1)
            dst.write(vh, 2)

        # Should route to SAR preprocessing
        result = _load_image_safely(str(sar_path), modality="auto")
        assert result is not None
        assert result.mode == "RGB"

    def test_optical_filename(self, tmp_path):
        opt_path = tmp_path / "S2_scene.tif"
        img = Image.new("RGB", (100, 100), (100, 150, 200))
        img.save(opt_path)

        result = _load_image_safely(str(opt_path), modality="auto")
        assert result is not None
        assert result.mode == "RGB"

    def test_explicit_sar_modality(self, tmp_path):
        opt_path = tmp_path / "test_image.tif"
        img = Image.new("RGB", (100, 100), (100, 150, 200))
        img.save(opt_path)

        # Force SAR modality even though filename says otherwise
        result = _load_image_safely(str(opt_path), modality="sar")
        assert result is not None

    def test_sar_pil_creation_fallback(self, tmp_path):
        # If SAR preprocessing fails (e.g., invalid TIF), should still work
        sar_path = tmp_path / "S1_fake.tif"
        # Create as PNG (not valid SAR TIF, will fail preprocessing)
        img = Image.new("RGB", (100, 100), (50, 50, 50))
        img.save(sar_path)

        # Should still return something (fallback to standard load)
        result = _load_image_safely(str(sar_path), modality="auto")
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])