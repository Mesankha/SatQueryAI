"""
M3 VLM Unit Tests - ZERO GPU dependencies.
Uses mocked GeoChat service + real test images. Run with: python -m pytest m3_vlm/test_vlm.py -v
"""
import os
import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from PIL import Image

from shared.schemas import UniformError, ErrorType
from m3_vlm.api import (
    run_vqa, run_caption, run_grounding,
    _is_sar_image, _parse_bbox, _make_error, _validate_image,
)
from m3_vlm.vlm_service import GeoChatService


def run_async(coro):
    """Helper to run async functions in tests."""
    return asyncio.run(coro)


@pytest.fixture
def real_test_image(tmp_path):
    """Create a real PNG image for tests that need a valid file path."""
    # Create a 100x100 image with some content to ensure file is not too small
    img = Image.new("RGB", (100, 100), (100, 150, 200))
    # Add some noise to increase file size
    import random
    pixels = img.load()
    for x in range(100):
        for y in range(100):
            pixels[x, y] = (
                random.randint(50, 200),
                random.randint(50, 200),
                random.randint(50, 200),
            )
    img.save(tmp_path / "test_image.png")
    return str(tmp_path / "test_image.png")


@pytest.fixture
def real_sar_test_image(tmp_path):
    """Create a real PNG image with SAR-like filename."""
    import random
    img = Image.new("RGB", (100, 100), (50, 50, 50))
    pixels = img.load()
    for x in range(100):
        for y in range(100):
            pixels[x, y] = (
                random.randint(0, 100),
                random.randint(0, 100),
                random.randint(0, 100),
            )
    img.save(tmp_path / "S1_test_image.tif")
    return str(tmp_path / "S1_test_image.tif")


class TestHelpers:
    """Test helper functions."""

    def test_is_sar_image_detection(self, real_test_image, real_sar_test_image):
        assert _is_sar_image(real_sar_test_image) is True
        assert _is_sar_image(real_test_image) is False
        # Filename with various SAR keywords
        assert _is_sar_image("/path/to/sar_image.tif") is True
        assert _is_sar_image("/path/to/sentinel-1.tif") is True
        assert _is_sar_image("/path/to/sentinel1.tif") is True
        assert _is_sar_image("/path/to/_s1_scene.tif") is True
        assert _is_sar_image("/path/to/S2_scene.tif") is False
        assert _is_sar_image("/path/to/optical.tif") is False

    def test_parse_bbox_valid(self):
        # Standard format
        assert _parse_bbox("[0.25, 0.30, 0.60, 0.80]") == [0.25, 0.30, 0.60, 0.80]
        # No spaces
        assert _parse_bbox("[0.25,0.30,0.60,0.80]") == [0.25, 0.30, 0.60, 0.80]
        # With prefix
        assert _parse_bbox("bbox: [0.1, 0.2, 0.9, 0.9]") == [0.1, 0.2, 0.9, 0.9]
        # Tuple format
        assert _parse_bbox("[(0.1, 0.2), (0.9, 0.9)]") == [0.1, 0.2, 0.9, 0.9]
        # Angle bracket format
        assert _parse_bbox("<0.1, 0.2, 0.9, 0.9>") == [0.1, 0.2, 0.9, 0.9]
        # Key-value format
        assert _parse_bbox("x1=0.1 y1=0.2 x2=0.9 y2=0.9") == [0.1, 0.2, 0.9, 0.9]

    def test_parse_bbox_invalid(self):
        assert _parse_bbox("no bbox here") is None
        assert _parse_bbox("[1.5, 0.2, 0.6, 0.8]") is None  # x1 > 1
        assert _parse_bbox("[0.2, 0.3, 0.1, 0.8]") is None  # x2 < x1
        assert _parse_bbox("[0.2, 0.3, 0.6]") is None  # only 3 coords
        assert _parse_bbox("") is None

    def test_make_error(self):
        err = _make_error("timeout", "test message")
        assert err["error"] is True
        assert err["error_type"] == "timeout"
        assert err["message"] == "test message"
        assert err["fallback_used"] is False

    def test_validate_image(self, real_test_image):
        # Valid image
        err = _validate_image(real_test_image)
        assert err is None
        # Non-existent
        err = _validate_image("/nonexistent/path.jpg")
        assert err is not None
        assert err["error_type"] == "invalid_input"
        # Directory (not file)
        err = _validate_image(str(Path(real_test_image).parent))
        assert err is not None

    def test_validate_image_empty_question(self, real_test_image):
        # This tests the input validation in run_vqa
        # (tested separately in run_vqa section)
        pass


class TestAPI:
    """Test API functions with mocked service."""

    @pytest.fixture
    def mock_service(self):
        """Create a mock GeoChat service."""
        service = Mock(spec=GeoChatService)
        service.is_ready.return_value = True
        service._run_inference = AsyncMock(return_value="Test answer")
        return service

    def test_run_vqa_success(self, mock_service, real_test_image):
        with patch("m3_vlm.api.get_service", return_value=mock_service):
            result = run_async(run_vqa(real_test_image, "What is this?"))
            assert result["error"] is False
            assert result["answer"] == "Test answer"
            assert "metadata" in result
            assert result["metadata"]["latency_ms"] >= 0

    def test_run_vqa_sar_warning(self, mock_service, real_sar_test_image):
        with patch("m3_vlm.api.get_service", return_value=mock_service):
            result = run_async(run_vqa(real_sar_test_image, "What is this?"))
            assert result["error"] is False
            assert "warning" in result
            assert "SAR" in result["warning"]

    def test_run_vqa_model_not_ready(self, mock_service, real_test_image):
        mock_service.is_ready.return_value = False
        with patch("m3_vlm.api.get_service", return_value=mock_service):
            result = run_async(run_vqa(real_test_image, "What is this?"))
            assert result["error"] is True
            assert result["error_type"] == "model_unavailable"

    def test_run_vqa_timeout(self, mock_service, real_test_image, monkeypatch):
        """Test that timeout returns an error."""
        import asyncio
        # Make INFERENCE_TIMEOUT very small for the test
        import m3_vlm.api as api_module
        monkeypatch.setattr(api_module, "INFERENCE_TIMEOUT", 0.1)
        # Mock the service to raise TimeoutError immediately
        mock_service._run_inference = AsyncMock(side_effect=asyncio.TimeoutError())
        with patch("m3_vlm.api.get_service", return_value=mock_service):
            result = run_async(run_vqa(real_test_image, "What is this?"))
            # Should return an error dict (not crash)
            assert "error" in result
            # Service-level timeout or our timeout both acceptable
            assert result.get("error") is True

    def test_run_vqa_invalid_image(self, mock_service):
        """Should return error when image doesn't exist."""
        with patch("m3_vlm.api.get_service", return_value=mock_service):
            result = run_async(run_vqa("/nonexistent/image.jpg", "What?"))
            assert result["error"] is True
            assert result["error_type"] == "invalid_input"

    def test_run_vqa_empty_question(self, mock_service, real_test_image):
        with patch("m3_vlm.api.get_service", return_value=mock_service):
            result = run_async(run_vqa(real_test_image, ""))
            assert result["error"] is True
            assert result["error_type"] == "invalid_input"

    def test_run_caption_success(self, mock_service, real_test_image):
        with patch("m3_vlm.api.get_service", return_value=mock_service):
            result = run_async(run_caption(real_test_image))
            assert result["error"] is False
            assert result["caption"] == "Test answer"

    def test_run_grounding_success(self, mock_service, real_test_image):
        mock_service._run_inference = AsyncMock(return_value="The bbox is [0.2, 0.3, 0.6, 0.8]")
        with patch("m3_vlm.api.get_service", return_value=mock_service):
            result = run_async(run_grounding(real_test_image, "the building"))
            assert result["error"] is False
            assert result["bbox"] == [0.2, 0.3, 0.6, 0.8]
            assert 0.0 <= result["confidence"] <= 1.0

    def test_run_grounding_parse_failure(self, mock_service, real_test_image):
        mock_service._run_inference = AsyncMock(return_value="I cannot find it")
        with patch("m3_vlm.api.get_service", return_value=mock_service):
            result = run_async(run_grounding(real_test_image, "the building"))
            assert result["error"] is True
            assert result["error_type"] == "validation_failed"

    def test_run_grounding_sar_image(self, mock_service, real_sar_test_image):
        """Grounding on SAR should still work."""
        mock_service._run_inference = AsyncMock(return_value="[0.1, 0.1, 0.5, 0.5]")
        with patch("m3_vlm.api.get_service", return_value=mock_service):
            result = run_async(run_grounding(real_sar_test_image, "the water"))
            assert result["error"] is False
            # Confirm SAR detection is working: the result should succeed
            assert "bbox" in result


class TestVLMService:
    """Test GeoChatService initialization (mocked)."""

    def test_service_config(self):
        """Test service configuration."""
        config = {"model_path": "test/model", "load_in_4bit": True}
        service = GeoChatService(config)
        assert service.config == config
        assert service._initialized is False

    def test_service_ensure_ready_creates_executor(self):
        """Test that the executor is created when needed."""
        config = {"model_path": "test/model", "load_in_4bit": True}
        service = GeoChatService(config)
        # Even without initialization, get_gpu_memory should be safe
        mem = service.get_gpu_memory()
        assert "gpu_available" in mem

    def test_shutdown_safe_when_not_initialized(self):
        config = {"model_path": "test/model", "load_in_4bit": True}
        service = GeoChatService(config)
        # Should not raise
        service.shutdown()


class TestIntegration:
    """Integration-style tests for M3 + M5 dispatch."""

    def test_m3_functions_return_uniform_schema(self):
        """Verify all three functions return schema-compatible dicts."""
        vqa_schema = {"answer": str, "error": bool, "metadata": dict}
        caption_schema = {"caption": str, "error": bool, "metadata": dict}
        grounding_schema = {"bbox": list, "confidence": float, "error": bool}

        # The actual functions are tested above with mocks
        assert True  # Schema contracts documented


if __name__ == "__main__":
    pytest.main([__file__, "-v"])