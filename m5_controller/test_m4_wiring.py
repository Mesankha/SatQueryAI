"""
M4 Wiring Tests - verify the M0 <-> M2 <-> M4 <-> M5 pipeline works end-to-end.

These tests:
  1. M4 change detection runs directly
  2. M5 dispatch_table calls M4 for change queries
  3. M4 results flow back through M5
  4. Paired scenes are properly handled
"""
import asyncio
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Setup path
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def create_test_image_pair(path1, path2, change_type="flood"):
    """Create two synthetic satellite images for change detection."""
    size = 256
    t1 = Image.new("RGB", (size, size), (100, 150, 80))  # vegetation
    t2 = Image.new("RGB", (size, size), (100, 150, 80))

    if change_type == "flood":
        # Add water in t2
        for x in range(50, 200):
            for y in range(100, 180):
                t2.putpixel((x, y), (30, 80, 180))
    elif change_type == "urban":
        # Add buildings in t2
        for x in range(80, 180):
            for y in range(80, 180):
                t2.putpixel((x, y), (120, 120, 120))

    t1.save(path1, "PNG")
    t2.save(path2, "PNG")


@pytest.fixture
def m0_with_paired_scenes():
    """Create M0 catalog with paired scenes (bi-temporal) for change detection."""
    with tempfile.TemporaryDirectory() as tmp:
        # Create test images
        t1_path = os.path.join(tmp, "before.png")
        t2_path = os.path.join(tmp, "after.png")
        create_test_image_pair(t1_path, t2_path, "flood")

        # Create M0 catalog
        from m0_catalog.catalog import Catalog
        from shared.schemas import SceneMetadata

        db_path = os.path.join(tmp, "test_catalog.db")
        catalog = Catalog(db_path, read_only=False)

        wkt = "POLYGON((77.0 28.5, 77.3 28.5, 77.3 28.8, 77.0 28.8, 77.0 28.5))"

        scene_t1 = SceneMetadata(
            scene_id="test-t1",
            sensor="Sentinel-2",
            modality="optical",
            acquisition_time=datetime(2023, 8, 15, 10, 30, tzinfo=timezone.utc),
            geometry_wkt=wkt,
            file_path=t1_path,
            object="vegetation",
        )
        scene_t2 = SceneMetadata(
            scene_id="test-t2",
            sensor="Sentinel-2",
            modality="optical",
            acquisition_time=datetime(2024, 2, 15, 10, 30, tzinfo=timezone.utc),
            geometry_wkt=wkt,
            file_path=t2_path,
            object="vegetation",
            paired_scene_id="test-t1",
        )
        catalog.insert_scene(scene_t1, dataset_origin="test")
        catalog.insert_scene(scene_t2, dataset_origin="test")
        catalog.close()

        # Set env var for M2/M5 to use this catalog
        old_env = os.environ.get("M0_CATALOG_DB")
        os.environ["M0_CATALOG_DB"] = db_path

        yield db_path, scene_t1.scene_id, scene_t2.scene_id

        if old_env is None:
            os.environ.pop("M0_CATALOG_DB", None)
        else:
            os.environ["M0_CATALOG_DB"] = old_env


class TestM4Direct:
    """Test M4 change detection directly."""

    def test_m4_imports(self):
        """M4 module imports successfully."""
        from m4_changedetect import detect_change, ChangeResult, ChangeDetector
        assert detect_change is not None
        assert ChangeResult is not None
        assert ChangeDetector is not None

    def test_m4_detect_change_returns_result(self, tmp_path):
        """M4 detect_change returns a ChangeResult."""
        from m4_changedetect import detect_change, ChangeResult

        t1 = tmp_path / "before.png"
        t2 = tmp_path / "after.png"
        create_test_image_pair(str(t1), str(t2), "flood")

        result = detect_change(str(t1), str(t2))

        assert result is not None
        assert hasattr(result, "change_score")
        assert hasattr(result, "change_description")

    def test_m4_handles_missing_metadata(self, tmp_path):
        """M4 handles missing metadata gracefully."""
        from m4_changedetect import detect_change

        t1 = tmp_path / "before.png"
        t2 = tmp_path / "after.png"
        create_test_image_pair(str(t1), str(t2), "flood")

        # No metadata - should still work (just less precise)
        result = detect_change(str(t1), str(t2))
        assert result is not None

    def test_m4_with_metadata(self, tmp_path):
        """M4 accepts metadata for temporal pair."""
        from m4_changedetect import detect_change

        t1 = tmp_path / "before.png"
        t2 = tmp_path / "after.png"
        create_test_image_pair(str(t1), str(t2), "flood")

        meta = {"constellation": "SENTINEL-2", "date": "2024-02-15"}
        result = detect_change(str(t1), str(t2), metadata_t1=meta, metadata_t2=meta)
        assert result is not None


class TestM4ThroughM5:
    """Test M4 is called from M5 dispatch."""

    @pytest.mark.asyncio
    async def test_m5_routes_change_query_to_m4(self, m0_with_paired_scenes):
        """M5 routes 'change' task_type to M4."""
        import m5_controller  # Apply M2 patch via package import
        from m5_controller.dispatch_table import orchestrate_async

        db_path, t1_id, t2_id = m0_with_paired_scenes

        response = await orchestrate_async(
            "What changed in Delhi between August 2023 and February 2024?"
        )

        assert response.trace.task_selected == "change"
        # M4 may return error due to simplified test images, but pipeline should run
        assert response.trace is not None

    @pytest.mark.asyncio
    async def test_m5_change_pipeline_calls_m4(self, m0_with_paired_scenes):
        """M5's change pipeline actually invokes M4 detect_change."""
        import m5_controller  # Apply M2 patch
        from m5_controller.dispatch_table import orchestrate_async
        from unittest.mock import patch

        db_path, t1_id, t2_id = m0_with_paired_scenes

        with patch("m4_changedetect.detect_change") as mock_detect:
            from m4_changedetect.schemas.change_schema import ChangeResult
            mock_detect.return_value = ChangeResult(
                change_score=0.75,
                change_description="Mocked change: vegetation converted to water",
            )

            response = await orchestrate_async(
                "What changed in Delhi between August 2023 and February 2024?"
            )

            # Verify M4 was called
            assert mock_detect.called
            # Verify the result was included
            if response.results:
                assert response.results[0].model_outputs.change_description is not None
                assert "Mocked change" in response.results[0].model_outputs.change_description


class TestM4PairedScenes:
    """Test M4 with paired bi-temporal scenes."""

    @pytest.mark.asyncio
    async def test_paired_scenes_both_used(self, m0_with_paired_scenes):
        """M4 should use both t1 and t2 images from paired scenes."""
        import m5_controller
        from m5_controller.dispatch_table import orchestrate_async
        from unittest.mock import patch
        from m4_changedetect.schemas.change_schema import ChangeResult

        db_path, t1_id, t2_id = m0_with_paired_scenes

        captured_paths = []

        def mock_capture(t1, t2, *args, **kwargs):
            captured_paths.append((t1, t2))
            return ChangeResult(
                change_score=0.5,
                change_description="Test change",
            )

        with patch("m4_changedetect.detect_change", side_effect=mock_capture):
            await orchestrate_async(
                "What changed in Delhi between August 2023 and February 2024?"
            )

        # M4 should have been called with both image paths
        assert len(captured_paths) >= 1, "M4 was not called"
        called_t1, called_t2 = captured_paths[0]
        # Both paths should point to real files
        assert os.path.exists(called_t1), f"t1 path doesn't exist: {called_t1}"
        assert os.path.exists(called_t2), f"t2 path doesn't exist: {called_t2}"


class TestM4ErrorHandling:
    """Test M4 error handling through M5."""

    @pytest.mark.asyncio
    async def test_m4_exception_returns_graceful_error(self, m0_with_paired_scenes):
        """If M4 raises an exception, M5 should return a graceful error result."""
        import m5_controller
        from m5_controller.dispatch_table import orchestrate_async
        from unittest.mock import patch

        db_path, t1_id, t2_id = m0_with_paired_scenes

        with patch("m4_changedetect.detect_change", side_effect=Exception("M4 crash")):
            response = await orchestrate_async(
                "What changed in Delhi between August 2023 and February 2024?"
            )

        # M5 should still return a response, with error info
        assert response is not None
        assert response.trace is not None
        # Fallback should be marked
        assert response.trace.fallback_used is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])