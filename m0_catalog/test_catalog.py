"""
M0 Catalog Unit Tests - ZERO external dependencies.
Uses fixtures and temporary databases.
Run with: python -m pytest m0_catalog/test_catalog.py -v
"""
import json
import tempfile
import os
from pathlib import Path

import pytest

from shared.schemas import SceneMetadata
from m0_catalog.catalog import Catalog
from m0_catalog.validate import verify_crs, verify_pairs, generate_manifest


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def sample_scenes():
    """Create sample SceneMetadata objects for testing."""
    from datetime import datetime, timezone
    return [
        SceneMetadata(
            scene_id="test-scene-1",
            sensor="Sentinel-2",
            modality="multispectral",
            acquisition_time=datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            geometry_wkt="POLYGON((77.0 28.5, 77.5 28.5, 77.5 29.0, 77.0 29.0, 77.0 28.5))",
            file_path="/fake/path/scene1.tif",
            cloud_cover=5.2,
            object="agriculture",
            paired_scene_id="test-scene-2",
        ),
        SceneMetadata(
            scene_id="test-scene-2",
            sensor="Sentinel-2",
            modality="multispectral",
            acquisition_time=datetime(2024, 6, 20, 8, 15, tzinfo=timezone.utc),
            geometry_wkt="POLYGON((77.0 28.5, 77.5 28.5, 77.5 29.0, 77.0 29.0, 77.0 28.5))",
            file_path="/fake/path/scene2.tif",
            cloud_cover=3.1,
            object="agriculture",
            paired_scene_id="test-scene-1",
        ),
        SceneMetadata(
            scene_id="test-scene-3",
            sensor="Sentinel-1",
            modality="sar",
            acquisition_time=datetime(2023, 8, 10, 6, 45, tzinfo=timezone.utc),
            geometry_wkt="POLYGON((91.0 26.0, 92.0 26.0, 92.0 27.0, 91.0 27.0, 91.0 26.0))",
            file_path="/fake/path/scene3.tif",
            cloud_cover=None,
            object="water",
            paired_scene_id=None,
        ),
    ]


class TestCatalog:
    """Test Catalog CRUD operations."""

    def test_catalog_init(self, temp_db):
        catalog = Catalog(temp_db, read_only=False)
        assert catalog._conn is not None
        catalog.close()

    def test_insert_and_get_scene(self, temp_db, sample_scenes):
        catalog = Catalog(temp_db, read_only=False)
        scene = sample_scenes[0]

        catalog.insert_scene(scene, "test_origin")
        retrieved = catalog.get_scene(scene.scene_id)

        assert retrieved is not None
        assert retrieved.scene_id == scene.scene_id
        assert retrieved.sensor == scene.sensor
        assert retrieved.object == scene.object
        catalog.close()

    def test_insert_multiple_scenes(self, temp_db, sample_scenes):
        catalog = Catalog(temp_db, read_only=False)
        count = catalog.insert_scenes(sample_scenes, "test_origin")
        assert count == 3

        all_scenes = catalog.get_all_scenes()
        assert len(all_scenes) == 3
        catalog.close()

    def test_query_scenes_with_filters(self, temp_db, sample_scenes):
        catalog = Catalog(temp_db, read_only=False)
        catalog.insert_scenes(sample_scenes, "test_origin")

        # Filter by sensor
        s2_scenes = catalog.query_scenes(sensor="Sentinel-2")
        assert len(s2_scenes) == 2

        # Filter by modality
        sar_scenes = catalog.query_scenes(modality="sar")
        assert len(sar_scenes) == 1

        # Filter by object
        ag_scenes = catalog.query_scenes(object_filter="agriculture")
        assert len(ag_scenes) == 2

        catalog.close()

    def test_paired_scene_retrieval(self, temp_db, sample_scenes):
        catalog = Catalog(temp_db, read_only=False)
        catalog.insert_scenes(sample_scenes, "test_origin")

        scene1 = catalog.get_scene("test-scene-1")
        paired = catalog.get_paired_scene("test-scene-1")

        assert paired is not None
        assert paired.scene_id == "test-scene-2"
        catalog.close()

    def test_catalog_stats(self, temp_db, sample_scenes):
        catalog = Catalog(temp_db, read_only=False)
        catalog.insert_scenes(sample_scenes, "test_origin")

        stats = catalog.get_stats()
        assert stats["total_scenes"] == 3
        assert stats["by_sensor"]["Sentinel-2"] == 2
        assert stats["by_sensor"]["Sentinel-1"] == 1
        assert stats["by_modality"]["multispectral"] == 2
        assert stats["by_modality"]["sar"] == 1
        assert stats["paired_scenes"] == 2  # test-scene-1 and test-scene-2
        catalog.close()

    def test_read_only_mode(self, temp_db, sample_scenes):
        # First populate
        catalog = Catalog(temp_db, read_only=False)
        catalog.insert_scene(sample_scenes[0], "test_origin")
        catalog.close()

        # Open read-only
        catalog = Catalog(temp_db, read_only=True)
        scene = catalog.get_scene("test-scene-1")
        assert scene is not None

        # Should fail to insert
        with pytest.raises(PermissionError):
            catalog.insert_scene(sample_scenes[1], "test_origin")
        catalog.close()


class TestValidation:
    """Test validation functions."""

    def test_verify_crs(self, temp_db, sample_scenes):
        catalog = Catalog(temp_db, read_only=False)
        catalog.insert_scenes(sample_scenes, "test_origin")

        results = verify_crs(catalog, sample_size=2)
        assert "passed" in results
        assert "checks" in results
        assert len(results["checks"]) > 0
        catalog.close()

    def test_verify_pairs(self, temp_db, sample_scenes):
        catalog = Catalog(temp_db, read_only=False)
        catalog.insert_scenes(sample_scenes, "test_origin")

        results = verify_pairs(catalog)
        assert results["total_pairs"] == 1
        assert results["valid_pairs"] == 1
        assert results["passed"] is True
        catalog.close()

    def test_verify_pairs_broken(self, temp_db, sample_scenes):
        catalog = Catalog(temp_db, read_only=False)
        # Insert only one of the pair
        catalog.insert_scene(sample_scenes[0], "test_origin")

        results = verify_pairs(catalog)
        assert results["passed"] is False
        assert len(results["errors"]) > 0
        catalog.close()

    def test_generate_manifest(self, temp_db, sample_scenes):
        catalog = Catalog(temp_db, read_only=False)
        catalog.insert_scenes(sample_scenes, "test_origin")

        manifest = generate_manifest(catalog)
        assert "generated_at" in manifest
        assert "statistics" in manifest
        assert "quality_gates" in manifest
        assert "caveats" in manifest
        assert manifest["statistics"]["total_scenes"] == 3
        catalog.close()


class TestManifest:
    """Test manifest read/write."""

    def test_write_manifest(self, temp_db, sample_scenes):
        catalog = Catalog(temp_db, read_only=False)
        catalog.insert_scenes(sample_scenes, "test_origin")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            manifest_path = f.name

        try:
            from m0_catalog.manifest import write_manifest, read_manifest
            write_manifest(catalog, manifest_path)

            manifest = read_manifest(manifest_path)
            assert manifest["statistics"]["total_scenes"] == 3
        finally:
            if os.path.exists(manifest_path):
                os.unlink(manifest_path)
        catalog.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])