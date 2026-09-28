"""
M0 Image Resolver tests - no network required for most tests.
"""
import os
import tempfile
from pathlib import Path

import pytest

from m0_catalog.image_resolver import (
    resolve_image,
    is_cached,
    get_cache_size_mb,
    clear_cache,
    _is_url,
    _url_to_cache_path,
    CACHE_DIR,
)


class TestIsUrl:
    def test_http_url(self):
        assert _is_url("http://example.com/file.tif") is True

    def test_https_url(self):
        assert _is_url("https://example.com/file.tif") is True

    def test_s3_url(self):
        assert _is_url("s3://bucket/file.tif") is True

    def test_gs_url(self):
        assert _is_url("gs://bucket/file.tif") is True

    def test_local_path(self):
        assert _is_url("./data/file.tif") is False
        assert _is_url("C:/data/file.tif") is False
        assert _is_url("/absolute/path.tif") is False


class TestUrlToCachePath:
    def test_deterministic_hash(self):
        url = "https://planetarycomputer.microsoft.com/api/stac/v1/test"
        p1 = _url_to_cache_path(url)
        p2 = _url_to_cache_path(url)
        assert p1 == p2
        assert p1.parent == CACHE_DIR

    def test_different_urls_different_paths(self):
        p1 = _url_to_cache_path("https://example.com/file1.tif")
        p2 = _url_to_cache_path("https://example.com/file2.tif")
        assert p1 != p2

    def test_extension_preserved(self):
        p = _url_to_cache_path("https://example.com/image.png")
        assert p.suffix == ".png"


class TestResolveImage:
    def test_empty_path(self):
        assert resolve_image("") is None

    def test_nonexistent_local_file(self):
        assert resolve_image("Z:/nonexistent/file.tif") is None

    def test_existing_local_file(self, tmp_path):
        f = tmp_path / "test.tif"
        f.write_bytes(b"fake image data")
        result = resolve_image(str(f))
        assert result == str(f)
        assert Path(result).exists()

    def test_failed_url_returns_none(self):
        # 404 will fail
        result = resolve_image("https://httpbin.org/status/404")
        assert result is None


class TestCache:
    def test_cache_dir_exists(self):
        assert CACHE_DIR.exists()
        assert CACHE_DIR.is_dir()

    def test_cache_size_is_number(self):
        size = get_cache_size_mb()
        assert isinstance(size, float)
        assert size >= 0

    def test_clear_cache(self, tmp_path):
        # Add a fake old file
        test_file = CACHE_DIR / "old_test_file.tmp"
        test_file.write_bytes(b"x" * 100)
        # Set mtime to long ago
        import time
        old_time = time.time() - (100 * 86400)  # 100 days ago
        os.utime(test_file, (old_time, old_time))
        # Clear with 30-day cutoff
        removed = clear_cache(max_age_days=30)
        assert removed >= 1
        assert not test_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])