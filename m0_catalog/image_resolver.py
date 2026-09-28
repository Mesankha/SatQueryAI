"""
M0 Image Resolver — Handles local paths AND remote URLs transparently.

When the M0 catalog's `file_path` is a local file -> read directly.
When it's a URL (e.g. Planetary Computer signed URL) -> download to cache
on first access, then serve from cache for subsequent calls.

This is the key to "keep DB local but fetch imagery on demand".
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Cache directory for downloaded images
CACHE_DIR = Path(os.getenv("SATQUERY_IMAGE_CACHE", "./data/image_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Lock per URL to prevent concurrent downloads
_locks: dict = {}
_locks_lock = threading.Lock()


def _get_lock(key: str) -> threading.Lock:
    with _locks_lock:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def _is_url(path: str) -> bool:
    return path.startswith(("http://", "https://", "s3://", "gs://"))


def _url_to_cache_path(url: str) -> Path:
    """Generate a deterministic cache filename from a URL."""
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    # Try to preserve file extension
    ext = ".tif"
    for e in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jp2"):
        if e in url.lower():
            ext = e
            break
    return CACHE_DIR / f"{h}{ext}"


def resolve_image(
    file_path: str,
    force_download: bool = False,
    timeout: float = 3.0,
) -> str:
    """
    Resolve a file_path to a local file path.
    - If file_path is a local file: returns it unchanged
    - If file_path is a URL: downloads to cache and returns cache path
    - Returns None if resolution fails (caller should handle gracefully)

    Args:
        file_path: Local path or URL
        force_download: Re-download even if cached
        timeout: HTTP timeout in seconds

    Returns:
        Local file path (cached if from URL) or original path if already local
    """
    if not file_path:
        return None

    # Local file case
    if not _is_url(file_path):
        if Path(file_path).exists():
            return file_path
        logger.warning(f"Local file not found: {file_path}")
        return None

    # URL case
    cache_path = _url_to_cache_path(file_path)
    if cache_path.exists() and not force_download:
        return str(cache_path)

    # Need to download
    lock = _get_lock(str(cache_path))
    with lock:
        # Double-check after acquiring lock
        if cache_path.exists() and not force_download:
            return str(cache_path)

        try:
            logger.info(f"Downloading {file_path[:80]}... -> {cache_path.name}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            req = urllib.request.Request(
                file_path,
                headers={"User-Agent": "SatQuery-AI/1.0 (research)"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                with open(cache_path, "wb") as f:
                    shutil.copyfileobj(response, f)

            size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(f"Downloaded {size_mb:.1f}MB to cache")
            return str(cache_path)
        except Exception as e:
            logger.error(f"Download failed for {file_path[:80]}: {e}")
            # Clean up partial file
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except Exception:
                    pass
            return None


def is_cached(file_path: str) -> bool:
    """Check if a URL/file is already cached locally."""
    if not _is_url(file_path):
        return Path(file_path).exists()
    cache_path = _url_to_cache_path(file_path)
    return cache_path.exists()


def clear_cache(max_age_days: int = 30) -> int:
    """Remove cached images older than max_age_days. Returns count removed."""
    import time
    cutoff = time.time() - (max_age_days * 86400)
    removed = 0
    for p in CACHE_DIR.iterdir():
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except Exception:
            pass
    logger.info(f"Cleared {removed} cached images older than {max_age_days} days")
    return removed


def get_cache_size_mb() -> float:
    """Get current cache size in MB."""
    total = 0
    for p in CACHE_DIR.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


# Synchronous convenience for non-async code
def resolve_image_sync(file_path: str, **kwargs) -> str:
    return resolve_image(file_path, **kwargs)


# Async wrapper for use in async contexts
async def resolve_image_async(file_path: str, **kwargs) -> str:
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: resolve_image(file_path, **kwargs))