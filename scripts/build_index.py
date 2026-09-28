#!/usr/bin/env python
"""
build_index.py - Build FAISS index from catalog scenes for image-similarity re-rank.

Embeds every catalog scene image into FAISS (IndexFlatIP) for future image-similarity re-rank.
Writes a parallel scene_ids list to ./data/faiss_ids.json.

NOTE: This index is NOT yet used at query time. It is built for future image-similarity re-rank.
The module docstring documents this limitation.
"""
import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import faiss
import numpy as np

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from m2_retrieval.dofa_embedder import embed_images, load_dofa, unload_dofa
from shared.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_index(
    db_path: str = "./data/catalog.db",
    faiss_index_path: str = "./data/faiss.index",
    faiss_ids_path: str = "./data/faiss_ids.json",
    batch_size: int = 16,
) -> dict[str, Any]:
    """
    Build FAISS index from all catalog scenes.

    Args:
        db_path: Path to catalog SQLite database
        faiss_index_path: Output path for FAISS index
        faiss_ids_path: Output path for scene IDs JSON
        batch_size: Batch size for embedding

    Returns:
        Dict with status and statistics
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Catalog database not found: {db_path}")

    # Load all scenes with valid file paths
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT scene_id, file_path FROM scenes
        WHERE file_path IS NOT NULL AND file_path != ''
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        logger.warning("No scenes with file paths found in catalog")
        return {"status": "no_scenes", "indexed": 0}

    scene_ids = [row["scene_id"] for row in rows]
    file_paths = [row["file_path"] for row in rows]

    # Filter to only existing files
    valid_pairs = [(sid, fp) for sid, fp in zip(scene_ids, file_paths) if Path(fp).exists()]
    if len(valid_pairs) < len(scene_ids):
        logger.warning(f"Skipping {len(scene_ids) - len(valid_pairs)} scenes with missing files")

    scene_ids = [p[0] for p in valid_pairs]
    file_paths = [p[1] for p in valid_pairs]

    logger.info(f"Embedding {len(scene_ids)} scenes...")

    # Load DOFA and embed
    load_dofa()
    # Dummy wavelengths for API compatibility
    wavelengths_nm = [[0.0]] * len(file_paths)
    embeddings = embed_images(file_paths, wavelengths_nm, batch_size=batch_size)

    # Verify embeddings
    if embeddings.shape[0] != len(scene_ids):
        logger.warning(f"Embedding count mismatch: {embeddings.shape[0]} vs {len(scene_ids)}")
        scene_ids = scene_ids[:embeddings.shape[0]]

    # L2 normalize for IndexFlatIP (inner product = cosine on normalized vectors)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # Save index and IDs
    Path(faiss_index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, faiss_index_path)

    with open(faiss_ids_path, "w") as f:
        json.dump(scene_ids, f)

    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    logger.info(f"Saved index to {faiss_index_path}")
    logger.info(f"Saved IDs to {faiss_ids_path}")

    # Unload DOFA to free GPU
    unload_dofa()

    return {
        "status": "success",
        "indexed": index.ntotal,
        "dimension": dim,
        "faiss_index": faiss_index_path,
        "faiss_ids": faiss_ids_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from catalog scenes")
    parser.add_argument("--db", default="./data/catalog.db", help="Catalog database path")
    parser.add_argument("--faiss-index", default="./data/faiss.index", help="Output FAISS index path")
    parser.add_argument("--faiss-ids", default="./data/faiss_ids.json", help="Output scene IDs JSON path")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding")

    args = parser.parse_args()

    try:
        result = build_index(
            db_path=args.db,
            faiss_index_path=args.faiss_index,
            faiss_ids_path=args.faiss_ids,
            batch_size=args.batch_size,
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        logger.error(f"Index build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import sys
    main()