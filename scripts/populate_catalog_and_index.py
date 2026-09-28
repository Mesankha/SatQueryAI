#!/usr/bin/env python
"""
Populate catalog.db from fixtures/fixtures_scenes.json and build FAISS index.
Run this to make M2 retrieval real (using fixture data but real pipeline).
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss

DB_PATH = "./data/catalog.db"
FIXTURES_PATH = "./fixtures/fixtures_scenes.json"
FAISS_INDEX_PATH = "./data/faiss.index"
FAISS_IDS_PATH = "./data/faiss_ids.json"

def populate_catalog():
    """Load fixtures into SQLite catalog."""
    with open(FIXTURES_PATH) as f:
        scenes = json.load(f)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scenes (
            scene_id TEXT PRIMARY KEY,
            sensor TEXT NOT NULL,
            modality TEXT NOT NULL,
            acquisition_time TEXT NOT NULL,
            geometry_wkt TEXT NOT NULL,
            file_path TEXT,
            cloud_cover REAL,
            object TEXT,
            paired_scene_id TEXT,
            wavelengths_nm TEXT
        )
    """)

    for scene in scenes:
        conn.execute("""
            INSERT OR REPLACE INTO scenes
            (scene_id, sensor, modality, acquisition_time, geometry_wkt, file_path, cloud_cover, object, paired_scene_id, dataset_origin, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scene["scene_id"],
            scene["sensor"],
            scene["modality"],
            scene["acquisition_time"],
            scene["geometry_wkt"],
            scene.get("file_path"),
            scene.get("cloud_cover"),
            scene.get("object"),
            scene.get("paired_scene_id"),
            "fixture",
            datetime.utcnow().isoformat() + "Z"
        ))

    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
    conn.close()
    print(f"Catalog populated: {count} scenes")
    return scenes

def build_faiss_index(scenes):
    """Build FAISS index from scene descriptions using sentence-transformers."""
    # Create text descriptions for embedding
    texts = []
    ids = []
    for scene in scenes:
        # Build a rich text description
        parts = []
        if scene.get("object"):
            parts.append(f"{scene['object']}")
        parts.append(f"{scene['sensor']}")
        parts.append(f"{scene['modality']}")
        # Add location from geometry
        wkt = scene.get("geometry_wkt", "")
        if "POLYGON" in wkt:
            # Extract approximate center
            coords = wkt.replace("POLYGON((", "").replace("))", "").split(", ")
            if coords:
                first = coords[0].split()
                if len(first) >= 2:
                    lon, lat = float(first[0]), float(first[1])
                    parts.append(f"near {lon:.1f},{lat:.1f}")
        text = " ".join(parts)
        texts.append(text)
        ids.append(scene["scene_id"])

    print(f"Embedding {len(texts)} scene descriptions...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # Normalize for cosine similarity (IndexFlatIP)
    faiss.normalize_L2(embeddings)

    # Build index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_IDS_PATH, "w") as f:
        json.dump(ids, f)

    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    print(f"Saved to {FAISS_INDEX_PATH} and {FAISS_IDS_PATH}")

def main():
    Path("./data").mkdir(exist_ok=True)
    scenes = populate_catalog()
    build_faiss_index(scenes)
    print("\nDone! Now set 'retrieval.mock: false' in config.yaml to use real M2 pipeline.")

if __name__ == "__main__":
    main()