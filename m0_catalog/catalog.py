"""
M0 Catalog - SQLite database for scene metadata.
M0 owns writes; M2/M3/M4/M5 read-only.
"""
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from shared.schemas import SceneMetadata
from .config import get_m0_config


SCHEMA_SQL = """
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
    dataset_origin TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_acquisition_time ON scenes(acquisition_time);
CREATE INDEX IF NOT EXISTS idx_dataset_origin ON scenes(dataset_origin);
CREATE INDEX IF NOT EXISTS idx_paired_scene_id ON scenes(paired_scene_id);
CREATE INDEX IF NOT EXISTS idx_sensor ON scenes(sensor);
CREATE INDEX IF NOT EXISTS idx_modality ON scenes(modality);
CREATE INDEX IF NOT EXISTS idx_object ON scenes(object);
"""


class Catalog:
    """SQLite catalog for scene metadata."""

    def __init__(self, db_path: Optional[str] = None, read_only: bool = False):
        config = get_m0_config()
        self.db_path = db_path or config.output.db_path
        self.read_only = read_only
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _scene_to_row(self, scene: SceneMetadata, dataset_origin: str) -> Dict[str, Any]:
        """Convert SceneMetadata to database row dict."""
        return {
            "scene_id": scene.scene_id,
            "sensor": scene.sensor,
            "modality": scene.modality,
            "acquisition_time": scene.acquisition_time.isoformat() if scene.acquisition_time else "",
            "geometry_wkt": scene.geometry_wkt,
            "file_path": scene.file_path,
            "cloud_cover": scene.cloud_cover,
            "object": scene.object,
            "paired_scene_id": scene.paired_scene_id,
            "dataset_origin": dataset_origin,
        }

    def _row_to_scene(self, row: sqlite3.Row) -> SceneMetadata:
        """Convert database row to SceneMetadata."""
        return SceneMetadata(
            scene_id=row["scene_id"],
            sensor=row["sensor"],
            modality=row["modality"],
            acquisition_time=datetime.fromisoformat(row["acquisition_time"].replace("Z", "+00:00")) if row["acquisition_time"] else None,
            geometry_wkt=row["geometry_wkt"],
            file_path=row["file_path"],
            cloud_cover=row["cloud_cover"],
            object=row["object"],
            paired_scene_id=row["paired_scene_id"],
        )

    def insert_scene(self, scene: SceneMetadata, dataset_origin: str) -> bool:
        """Insert or replace a scene."""
        if self.read_only:
            raise PermissionError("Catalog opened in read-only mode")

        row = self._scene_to_row(scene, dataset_origin)
        sql = """
            INSERT OR REPLACE INTO scenes
            (scene_id, sensor, modality, acquisition_time, geometry_wkt, file_path,
             cloud_cover, object, paired_scene_id, dataset_origin)
            VALUES (:scene_id, :sensor, :modality, :acquisition_time, :geometry_wkt,
                    :file_path, :cloud_cover, :object, :paired_scene_id, :dataset_origin)
        """
        self._conn.execute(sql, row)
        self._conn.commit()
        return True

    def insert_scenes(self, scenes: List[SceneMetadata], dataset_origin: str) -> int:
        """Batch insert scenes."""
        if self.read_only:
            raise PermissionError("Catalog opened in read-only mode")

        count = 0
        for scene in scenes:
            try:
                self.insert_scene(scene, dataset_origin)
                count += 1
            except Exception as e:
                print(f"Failed to insert scene {scene.scene_id}: {e}")
        return count

    def get_scene(self, scene_id: str) -> Optional[SceneMetadata]:
        """Get scene by ID."""
        cursor = self._conn.execute("SELECT * FROM scenes WHERE scene_id = ?", (scene_id,))
        row = cursor.fetchone()
        if row:
            return self._row_to_scene(row)
        return None

    def get_all_scenes(self, limit: Optional[int] = None) -> List[SceneMetadata]:
        """Get all scenes."""
        sql = "SELECT * FROM scenes ORDER BY acquisition_time DESC"
        if limit:
            sql += f" LIMIT {limit}"
        cursor = self._conn.execute(sql)
        return [self._row_to_scene(row) for row in cursor.fetchall()]

    def query_scenes(
        self,
        sensor: Optional[str] = None,
        modality: Optional[str] = None,
        object_filter: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        dataset_origin: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SceneMetadata]:
        """Query scenes with filters."""
        conditions = []
        params = []

        if sensor:
            conditions.append("sensor = ?")
            params.append(sensor)
        if modality:
            conditions.append("modality = ?")
            params.append(modality)
        if object_filter:
            conditions.append("object LIKE ?")
            params.append(f"%{object_filter}%")
        if start_date:
            conditions.append("acquisition_time >= ?")
            params.append(start_date.isoformat())
        if end_date:
            conditions.append("acquisition_time <= ?")
            params.append(end_date.isoformat())
        if dataset_origin:
            conditions.append("dataset_origin = ?")
            params.append(dataset_origin)

        sql = "SELECT * FROM scenes"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY acquisition_time DESC"
        if limit:
            sql += f" LIMIT {limit}"

        cursor = self._conn.execute(sql, params)
        return [self._row_to_scene(row) for row in cursor.fetchall()]

    def get_paired_scene(self, scene_id: str) -> Optional[SceneMetadata]:
        """Get the paired scene for a given scene."""
        scene = self.get_scene(scene_id)
        if scene and scene.paired_scene_id:
            return self.get_scene(scene.paired_scene_id)
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get catalog statistics."""
        cursor = self._conn.execute("SELECT COUNT(*) as total FROM scenes")
        total = cursor.fetchone()["total"]

        cursor = self._conn.execute("SELECT dataset_origin, COUNT(*) as count FROM scenes GROUP BY dataset_origin")
        by_origin = {row["dataset_origin"]: row["count"] for row in cursor.fetchall()}

        cursor = self._conn.execute("SELECT sensor, COUNT(*) as count FROM scenes GROUP BY sensor")
        by_sensor = {row["sensor"]: row["count"] for row in cursor.fetchall()}

        cursor = self._conn.execute("SELECT modality, COUNT(*) as count FROM scenes GROUP BY modality")
        by_modality = {row["modality"]: row["count"] for row in cursor.fetchall()}

        cursor = self._conn.execute("SELECT COUNT(*) as count FROM scenes WHERE paired_scene_id IS NOT NULL")
        paired = cursor.fetchone()["count"]

        return {
            "total_scenes": total,
            "by_dataset_origin": by_origin,
            "by_sensor": by_sensor,
            "by_modality": by_modality,
            "paired_scenes": paired,
        }

    def count(self) -> int:
        """Get total scene count."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM scenes")
        return cursor.fetchone()[0]


def get_catalog(read_only: bool = True) -> Catalog:
    """Get catalog instance."""
    config = get_m0_config()
    return Catalog(config.output.db_path, read_only=read_only)