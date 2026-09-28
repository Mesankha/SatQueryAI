"""Populate M0 catalog from fixture scenes for end-to-end demo."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from m0_catalog.catalog import Catalog
from shared.schemas import SceneMetadata


def main():
    fixtures_path = "fixtures/fixtures_scenes.json"
    db_path = "data/catalog.db"

    # Load fixtures
    with open(fixtures_path) as f:
        fixtures = json.load(f)

    # Open catalog in write mode
    catalog = Catalog(db_path, read_only=False)

    count = 0
    for fix in fixtures:
        try:
            # Parse acquisition_time
            acq_time = datetime.fromisoformat(
                fix["acquisition_time"].replace("Z", "+00:00")
            )

            scene = SceneMetadata(
                scene_id=fix["scene_id"],
                sensor=fix["sensor"],
                modality=fix["modality"],
                acquisition_time=acq_time,
                geometry_wkt=fix["geometry_wkt"],
                file_path=fix.get("file_path"),
                cloud_cover=fix.get("cloud_cover"),
                object=fix.get("object"),
                paired_scene_id=fix.get("paired_scene_id"),
            )
            catalog.insert_scene(scene, dataset_origin="fixture")
            count += 1
            print(f"Inserted: {fix['scene_id'][:20]}... ({fix['sensor']})")
        except Exception as e:
            print(f"Failed to insert {fix.get('scene_id', 'unknown')}: {e}")

    catalog.close()
    print(f"\nTotal: {count} scenes inserted into M0 catalog")


if __name__ == "__main__":
    main()
