"""
Phase 4: M0 Catalog Population with Synthetic Test Data
Creates realistic synthetic scenes for testing M2 retrieval end-to-end.
"""
import random
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

from shared.schemas import SceneMetadata
from m0_catalog.catalog import Catalog
from m0_catalog.config import get_m0_config


def create_synthetic_scenes() -> list[SceneMetadata]:
    """Create synthetic test scenes covering all modalities and tasks."""
    
    # Indian cities/regions for geographic diversity
    locations = [
        ("Delhi", 28.6139, 77.2090),
        ("Mumbai", 19.0760, 72.8777),
        ("Bangalore", 12.9716, 77.5946),
        ("Chennai", 13.0827, 80.2707),
        ("Kolkata", 22.5726, 88.3639),
        ("Hyderabad", 17.3850, 78.4867),
        ("Pune", 18.5204, 73.8567),
        ("Ahmedabad", 23.0225, 72.5714),
        ("Jaipur", 26.9124, 75.7873),
        ("Lucknow", 26.8467, 80.9462),
        ("Assam", 26.2006, 92.9376),
        ("Kerala", 10.8505, 76.2711),
        ("Rajasthan", 27.0238, 74.2179),
        ("West Bengal", 22.9868, 87.8550),
    ]
    
    # Land cover classes
    land_covers = [
        "urban", "agriculture", "forest", "water", "barren",
        "wetland", "grassland", "built-up", "cropland", "residential"
    ]
    
    # Sensor configurations
    sensors = [
        {"sensor": "Sentinel-2", "modality": "multispectral", "prefix": "S2"},
        {"sensor": "Sentinel-1", "modality": "sar", "prefix": "S1"},
    ]
    
    scenes = []
    base_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    
    for i, (loc_name, lat, lon) in enumerate(locations):
        # Create a small bounding box around the location
        # ~0.1 degree ~ 11km
        min_lon, max_lon = lon - 0.05, lon + 0.05
        min_lat, max_lat = lat - 0.05, lat + 0.05
        geometry_wkt = f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
        
        # Create paired scenes for change detection
        for sensor_config in sensors:
            for j in range(2):  # Two dates per location per sensor
                date_offset = random.randint(0, 365)
                acquisition_time = base_date + timedelta(days=date_offset)
                
                land_cover = random.choice(land_covers)
                
                scene_id = f"{sensor_config['prefix'].lower()}_{loc_name.lower()}_{land_cover}_{i:02d}_{j:02d}_{uuid.uuid4().hex[:8]}"
                
                # Create file path
                file_path = f"./data/scenes/{scene_id}.tif"
                
                scene = SceneMetadata(
                    scene_id=scene_id,
                    sensor=sensor_config["sensor"],
                    modality=sensor_config["modality"],
                    acquisition_time=acquisition_time,
                    geometry_wkt=geometry_wkt,
                    file_path=file_path,
                    cloud_cover=random.uniform(0, 20) if sensor_config["modality"] != "sar" else None,
                    object=land_cover,
                    paired_scene_id=None,  # Will be set for pairs
                )
                scenes.append(scene)
    
    # Create pairs for change detection (same location, same sensor, different dates)
    for sensor_config in sensors:
        prefix = sensor_config["prefix"].lower()
        sensor_scenes = [s for s in scenes if s.sensor == sensor_config["sensor"]]
        
        # Group by location
        by_location = {}
        for s in sensor_scenes:
            loc = s.scene_id.split("_")[1]  # location name
            if loc not in by_location:
                by_location[loc] = []
            by_location[loc].append(s)
        
        # Pair consecutive dates for each location
        for loc, loc_scenes in by_location.items():
            loc_scenes.sort(key=lambda x: x.acquisition_time)
            for k in range(len(loc_scenes) - 1):
                loc_scenes[k].paired_scene_id = loc_scenes[k + 1].scene_id
                loc_scenes[k + 1].paired_scene_id = loc_scenes[k].scene_id
    
    return scenes


def create_fusion_pairs() -> list[tuple[SceneMetadata, SceneMetadata]]:
    """Create optical-SAR pairs for fusion testing (same location, similar dates)."""
    
    locations = [
        ("Assam", 26.2006, 92.9376),
        ("Kerala", 10.8505, 76.2711),
        ("Rajasthan", 27.0238, 74.2179),
        ("West Bengal", 22.9868, 87.8550),
    ]
    
    pairs = []
    base_date = datetime(2024, 2, 15, tzinfo=timezone.utc)
    
    for i, (loc_name, lat, lon) in enumerate(locations):
        min_lon, max_lon = lon - 0.05, lon + 0.05
        min_lat, max_lat = lat - 0.05, lat + 0.05
        geometry_wkt = f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
        
        # Optical scene
        optical = SceneMetadata(
            scene_id=f"S2_{loc_name.lower()}_fusion_{i:02d}",
            sensor="Sentinel-2",
            modality="multispectral",
            acquisition_time=base_date + timedelta(days=random.randint(-5, 5)),
            geometry_wkt=geometry_wkt,
            file_path=f"./data/scenes/S2_{loc_name.lower()}_fusion_{i:02d}.tif",
            cloud_cover=random.uniform(0, 15),
            object="mixed",
            paired_scene_id=None,
        )
        
        # SAR scene (co-registered)
        sar = SceneMetadata(
            scene_id=f"S1_{loc_name.lower()}_fusion_{i:02d}",
            sensor="Sentinel-1",
            modality="sar",
            acquisition_time=base_date + timedelta(days=random.randint(-5, 5)),
            geometry_wkt=geometry_wkt,
            file_path=f"./data/scenes/S1_{loc_name.lower()}_fusion_{i:02d}.tif",
            cloud_cover=None,
            object="mixed",
            paired_scene_id=None,
        )
        
        pairs.append((optical, sar))
    
    return pairs


def populate_catalog():
    """Populate catalog with synthetic test data."""
    config = get_m0_config()
    db_path = config.output.db_path
    
    print("=" * 60)
    print("M0 CATALOG - SYNTHETIC DATA POPULATION")
    print("=" * 60)
    
    # Create data directories
    Path(config.output.image_root).mkdir(parents=True, exist_ok=True)
    Path("./data/scenes").mkdir(parents=True, exist_ok=True)
    
    catalog = Catalog(db_path, read_only=False)
    
    # Clear existing data
    print("Clearing existing catalog...")
    catalog._conn.execute("DELETE FROM scenes")
    catalog._conn.commit()
    
    # Create synthetic scenes
    print("Generating synthetic scenes...")
    scenes = create_synthetic_scenes()
    print(f"Generated {len(scenes)} individual scenes")
    
    # Create fusion pairs
    print("Generating fusion pairs...")
    fusion_pairs = create_fusion_pairs()
    for optical, sar in fusion_pairs:
        scenes.append(optical)
        scenes.append(sar)
    print(f"Generated {len(fusion_pairs)} fusion pairs ({len(fusion_pairs)*2} scenes)")
    
    # Insert all scenes
    print("Inserting into catalog...")
    inserted = catalog.insert_scenes(scenes, "synthetic_test")
    print(f"Inserted {inserted} scenes")
    
    # Verify
    stats = catalog.get_stats()
    print("\nCatalog Statistics:")
    print(f"  Total Scenes: {stats['total_scenes']}")
    print(f"  By Sensor: {stats['by_sensor']}")
    print(f"  By Modality: {stats['by_modality']}")
    print(f"  Paired Scenes: {stats['paired_scenes']}")
    print(f"  By Origin: {stats['by_dataset_origin']}")
    
    catalog.close()
    print("\nCatalog population complete!")
    return stats


if __name__ == "__main__":
    populate_catalog()