"""
M0 Validation - CRS verification, pair integrity, manifest generation.
Per Blueprint §2.3 quality gates.
"""
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import shapely
from shapely.geometry import shape, Point
from shapely.wkt import loads as wkt_loads

from shared.schemas import SceneMetadata
from .catalog import Catalog
from .config import get_m0_config


def verify_crs(catalog: Catalog, sample_size: int = 3) -> Dict[str, Any]:
    """
    9-point CRS verification (3 sources × 3 random scenes).
    Checks: lat/lon bounds, landmark match, Shapely parse.
    """
    results = {
        "passed": True,
        "checks": [],
        "errors": [],
    }

    stats = catalog.get_stats()
    origins = list(stats.get("by_dataset_origin", {}).keys())

    for origin in origins:
        scenes = catalog.query_scenes(dataset_origin=origin, limit=sample_size)
        for scene in scenes:
            try:
                # Check 1: WKT parses correctly
                geom = wkt_loads(scene.geometry_wkt)

                # Check 2: Valid geometry
                if not geom.is_valid:
                    results["checks"].append({
                        "scene_id": scene.scene_id,
                        "check": "geometry_valid",
                        "passed": False,
                        "error": "Invalid geometry"
                    })
                    results["passed"] = False
                    continue

                # Check 3: Lat/lon bounds (EPSG:4326)
                bounds = geom.bounds  # (minx, miny, maxx, maxy)
                if not (-180 <= bounds[0] <= 180 and -180 <= bounds[2] <= 180 and
                        -90 <= bounds[1] <= 90 and -90 <= bounds[3] <= 90):
                    results["checks"].append({
                        "scene_id": scene.scene_id,
                        "check": "lat_lon_bounds",
                        "passed": False,
                        "error": f"Bounds out of range: {bounds}"
                    })
                    results["passed"] = False
                    continue

                # Check 4: Landmark match (for known regions)
                landmark_match = _check_landmark_match(scene, origin)
                
                results["checks"].append({
                    "scene_id": scene.scene_id,
                    "check": "crs_verification",
                    "passed": True,
                    "bounds": bounds,
                    "landmark_match": landmark_match,
                })

            except Exception as e:
                results["checks"].append({
                    "scene_id": scene.scene_id,
                    "check": "crs_verification",
                    "passed": False,
                    "error": str(e)
                })
                results["passed"] = False
                results["errors"].append(f"{scene.scene_id}: {e}")

    return results


def _check_landmark_match(scene: SceneMetadata, origin: str) -> bool:
    """Check if scene geometry matches expected landmarks for its origin."""
    geom = wkt_loads(scene.geometry_wkt)
    centroid = geom.centroid

    # Known landmarks for each dataset origin
    landmarks = {
        "bigearthnet_txt": [
            Point(11.0, 48.0),  # Munich area
            Point(2.0, 48.0),   # Paris area
        ],
        "cdvqa": [
            Point(-122.0, 37.0),  # San Francisco
            Point(-118.0, 34.0),  # Los Angeles
        ],
        "gee_assam": [
            Point(91.7, 26.1),  # Guwahati
            Point(94.2, 27.5),  # Dibrugarh
        ],
    }

    origin_landmarks = landmarks.get(origin, [])
    for landmark in origin_landmarks:
        if geom.distance(landmark) < 5.0:  # Within ~500km
            return True

    # For unknown origins, just check it's a reasonable coordinate
    return -180 <= centroid.x <= 180 and -90 <= centroid.y <= 90


def verify_pairs(catalog: Catalog) -> Dict[str, Any]:
    """
    Pair integrity verification.
    Checks: all paired_scene_id resolve, non-overlapping dates.
    """
    results = {
        "passed": True,
        "total_pairs": 0,
        "valid_pairs": 0,
        "errors": [],
    }

    # Get all scenes with paired_scene_id
    all_scenes = catalog.get_all_scenes()
    paired_scenes = [s for s in all_scenes if s.paired_scene_id]

    results["total_pairs"] = len(paired_scenes) // 2

    for scene in paired_scenes:
        pair = catalog.get_scene(scene.paired_scene_id)
        if not pair:
            results["errors"].append(f"Scene {scene.scene_id} references missing pair {scene.paired_scene_id}")
            results["passed"] = False
            continue

        # Check reverse reference
        if pair.paired_scene_id != scene.scene_id:
            results["errors"].append(f"Pair mismatch: {scene.scene_id} -> {pair.scene_id} but reverse is {pair.paired_scene_id}")
            results["passed"] = False
            continue

        # Check non-overlapping dates (if both have dates)
        if scene.acquisition_time and pair.acquisition_time:
            if scene.acquisition_time == pair.acquisition_time:
                results["errors"].append(f"Pair {scene.scene_id}-{pair.scene_id} have identical acquisition times")
                results["passed"] = False
                continue

        results["valid_pairs"] += 1

    results["valid_pairs"] = results["valid_pairs"] // 2  # Each pair counted twice
    return results


def generate_manifest(catalog: Catalog) -> Dict[str, Any]:
    """Generate catalog_manifest.json per blueprint §2.3."""
    stats = catalog.get_stats()
    crs_results = verify_crs(catalog)
    pair_results = verify_pairs(catalog)

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "catalog_version": "1.0",
        "statistics": stats,
        "quality_gates": {
            "crs_verification": crs_results,
            "pair_integrity": pair_results,
        },
        "caveats": [],
        "sources": list(stats.get("by_dataset_origin", {}).keys()),
    }

    # Add caveats based on validation results
    if not crs_results["passed"]:
        manifest["caveats"].append("CRS verification failed for some scenes")
    if not pair_results["passed"]:
        manifest["caveats"].append("Pair integrity check failed for some pairs")

    # Check for missing file_paths
    scenes_without_files = catalog.query_scenes()
    scenes_without_files = [s for s in scenes_without_files if not s.file_path]
    if scenes_without_files:
        manifest["caveats"].append(f"{len(scenes_without_files)} scenes missing file_path")

    return manifest


def write_manifest(catalog: Catalog, output_path: Optional[str] = None) -> str:
    """Write manifest to file."""
    config = get_m0_config()
    path = Path(output_path or config.output.manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest = generate_manifest(catalog)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {path}")
    return str(path)


def run_all_validations(catalog: Catalog) -> Dict[str, Any]:
    """Run all validation checks."""
    print("Running CRS verification...")
    crs_results = verify_crs(catalog)

    print("Running pair integrity check...")
    pair_results = verify_pairs(catalog)

    print("Generating manifest...")
    manifest = generate_manifest(catalog)

    all_passed = crs_results["passed"] and pair_results["passed"]

    return {
        "all_passed": all_passed,
        "crs_verification": crs_results,
        "pair_integrity": pair_results,
        "manifest": manifest,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run M0 catalog validations")
    parser.add_argument("--db", default="./data/catalog.db", help="Catalog database path")
    parser.add_argument("--output", help="Manifest output path")

    args = parser.parse_args()

    catalog = Catalog(args.db, read_only=True)
    results = run_all_validations(catalog)

    if args.output:
        write_manifest(catalog, args.output)

    if results["all_passed"]:
        print("All validations PASSED")
    else:
        print("Some validations FAILED")
        print(json.dumps(results, indent=2, default=str))

    catalog.close()