"""
M0 Manifest Generation
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .catalog import Catalog
from .validate import generate_manifest
from .config import get_m0_config


def write_manifest(catalog: Catalog, output_path: Optional[str] = None) -> str:
    """Write catalog manifest to JSON file."""
    config = get_m0_config()
    path = Path(output_path or config.output.manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest = generate_manifest(catalog)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {path}")
    return str(path)


def read_manifest(manifest_path: Optional[str] = None) -> Dict[str, Any]:
    """Read manifest from JSON file."""
    config = get_m0_config()
    path = Path(manifest_path or config.output.manifest_path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path) as f:
        return json.load(f)


def print_manifest_summary(manifest_path: Optional[str] = None):
    """Print human-readable manifest summary."""
    manifest = read_manifest(manifest_path)

    print("=" * 60)
    print("CATALOG MANIFEST SUMMARY")
    print("=" * 60)
    print(f"Generated: {manifest.get('generated_at', 'unknown')}")
    print(f"Version: {manifest.get('catalog_version', 'unknown')}")
    print(f"Sources: {', '.join(manifest.get('sources', []))}")
    print()

    stats = manifest.get("statistics", {})
    print(f"Total Scenes: {stats.get('total_scenes', 0)}")
    print(f"By Origin: {stats.get('by_dataset_origin', {})}")
    print(f"By Sensor: {stats.get('by_sensor', {})}")
    print(f"By Modality: {stats.get('by_modality', {})}")
    print(f"Paired Scenes: {stats.get('paired_scenes', 0)}")
    print()

    crs = manifest.get("quality_gates", {}).get("crs_verification", {})
    pairs = manifest.get("quality_gates", {}).get("pair_integrity", {})
    print(f"CRS Verification: {'PASS' if crs.get('passed') else 'FAIL'}")
    print(f"Pair Integrity: {'PASS' if pairs.get('passed') else 'FAIL'}")
    print()

    caveats = manifest.get("caveats", [])
    if caveats:
        print("Caveats:")
        for c in caveats:
            print(f"  - {c}")
    else:
        print("No caveats.")

    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="M0 Manifest utilities")
    parser.add_argument("--manifest", help="Manifest file path")
    parser.add_argument("--db", default="./data/catalog.db", help="Catalog database path")
    parser.add_argument("--output", help="Output manifest path")

    args = parser.parse_args()

    if args.output:
        catalog = Catalog(args.db, read_only=True)
        write_manifest(catalog, args.output)
        catalog.close()
    else:
        print_manifest_summary(args.manifest)