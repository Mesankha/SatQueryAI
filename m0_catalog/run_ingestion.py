"""
M0 Catalog - Ingestion CLI Entry Point
Run: python -m m0_catalog.run_ingestion
"""
import argparse
import sys
from pathlib import Path

from .catalog import Catalog
from .ingest_bigearthnet import ingest_bigearthnet
from .ingest_cdvqa import ingest_cdvqa
from .ingest_gee import ingest_gee_assam
from .validate import run_all_validations
from .manifest import write_manifest
from .generate_lora_data import generate_lora_training_data
from .config import get_m0_config


def main():
    parser = argparse.ArgumentParser(description="M0 Catalog Ingestion Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Ingest all
    ingest_parser = subparsers.add_parser("ingest", help="Run full ingestion pipeline")
    ingest_parser.add_argument("--db", default="./data/catalog.db", help="Catalog database path")
    ingest_parser.add_argument("--skip-gee", action="store_true", help="Skip GEE ingestion")
    ingest_parser.add_argument("--max-bigearthnet", type=int, help="Max BigEarthNet patches")
    ingest_parser.add_argument("--max-cdvqa", type=int, help="Max CDVQA pairs")
    ingest_parser.add_argument("--max-gee", type=int, help="Max GEE scenes")

    # Ingest individual sources
    be_parser = subparsers.add_parser("ingest-bigearthnet", help="Ingest BigEarthNet only")
    be_parser.add_argument("--db", default="./data/catalog.db")
    be_parser.add_argument("--source", default="./data/raw/bigearthnet")
    be_parser.add_argument("--max-patches", type=int)

    cdvqa_parser = subparsers.add_parser("ingest-cdvqa", help="Ingest CDVQA only")
    cdvqa_parser.add_argument("--db", default="./data/catalog.db")
    cdvqa_parser.add_argument("--source", default="./data/raw/cdvqa")
    cdvqa_parser.add_argument("--max-pairs", type=int)

    gee_parser = subparsers.add_parser("ingest-gee", help="Ingest GEE Assam only")
    gee_parser.add_argument("--db", default="./data/catalog.db")
    gee_parser.add_argument("--project-id", help="GEE project ID")
    gee_parser.add_argument("--max-scenes", type=int, default=100)

    # Validate
    validate_parser = subparsers.add_parser("validate", help="Run validation checks")
    validate_parser.add_argument("--db", default="./data/catalog.db")

    # Manifest
    manifest_parser = subparsers.add_parser("manifest", help="Generate/print manifest")
    manifest_parser.add_argument("--db", default="./data/catalog.db")
    manifest_parser.add_argument("--output", help="Output manifest path")
    manifest_parser.add_argument("--print", action="store_true", help="Print summary")

    # LoRA data
    lora_parser = subparsers.add_parser("generate-lora", help="Generate LoRA training data")
    lora_parser.add_argument("--db", default="./data/catalog.db")
    lora_parser.add_argument("--output-dir", default="./data/bigearthnet_lora")
    lora_parser.add_argument("--max-samples", type=int, default=5000)
    lora_parser.add_argument("--create-dummy", action="store_true", help="Create dummy data")

    # Stats
    stats_parser = subparsers.add_parser("stats", help="Print catalog statistics")
    stats_parser.add_argument("--db", default="./data/catalog.db")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "ingest":
            return run_full_ingestion(args)
        elif args.command == "ingest-bigearthnet":
            return run_bigearthnet_ingestion(args)
        elif args.command == "ingest-cdvqa":
            return run_cdvqa_ingestion(args)
        elif args.command == "ingest-gee":
            return run_gee_ingestion(args)
        elif args.command == "validate":
            return run_validation(args)
        elif args.command == "manifest":
            return run_manifest(args)
        elif args.command == "generate-lora":
            return run_lora_generation(args)
        elif args.command == "stats":
            return run_stats(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def run_full_ingestion(args) -> int:
    """Run complete ingestion pipeline."""
    print("=" * 60)
    print("M0 CATALOG - FULL INGESTION PIPELINE")
    print("=" * 60)

    catalog = Catalog(args.db, read_only=False)
    config = get_m0_config()

    total_ingested = 0

    # BigEarthNet
    if config.bigearthnet.enabled:
        print("\n[1/3] Ingesting BigEarthNet...")
        try:
            count = ingest_bigearthnet(
                catalog,
                config.bigearthnet.source_path,
                max_patches=args.max_bigearthnet,
            )
            total_ingested += count
        except Exception as e:
            print(f"BigEarthNet ingestion failed: {e}")

    # CDVQA
    if config.cdvqa.enabled:
        print("\n[2/3] Ingesting CDVQA...")
        try:
            count = ingest_cdvqa(
                catalog,
                config.cdvqa.source_path,
                max_pairs=args.max_cdvqa,
            )
            total_ingested += count
        except Exception as e:
            print(f"CDVQA ingestion failed: {e}")

    # GEE Assam
    if config.gee_assam.enabled and not args.skip_gee:
        print("\n[3/3] Ingesting GEE Assam...")
        try:
            count = ingest_gee_assam(
                catalog,
                project_id=config.gee_assam.project_id,
                region=config.gee_assam.region,
                date_range=config.gee_assam.date_range,
                max_scenes=args.max_gee,
            )
            total_ingested += count
        except Exception as e:
            print(f"GEE ingestion failed: {e}")
    elif args.skip_gee:
        print("\n[3/3] Skipping GEE (--skip-gee flag)")

    catalog.close()

    print(f"\nTotal scenes ingested: {total_ingested}")

    # Run validations
    print("\nRunning validations...")
    catalog = Catalog(args.db, read_only=True)
    results = run_all_validations(catalog)
    catalog.close()

    # Generate manifest
    print("\nGenerating manifest...")
    catalog = Catalog(args.db, read_only=True)
    write_manifest(catalog)
    catalog.close()

    # Generate LoRA data
    if config.lora_dataset.enabled:
        print("\nGenerating LoRA training data...")
        catalog = Catalog(args.db, read_only=True)
        generate_lora_training_data(catalog, config.lora_dataset.output_dir, config.lora_dataset.max_samples)
        catalog.close()

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    return 0 if results["all_passed"] else 1


def run_bigearthnet_ingestion(args) -> int:
    catalog = Catalog(args.db, read_only=False)
    config = get_m0_config()
    count = ingest_bigearthnet(catalog, args.source, max_patches=args.max_patches)
    catalog.close()
    print(f"Ingested {count} BigEarthNet scenes")
    return 0


def run_cdvqa_ingestion(args) -> int:
    catalog = Catalog(args.db, read_only=False)
    count = ingest_cdvqa(catalog, args.source, max_pairs=args.max_pairs)
    catalog.close()
    print(f"Ingested {count} CDVQA scenes")
    return 0


def run_gee_ingestion(args) -> int:
    catalog = Catalog(args.db, read_only=False)
    count = ingest_gee_assam(
        catalog,
        project_id=args.project_id,
        max_scenes=args.max_scenes,
    )
    catalog.close()
    print(f"Ingested {count} GEE scenes")
    return 0


def run_validation(args) -> int:
    catalog = Catalog(args.db, read_only=True)
    results = run_all_validations(catalog)
    catalog.close()

    if results["all_passed"]:
        print("All validations PASSED")
        return 0
    else:
        print("Some validations FAILED")
        import json
        print(json.dumps(results, indent=2, default=str))
        return 1


def run_manifest(args) -> int:
    catalog = Catalog(args.db, read_only=True)
    if args.output:
        write_manifest(catalog, args.output)
    if args.print:
        from .manifest import print_manifest_summary
        print_manifest_summary(args.output)
    catalog.close()
    return 0


def run_lora_generation(args) -> int:
    if args.create_dummy:
        from .generate_lora_data import create_dummy_bigearthnet_lora
        create_dummy_bigearthnet_lora(args.output_dir, args.max_samples)
    else:
        catalog = Catalog(args.db, read_only=True)
        generate_lora_training_data(catalog, args.output_dir, args.max_samples)
        catalog.close()
    return 0


def run_stats(args) -> int:
    catalog = Catalog(args.db, read_only=True)
    stats = catalog.get_stats()
    catalog.close()

    print("Catalog Statistics:")
    print(f"  Total Scenes: {stats['total_scenes']}")
    print(f"  By Origin: {stats['by_dataset_origin']}")
    print(f"  By Sensor: {stats['by_sensor']}")
    print(f"  By Modality: {stats['by_modality']}")
    print(f"  Paired Scenes: {stats['paired_scenes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())