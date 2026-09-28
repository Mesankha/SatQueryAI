#!/usr/bin/env python
"""
build_catalog.py - CLI wrapper for M0 ingestion from Planetary Computer.

Usage:
    python scripts/build_catalog.py --aois config/aois_tier1.yaml [--dry-run]
"""
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from m0_ingest.pc_ingest import main

if __name__ == "__main__":
    main()