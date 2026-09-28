"""
Populate M0 catalog with real Sentinel scenes from Planetary Computer
for the 7 Assam hackathon AOIs + existing fixture scenes.

Run: python populate_assam_scenes.py
"""
import sys
from pathlib import Path

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from m5_controller.m2_adapter import populate_m0_from_planetary

# 7 Assam AOIs (bbox: min_lon, min_lat, max_lon, max_lat)
ASSAM_AOIS = {
    "guwahati":  (91.55, 26.05, 91.85, 26.25),
    "silchar":   (92.72, 24.75, 92.87, 24.88),
    "barpeta":   (90.92, 26.28, 91.08, 26.40),
    "tezpur":    (92.72, 26.58, 92.88, 26.70),
    "majuli":    (93.80, 26.85, 94.30, 27.15),
    "jorhat":    (94.10, 26.68, 94.30, 26.82),
    "dibrugarh": (94.85, 27.42, 95.00, 27.55),
}

DATE_RANGE = ("2023-01-01", "2024-12-31")
MAX_SCENES_PER_AOI = 20  # 20 S2 + 20 S1 = ~40 per city, ~280 total

def main():
    total = 0
    for city, bbox in ASSAM_AOIS.items():
        print(f"\n--- Populating {city.title()} (bbox={bbox}) ---")
        try:
            count = populate_m0_from_planetary(
                bbox=bbox,
                date_range=DATE_RANGE,
                label=city,
                db_path="./data/catalog.db",
                max_scenes=MAX_SCENES_PER_AOI,
            )
            print(f"  Added {count} scenes for {city.title()}")
            total += count
        except Exception as e:
            print(f"  ERROR for {city}: {e}")

    print(f"\n=== Total scenes added: {total} ===")

    # Print catalog stats
    from m0_catalog.catalog import get_catalog
    cat = get_catalog(read_only=True)
    stats = cat.get_stats()
    cat.close()
    print(f"Catalog total: {stats['total_scenes']}")
    print(f"By sensor: {stats['by_sensor']}")
    print(f"By origin: {stats['by_dataset_origin']}")

if __name__ == "__main__":
    main()
