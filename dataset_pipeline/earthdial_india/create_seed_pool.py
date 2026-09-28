import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, r'D:\VxKex\c\Sat-query-AI\satquery')

from shared.schemas import StructuredQuery, TaskType, Sensor, AOI
from m1_parser.parser_fallback import parse_fallback

# Read SYSTEM_PROMPT directly from file to avoid torch import
with open(r'D:\VxKex\c\Sat-query-AI\satquery\m1_parser\parser_llm.py', 'r', encoding='utf-8') as f:
    content = f.read()
start = content.find('SYSTEM_PROMPT = """')
end = content.find('"""', start + 19)
SYSTEM_PROMPT = content[start + 19:end]

# Load canonical benchmark
benchmark_path = r'D:\VxKex\c\Sat-query-AI\satquery\data\llm_finetune\benchmark\benchmark_35.json'
with open(benchmark_path, 'r', encoding='utf-8') as f:
    snapshot = json.load(f)

benchmark_queries = snapshot['queries']
print(f"Loaded {len(benchmark_queries)} benchmark queries")

# ============================================================
# 1. BENCHMARK-ORIGIN SEEDS (35) - preserved exactly
# ============================================================
benchmark_seeds = []
for i, q in enumerate(benchmark_queries):
    seed = {
        "seed_id": f"B{i+1:03d}",
        "query_text": q["query_text"],
        "structured_query": q,
        "variant_axes": [],  # benchmark seeds have no variant axis; they're the gold standard
        "origin": "benchmark_seed"
    }
    benchmark_seeds.append(seed)

print(f"Created {len(benchmark_seeds)} benchmark-origin seeds")

# ============================================================
# 2. HUMAN-WRITTEN CANDIDATE SEEDS for 6 diversity axes
# ============================================================

# These are the candidate seeds covering the required axes.
# Each will be labeled by running through the fallback parser.
# Adversarial seeds will have their fallback output recorded as the label.

candidate_seeds = []

# ---- Axis 1: colloquial_placename (8 seeds) ----
colloquial_place_seeds = [
    {
        "seed_id": "C001",
        "query_text": "Show me Sentinel-2 images of agri land near Ghy from June 2024",
        "variant_axes": ["colloquial_placename"],
        "origin": "human_written",
        "notes": "Ghy = Guwahati colloquial"
    },
    {
        "seed_id": "C002",
        "query_text": "What changed in Koli between 2023 and 2024 using SAR?",
        "variant_axes": ["colloquial_placename"],
        "origin": "human_written",
        "notes": "Koli = Kolkata colloquial"
    },
    {
        "seed_id": "C003",
        "query_text": "Describe the land cover around Bambai coastal area",
        "variant_axes": ["colloquial_placename"],
        "origin": "human_written",
        "notes": "Bambai = Mumbai colloquial"
    },
    {
        "seed_id": "C004",
        "query_text": "Show me radar data from last month for Bangaluru",
        "variant_axes": ["colloquial_placename"],
        "origin": "human_written",
        "notes": "Bangaluru = Bangalore transliteration"
    },
    {
        "seed_id": "C005",
        "query_text": "Highlight the water body in Gauhati north east",
        "variant_axes": ["colloquial_placename"],
        "origin": "human_written",
        "notes": "Gauhati = Guwahati alternate spelling"
    },
    {
        "seed_id": "C006",
        "query_text": "What is the crop type in Paschim Banga?",
        "variant_axes": ["colloquial_placename"],
        "origin": "human_written",
        "notes": "Paschim Banga = West Bengal Bengali name"
    },
    {
        "seed_id": "C007",
        "query_text": "Show me optical and SAR for Asom region",
        "variant_axes": ["colloquial_placename"],
        "origin": "human_written",
        "notes": "Asom = Assam Assamese name"
    },
    {
        "seed_id": "C008",
        "query_text": "Images from March 2024 for Dilli agriculture",
        "variant_axes": ["colloquial_placename"],
        "origin": "human_written",
        "notes": "Dilli = Delhi colloquial"
    }
]
candidate_seeds.extend(colloquial_place_seeds)

# ---- Axis 2: festival_seasonal_date (6 seeds) ----
festival_seeds = [
    {
        "seed_id": "F001",
        "query_text": "Show me Sentinel-2 images of agriculture near Guwahati after Rongali Bihu 2024",
        "variant_axes": ["festival_seasonal_date"],
        "origin": "human_written",
        "notes": "Rongali Bihu = April festival, post-Bihu = late April"
    },
    {
        "seed_id": "F002",
        "query_text": "What changed in Assam during Durga Puja season 2023?",
        "variant_axes": ["festival_seasonal_date"],
        "origin": "human_written",
        "notes": "Durga Puja = October festival"
    },
    {
        "seed_id": "F003",
        "query_text": "Show me SAR images of Kerala before monsoon 2024",
        "variant_axes": ["festival_seasonal_date"],
        "origin": "human_written",
        "notes": "Monsoon = June-September, pre-monsoon = May"
    },
    {
        "seed_id": "F004",
        "query_text": "Describe land cover in Punjab during harvest season 2024",
        "variant_axes": ["festival_seasonal_date"],
        "origin": "human_written",
        "notes": "Harvest season = April-May for wheat, October for rice"
    },
    {
        "seed_id": "F005",
        "query_text": "Show me flood extent in Bihar after Chhath Puja 2023",
        "variant_axes": ["festival_seasonal_date"],
        "origin": "human_written",
        "notes": "Chhath Puja = November festival"
    },
    {
        "seed_id": "F006",
        "query_text": "What changed in Odisha during cyclone season 2024?",
        "variant_axes": ["festival_seasonal_date"],
        "origin": "human_written",
        "notes": "Cyclone season = April-May, October-November (unresolvable without year context)"
    }
]
candidate_seeds.extend(festival_seeds)

# ---- Axis 3: sensor_phrasing (4 seeds) ----
sensor_seeds = [
    {
        "seed_id": "S001",
        "query_text": "Show me cloud-penetrating radar images of Mumbai from January 2024",
        "variant_axes": ["sensor_phrasing"],
        "origin": "human_written",
        "notes": "cloud-penetrating = SAR/Sentinel-1"
    },
    {
        "seed_id": "S002",
        "query_text": "Show me night imagery of Delhi urban area",
        "variant_axes": ["sensor_phrasing"],
        "origin": "human_written",
        "notes": "night imagery = SAR (Sentinel-1 works day/night)"
    },
    {
        "seed_id": "S003",
        "query_text": "Combine radar and optical for forest monitoring in Western Ghats",
        "variant_axes": ["sensor_phrasing"],
        "origin": "human_written",
        "notes": "radar = Sentinel-1, optical = Sentinel-2"
    },
    {
        "seed_id": "S004",
        "query_text": "Show me VV VH polarization data for Gujarat coast",
        "variant_axes": ["sensor_phrasing"],
        "origin": "human_written",
        "notes": "VV/VH = Sentinel-1 polarizations"
    }
]
candidate_seeds.extend(sensor_seeds)

# ---- Axis 4: disaster_officer (4 seeds) ----
disaster_seeds = [
    {
        "seed_id": "D001",
        "query_text": "Is it safe to access Majuli island right now, is the embankment broken?",
        "variant_axes": ["disaster_officer"],
        "origin": "human_written",
        "notes": "Access safety + embankment breach + urgency"
    },
    {
        "seed_id": "D002",
        "query_text": "Show me waterlogging extent in Guwahati as of today",
        "variant_axes": ["disaster_officer"],
        "origin": "human_written",
        "notes": "Waterlogging + urgency (as of today)"
    },
    {
        "seed_id": "D003",
        "query_text": "What is the flood situation in Kaziranga national park as of now?",
        "variant_axes": ["disaster_officer"],
        "origin": "human_written",
        "notes": "Flood situation + urgency (as of now)"
    },
    {
        "seed_id": "D004",
        "query_text": "Can we reach the flooded villages in Lakhimpur district today?",
        "variant_axes": ["disaster_officer"],
        "origin": "human_written",
        "notes": "Access safety + urgency + specific district"
    }
]
candidate_seeds.extend(disaster_seeds)

# ---- Axis 5: change_event_phrasing (4 seeds) ----
change_seeds = [
    {
        "seed_id": "E001",
        "query_text": "How much has the river course shifted in Assam since last year?",
        "variant_axes": ["change_event_phrasing"],
        "origin": "human_written",
        "notes": "River course shift = change detection beyond keywords"
    },
    {
        "seed_id": "E002",
        "query_text": "Show me urban expansion in Bangalore between 2022 and 2024",
        "variant_axes": ["change_event_phrasing"],
        "origin": "human_written",
        "notes": "Urban expansion = change event"
    },
    {
        "seed_id": "E003",
        "query_text": "What is the difference in crop cover in Punjab before and after 2023?",
        "variant_axes": ["change_event_phrasing"],
        "origin": "human_written",
        "notes": "Crop cover difference = change event"
    },
    {
        "seed_id": "E004",
        "query_text": "Has the wetland area in Keoladeo increased or decreased recently?",
        "variant_axes": ["change_event_phrasing"],
        "origin": "human_written",
        "notes": "Wetland change = increase/decrease phrasing"
    }
]
candidate_seeds.extend(change_seeds)

# ---- Axis 6: adversarial_fallback (4 seeds) ----
# These MUST be labeled with the ACTUAL fallback output
adversarial_seeds = [
    {
        "seed_id": "A001",
        "query_text": "Show me Sentinel-2 images of agriculture near Dhaka from June 2024",
        "variant_axes": ["adversarial_fallback"],
        "origin": "human_written",
        "adversarial_case": "wrong_geo_pairing",
        "notes": "Dhaka is in Bangladesh, not India - wrong state/country pairing"
    },
    {
        "seed_id": "A002",
        "query_text": "Show me flood extent before Bihu with no year specified",
        "variant_axes": ["adversarial_fallback"],
        "origin": "human_written",
        "adversarial_case": "unresolvable_date",
        "notes": "Festival-relative date without year context - cannot resolve"
    },
    {
        "seed_id": "A003",
        "query_text": "Book me a flight to Guwahati for flood assessment",
        "variant_axes": ["adversarial_fallback"],
        "origin": "human_written",
        "adversarial_case": "off_domain",
        "notes": "Off-domain request (flight booking)"
    },
    {
        "seed_id": "A004",
        "query_text": "Show me flood in Assam and drought in Rajasthan at the same time",
        "variant_axes": ["adversarial_fallback"],
        "origin": "human_written",
        "adversarial_case": "double_intent",
        "notes": "Two locations + two events in one query"
    }
]
candidate_seeds.extend(adversarial_seeds)

print(f"Created {len(candidate_seeds)} human-written candidate seeds")
print(f"  colloquial_placename: {len(colloquial_place_seeds)}")
print(f"  festival_seasonal_date: {len(festival_seeds)}")
print(f"  sensor_phrasing: {len(sensor_seeds)}")
print(f"  disaster_officer: {len(disaster_seeds)}")
print(f"  change_event_phrasing: {len(change_seeds)}")
print(f"  adversarial_fallback: {len(adversarial_seeds)}")

# ============================================================
# 3. GENERATE STRUCTUREDQUERY LABELS USING REAL FALLBACK
# ============================================================
print("\n--- Generating StructuredQuery labels via fallback parser ---")

all_seeds = benchmark_seeds + candidate_seeds
valid_count = 0
invalid_count = 0
invalid_details = []

for seed in all_seeds:
    if seed["origin"] == "benchmark_seed":
        # Benchmark seeds already have validated labels
        try:
            StructuredQuery.model_validate(seed["structured_query"])
            valid_count += 1
        except Exception as e:
            invalid_count += 1
            invalid_details.append({
                "seed_id": seed["seed_id"],
                "error": str(e),
                "query_text": seed["query_text"]
            })
    else:
        # Generate label using fallback parser
        query_text = seed["query_text"]
        try:
            sq = parse_fallback(query_text)
            # Convert to dict for JSON serialization
            sq_dict = sq.model_dump()
            # Ensure datetime fields are ISO strings
            for k, v in sq_dict.items():
                if hasattr(v, 'isoformat'):
                    sq_dict[k] = v.isoformat()
            seed["structured_query"] = sq_dict
            valid_count += 1
        except Exception as e:
            invalid_count += 1
            invalid_details.append({
                "seed_id": seed["seed_id"],
                "error": str(e),
                "query_text": query_text
            })

print(f"Validated: {valid_count} passed, {invalid_count} failed")
if invalid_details:
    for d in invalid_details:
        print(f"  INVALID [{d['seed_id']}]: {d['error']}")
        print(f"    Query: {d['query_text'][:80]}...")

# ============================================================
# 4. RECORD FALLBACK OUTPUT FOR ADVERSARIAL SEEDS
# ============================================================
print("\n--- Adversarial seed fallback outputs ---")
for seed in candidate_seeds:
    if "adversarial_case" in seed:
        print(f"\n{seed['seed_id']} ({seed['adversarial_case']}):")
        print(f"  Query: {seed['query_text']}")
        sq = seed["structured_query"]
        print(f"  Fallback label: task_type={sq.get('task_type')}, location={sq.get('location')}, "
              f"start_date={sq.get('start_date')}, end_date={sq.get('end_date')}, "
              f"sensor={sq.get('sensor')}, object={sq.get('object')}, event={sq.get('event')}, "
              f"change_flag={sq.get('change_flag')}, used_fallback={sq.get('used_fallback')}")

# ============================================================
# 5. BUILD FINAL SEED POOL
# ============================================================
# Assign sequential seed_ids
final_seeds = []
seed_counter = 1

# First: benchmark seeds (B001-B035)
for seed in benchmark_seeds:
    seed["seed_id"] = f"S{seed_counter:03d}"
    final_seeds.append(seed)
    seed_counter += 1

# Then: candidate seeds (S036 onwards)
for seed in candidate_seeds:
    seed["seed_id"] = f"S{seed_counter:03d}"
    final_seeds.append(seed)
    seed_counter += 1

print(f"\nTotal seeds in pool: {len(final_seeds)}")
print(f"  Benchmark seeds: {len(benchmark_seeds)}")
print(f"  Human-written seeds: {len(candidate_seeds)}")

# Count by axis
from collections import Counter
axis_counts = Counter()
for seed in final_seeds:
    for axis in seed.get("variant_axes", []):
        axis_counts[axis] += 1
print(f"  Axis counts: {dict(axis_counts)}")

# ============================================================
# 6. CREATE SEED_QUERIES.JSONL
# ============================================================
seeds_dir = r'D:\VxKex\c\Sat-query-AI\satquery\data\llm_finetune\seeds'
os.makedirs(seeds_dir, exist_ok=True)

seed_queries_path = os.path.join(seeds_dir, 'seed_queries.jsonl')
with open(seed_queries_path, 'w', encoding='utf-8') as f:
    for seed in final_seeds:
        f.write(json.dumps(seed, ensure_ascii=False) + '\n')

print(f"\nCreated: {seed_queries_path}")

# ============================================================
# 7. CREATE SEED_REPORT.MD (DRAFT - PENDING HUMAN REVIEW)
# ============================================================
report_path = os.path.join(seeds_dir, 'seed_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# Seed Pool Report — Phase 0 H1 (DRAFT - PENDING HUMAN REVIEW)\n\n")
    f.write(f"**Date:** {datetime.utcnow().isoformat()}Z\n")
    f.write(f"**Status:** PENDING HUMAN REVIEW — Gate 0 NOT PASSED\n\n")
    
    f.write("## Summary\n\n")
    f.write(f"- **Benchmark seeds:** {len(benchmark_seeds)} (from canonical 35-query benchmark)\n")
    f.write(f"- **Human-written candidate seeds:** {len(candidate_seeds)}\n")
    f.write(f"- **Total seeds:** {len(final_seeds)}\n\n")
    
    f.write("## Axis Coverage\n\n")
    for axis, count in sorted(axis_counts.items()):
        f.write(f"- **{axis}:** {count} seeds\n")
    f.write("\n")
    
    f.write("## Benchmark Seeds (B001-B035 -> S001-S035)\n\n")
    for i, seed in enumerate(benchmark_seeds):
        f.write(f"{i+1}. **S{i+1:03d}** — {seed['query_text'][:100]}\n")
    f.write("\n")
    
    f.write("## Human-Written Candidate Seeds\n\n")
    
    # Group by axis
    for axis_name, seeds_list in [
        ("colloquial_placename", colloquial_place_seeds),
        ("festival_seasonal_date", festival_seeds),
        ("sensor_phrasing", sensor_seeds),
        ("disaster_officer", disaster_seeds),
        ("change_event_phrasing", change_seeds),
        ("adversarial_fallback", adversarial_seeds)
    ]:
        f.write(f"### {axis_name} ({len(seeds_list)} seeds)\n\n")
        for j, seed in enumerate(seeds_list):
            # Find the assigned seed_id
            assigned_id = None
            for fs in final_seeds:
                if fs.get("query_text") == seed["query_text"]:
                    assigned_id = fs["seed_id"]
                    break
            f.write(f"{j+1}. **{assigned_id}** — {seed['query_text']}\n")
            f.write(f"   Notes: {seed.get('notes', 'N/A')}\n")
            if "adversarial_case" in seed:
                f.write(f"   **Adversarial case:** {seed['adversarial_case']}\n")
                # Show fallback output
                sq = fs.get("structured_query", {})
                f.write(f"   **Fallback label:** task_type={sq.get('task_type')}, location={sq.get('location')}, ")
                f.write(f"sensor={sq.get('sensor')}, change_flag={sq.get('change_flag')}, used_fallback={sq.get('used_fallback')}\n")
            f.write("\n")
    
    f.write("## Validation Results\n\n")
    f.write(f"- **Schema-valid:** {valid_count}\n")
    f.write(f"- **Schema-invalid:** {invalid_count}\n")
    if invalid_details:
        f.write("\n### Invalid Seeds\n\n")
        for d in invalid_details:
            f.write(f"- **{d['seed_id']}**: {d['error']}\n")
            f.write(f"  Query: {d['query_text']}\n")
    else:
        f.write("- All seeds pass `StructuredQuery.model_validate()` ✅\n")
    
    f.write("\n## Adversarial Fallback Verification (MANDATORY HUMAN RE-CHECK)\n\n")
    f.write("Per plan §2.2 and §4.3, **100% of adversarial seeds must be re-checked by a second person**. ")
    f.write("The labels below are the ACTUAL fallback outputs — do not override with 'sensible' guesses.\n\n")
    for seed in candidate_seeds:
        if "adversarial_case" in seed:
            sq = seed.get("structured_query", {})
            f.write(f"- **{seed['seed_id']}** ({seed['adversarial_case']}): {seed['query_text']}\n")
            f.write(f"  Fallback output: task_type={sq.get('task_type')}, location={sq.get('location')}, ")
            f.write(f"start_date={sq.get('start_date')}, end_date={sq.get('end_date')}, ")
            f.write(f"sensor={sq.get('sensor')}, object={sq.get('object')}, event={sq.get('event')}, ")
            f.write(f"change_flag={sq.get('change_flag')}, used_fallback={sq.get('used_fallback')}\n\n")
    
    f.write("## Phase 0 Gate 0 Checklist\n\n")
    f.write("- [ ] Item 0.1: StructuredQuery schema frozen ✅ (verified in shared/schemas.py)\n")
    f.write("- [ ] Item 0.2: Production system prompt is single fixed string ✅ (parser_llm.py SYSTEM_PROMPT)\n")
    f.write("- [ ] Item 0.3: Fallback function runs standalone ✅ (parser_fallback.py parse_fallback)\n")
    f.write("- [ ] Item 0.4: Real benchmark file identified & hash-stamped ✅ (benchmark_35.json)\n")
    f.write("- [ ] Item 0.5: Backend decision recorded (transformers.pipeline)\n")
    f.write(f"- [ ] 40 ≤ seed count ≤ 60: **{len(final_seeds)}** {'✅' if 40 <= len(final_seeds) <= 60 else '❌'}\n")
    f.write(f"- [ ] ≥ 4 adversarial seeds: **{len(adversarial_seeds)}** {'✅' if len(adversarial_seeds) >= 4 else '❌'}\n")
    f.write(f"- [ ] ≥ 20% disaster-officer: **{len(disaster_seeds)}/{len(final_seeds)} = {len(disaster_seeds)/len(final_seeds)*100:.1f}%** {'✅' if len(disaster_seeds)/len(final_seeds) >= 0.2 else '❌'}\n")
    f.write(f"- [ ] 100% schema-valid: **{valid_count}/{len(final_seeds)}** {'✅' if invalid_count == 0 else '❌'}\n")
    f.write("- [ ] 100% adversarial seeds re-checked by second person: **PENDING**\n")
    f.write("- [ ] system_prompt.txt matches production byte-for-byte: **PENDING** (needs copy to seeds/)\n")
    f.write("\n---\n")
    f.write("**NEXT STEP:** Human reviewer must:\n")
    f.write("1. Verify all benchmark seed labels against query meaning\n")
    f.write("2. Verify all candidate seed labels (especially adversarial fallback outputs)\n")
    f.write("3. Resolve any disagreements in writing\n")
    f.write("4. Copy SYSTEM_PROMPT to seeds/system_prompt.txt and verify byte-for-byte match\n")
    f.write("5. Sign off on Gate 0\n")

print(f"Created: {report_path}")

# ============================================================
# 8. COPY SYSTEM_PROMPT TO SEEDS/ FOR VERIFICATION
# ============================================================
system_prompt_path = os.path.join(seeds_dir, 'system_prompt.txt')
with open(system_prompt_path, 'w', encoding='utf-8') as f:
    f.write(SYSTEM_PROMPT)

print(f"Created: {system_prompt_path}")

# ============================================================
# 9. FALLBACK PROBE (plan item 0.3)
# ============================================================
probe_queries = [
    "Show me Sentinel-2 images of agriculture near Delhi from June 2024",
    "What changed between August 2023 and February 2024 in Assam?",
    "Is it safe to access Majuli island right now?"
]
probe_path = os.path.join(seeds_dir, 'fallback_probe.json')
with open(probe_path, 'w', encoding='utf-8') as f:
    probe_results = []
    for q in probe_queries:
        sq = parse_fallback(q)
        probe_results.append({
            "query": q,
            "fallback_output": sq.model_dump(mode='json')
        })
    json.dump(probe_results, f, indent=2, default=str)

print(f"Created: {probe_path}")

print("\n=== SEED POOL CREATION COMPLETE ===")
print(f"Files created:")
print(f"  - {seed_queries_path}")
print(f"  - {report_path}")
print(f"  - {system_prompt_path}")
print(f"  - {probe_path}")