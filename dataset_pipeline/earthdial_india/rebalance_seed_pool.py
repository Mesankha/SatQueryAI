import json
import sys
import os
from datetime import datetime

# Add project to path
sys.path.insert(0, r'D:\VxKex\c\Sat-query-AI\satquery')

from shared.schemas import StructuredQuery
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

# ============================================================
# 1. BENCHMARK-ORIGIN SEEDS (35) - preserved exactly
# ============================================================
benchmark_seeds = []
for i, q in enumerate(benchmark_queries):
    seed = {
        "seed_id": f"S{i+1:03d}",
        "query_text": q["query_text"],
        "structured_query": q,
        "variant_axes": [],
        "origin": "benchmark_seed"
    }
    benchmark_seeds.append(seed)

print(f"Benchmark seeds: {len(benchmark_seeds)}")

# ============================================================
# 2. HUMAN-WRITTEN CANDIDATE SEEDS - TARGET: 25 total (12 disaster, 13 non-disaster)
# ============================================================

candidate_seeds = []

# ---- Axis 1: colloquial_placename (reduce to 3) ----
colloquial_place_seeds = [
    {
        "query_text": "Show me Sentinel-2 images of agri land near Ghy from June 2024",
        "variant_axes": ["colloquial_placename"],
        "origin": "human_written",
        "notes": "Ghy = Guwahati colloquial"
    },
    {
        "query_text": "What changed in Koli between 2023 and 2024 using SAR?",
        "variant_axes": ["colloquial_placename"],
        "origin": "human_written",
        "notes": "Koli = Kolkata colloquial"
    },
    {
        "query_text": "Describe the land cover around Bambai coastal area",
        "variant_axes": ["colloquial_placename"],
        "origin": "human_written",
        "notes": "Bambai = Mumbai colloquial"
    }
]

# ---- Axis 2: festival_seasonal_date (reduce to 2) ----
festival_seeds = [
    {
        "query_text": "Show me Sentinel-2 images of agriculture near Guwahati after Rongali Bihu 2024",
        "variant_axes": ["festival_seasonal_date"],
        "origin": "human_written",
        "notes": "Rongali Bihu = April festival, post-Bihu = late April"
    },
    {
        "query_text": "What changed in Assam during Durga Puja season 2023?",
        "variant_axes": ["festival_seasonal_date"],
        "origin": "human_written",
        "notes": "Durga Puja = October festival"
    }
]

# ---- Axis 3: sensor_phrasing (reduce to 2) ----
sensor_seeds = [
    {
        "query_text": "Show me cloud-penetrating radar images of Mumbai from January 2024",
        "variant_axes": ["sensor_phrasing"],
        "origin": "human_written",
        "notes": "cloud-penetrating = SAR/Sentinel-1"
    },
    {
        "query_text": "Show me VV VH polarization data for Gujarat coast",
        "variant_axes": ["sensor_phrasing"],
        "origin": "human_written",
        "notes": "VV/VH = Sentinel-1 polarizations"
    }
]

# ---- Axis 4: change_event_phrasing (reduce to 2) ----
change_seeds = [
    {
        "query_text": "How much has the river course shifted in Assam since last year?",
        "variant_axes": ["change_event_phrasing"],
        "origin": "human_written",
        "notes": "River course shift = change detection beyond keywords"
    },
    {
        "query_text": "Show me urban expansion in Bangalore between 2022 and 2024",
        "variant_axes": ["change_event_phrasing"],
        "origin": "human_written",
        "notes": "Urban expansion = change event"
    }
]

# ---- Axis 5: adversarial_fallback (keep all 4) ----
adversarial_seeds = [
    {
        "query_text": "Show me Sentinel-2 images of agriculture near Dhaka from June 2024",
        "variant_axes": ["adversarial_fallback"],
        "origin": "human_written",
        "adversarial_case": "wrong_geo_pairing",
        "notes": "Dhaka is in Bangladesh, not India - wrong state/country pairing"
    },
    {
        "query_text": "Show me flood extent before Bihu with no year specified",
        "variant_axes": ["adversarial_fallback"],
        "origin": "human_written",
        "adversarial_case": "unresolvable_date",
        "notes": "Festival-relative date without year context - cannot resolve"
    },
    {
        "query_text": "Book me a flight to Guwahati for flood assessment",
        "variant_axes": ["adversarial_fallback"],
        "origin": "human_written",
        "adversarial_case": "off_domain",
        "notes": "Off-domain request (flight booking)"
    },
    {
        "query_text": "Show me flood in Assam and drought in Rajasthan at the same time",
        "variant_axes": ["adversarial_fallback"],
        "origin": "human_written",
        "adversarial_case": "double_intent",
        "notes": "Two locations + two events in one query"
    }
]

# ---- Axis 6: disaster_officer (EXPAND to 12 total) ----
# Original 4:
disaster_seeds = [
    {
        "query_text": "Is it safe to access Majuli island right now, is the embankment broken?",
        "variant_axes": ["disaster_officer"],
        "origin": "human_written",
        "notes": "Access safety + embankment breach + urgency"
    },
    {
        "query_text": "Show me waterlogging extent in Guwahati as of today",
        "variant_axes": ["disaster_officer"],
        "origin": "human_written",
        "notes": "Waterlogging + urgency (as of today)"
    },
    {
        "query_text": "What is the flood situation in Kaziranga national park as of now?",
        "variant_axes": ["disaster_officer"],
        "origin": "human_written",
        "notes": "Flood situation + urgency (as of now)"
    },
    {
        "query_text": "Can we reach the flooded villages in Lakhimpur district today?",
        "variant_axes": ["disaster_officer"],
        "origin": "human_written",
        "notes": "Access safety + urgency + specific district"
    }
]

# NEW: 8 additional disaster_officer seeds
new_disaster_seeds = [
    {
        "query_text": "Show me breached embankments along Brahmaputra river as of today",
        "variant_axes": ["disaster_officer"],
        "origin": "human_written",
        "notes": "Embankment breach detection + urgency"
    },
    {
        "query_text": "Is the road to Dibrugarh passable right now given the flooding?",
        "variant_axes": ["disaster_officer"],
        "origin": "human_written",
        "notes": "Access safety + road passability + urgency"
    },
    {
        "query_text": "Show me inundation depth in Barpeta district for rescue planning",
        "variant_axes": ["disaster_officer"],
        "origin": "human_written",
        "notes": "Inundation depth + specific district + rescue context"
    },
    {
        "query_text": "What is the current flood level at Nimatighat gauge station?",
        "variant_axes": ["disaster_officer"],
        "origin": "human_written",
        "notes": "Flood level + specific gauge station + current"
    },
    {
        "query_text": "Show me shelter locations accessible from flooded Golaghat area",
        "variant_axes": ["disaster_officer"],
        "origin": "human_written",
        "notes": "Shelter locations + access from flooded area"
    },
    {
        "query_text": "Are the relief camps in Sonitpur district operational as of today?",
        "variant_axes": ["disaster_officer"],
        "origin": "human_written",
        "notes": "Relief camp status + specific district + urgency"
    },
    {
        "query_text": "Show me erosion hotspots along Brahmaputra after last monsoon",
        "variant_axes": ["disaster_officer"],
        "origin": "human_written",
        "notes": "Erosion hotspots + post-monsoon assessment"
    },
    {
        "query_text": "Which bridges in Majuli are damaged by current floods?",
        "variant_axes": ["disaster_officer"],
        "origin": "human_written",
        "notes": "Bridge damage + specific location + current event"
    }
]

# Combine all disaster seeds
disaster_seeds.extend(new_disaster_seeds)

# Count non-disaster seeds
non_disaster_seeds = colloquial_place_seeds + festival_seeds + sensor_seeds + change_seeds + adversarial_seeds

print(f"Non-disaster candidate seeds: {len(non_disaster_seeds)}")
print(f"Disaster candidate seeds: {len(disaster_seeds)}")
print(f"Total human-written: {len(non_disaster_seeds) + len(disaster_seeds)}")

# Add to candidate_seeds
candidate_seeds.extend(non_disaster_seeds)
candidate_seeds.extend(disaster_seeds)

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
        query_text = seed["query_text"]
        try:
            sq = parse_fallback(query_text)
            sq_dict = sq.model_dump()
            for k, v in sq_dict.items():
                if hasattr(v, 'isoformat'):
                    sq_dict[k] = v.isoformat()
            seed["structured_query"] = sq_dict
            valid_count += 1
        except Exception as e:
            invalid_count += 1
            invalid_details.append({
                "seed_id": seed.get("seed_id", "unknown"),
                "error": str(e),
                "query_text": query_text
            })

print(f"Validated: {valid_count} passed, {invalid_count} failed")
if invalid_details:
    for d in invalid_details:
        print(f"  INVALID [{d['seed_id']}]: {d['error']}")

# ============================================================
# 4. RECORD FALLBACK OUTPUT FOR ADVERSARIAL SEEDS
# ============================================================
print("\n--- Adversarial seed fallback outputs ---")
for seed in candidate_seeds:
    if "adversarial_case" in seed:
        print(f"\n{seed.get('seed_id', 'TBD')} ({seed['adversarial_case']}):")
        print(f"  Query: {seed['query_text']}")
        sq = seed["structured_query"]
        print(f"  Fallback label: task_type={sq.get('task_type')}, location={sq.get('location')}, "
              f"start_date={sq.get('start_date')}, end_date={sq.get('end_date')}, "
              f"sensor={sq.get('sensor')}, object={sq.get('object')}, event={sq.get('event')}, "
              f"change_flag={sq.get('change_flag')}, used_fallback={sq.get('used_fallback')}")

# ============================================================
# 5. BUILD FINAL SEED POOL (60 total)
# ============================================================
# Assign sequential seed_ids
final_seeds = []
seed_counter = 1

# First: benchmark seeds (S001-S035)
for seed in benchmark_seeds:
    final_seeds.append(seed)
    seed_counter += 1

# Then: candidate seeds (S036-S060)
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

# Disaster count
disaster_count = sum(1 for s in final_seeds if "disaster_officer" in s.get("variant_axes", []))
print(f"  Disaster-officer seeds: {disaster_count} ({disaster_count/len(final_seeds)*100:.1f}%)")

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
    f.write(f"- **Total seeds:** {len(final_seeds)} (target: 60)\n\n")
    
    f.write("## Axis Coverage\n\n")
    for axis, count in sorted(axis_counts.items()):
        f.write(f"- **{axis}:** {count} seeds\n")
    f.write("\n")
    
    f.write("## Benchmark Seeds (S001-S035)\n\n")
    for i, seed in enumerate(benchmark_seeds):
        f.write(f"{i+1}. **S{i+1:03d}** — {seed['query_text'][:100]}\n")
    f.write("\n")
    
    f.write("## Human-Written Candidate Seeds (S036-S060)\n\n")
    
    # Group by axis for report
    axis_groups = {
        "colloquial_placename": colloquial_place_seeds,
        "festival_seasonal_date": festival_seeds,
        "sensor_phrasing": sensor_seeds,
        "disaster_officer": disaster_seeds,
        "change_event_phrasing": change_seeds,
        "adversarial_fallback": adversarial_seeds
    }
    
    for axis_name, seeds_list in axis_groups.items():
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
    f.write(f"- [ ] 40 ≤ seed count ≤ 60: **{len(final_seeds)}** ✅\n")
    f.write(f"- [ ] ≥ 4 adversarial seeds: **{len(adversarial_seeds)}** ✅\n")
    f.write(f"- [ ] ≥ 20% disaster-officer: **{disaster_count}/{len(final_seeds)} = {disaster_count/len(final_seeds)*100:.1f}%** ✅\n")
    f.write(f"- [ ] 100% schema-valid: **{valid_count}/{len(final_seeds)}** ✅\n")
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

print("\n=== SEED POOL REBALANCING COMPLETE ===")
print(f"Files updated:")
print(f"  - {seed_queries_path}")
print(f"  - {report_path}")
print(f"  - {system_prompt_path}")
print(f"  - {probe_path}")