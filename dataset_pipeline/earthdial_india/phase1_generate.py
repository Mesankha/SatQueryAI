"""
Phase 1: Synthetic Generation & Machine Validation
Teacher LLM: NVIDIA API (Nemotron 3 Ultra)
Quota: 2,400 total pairs across 7 strata
"""
import json
import os
import sys
import random
import hashlib
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, r'D:\VxKex\c\Sat-query-AI\satquery')

from shared.schemas import StructuredQuery

# Load seeds
seed_path = r'D:\VxKex\c\Sat-query-AI\satquery\data\llm_finetune\seeds\seed_queries.jsonl'
seeds = []
with open(seed_path, 'r', encoding='utf-8') as f:
    for line in f:
        seeds.append(json.loads(line))

# Load SYSTEM_PROMPT for reference
with open(r'D:\VxKex\c\Sat-query-AI\satquery\m1_parser\parser_llm.py', 'r', encoding='utf-8') as f:
    content = f.read()
start = content.find('SYSTEM_PROMPT = """')
end = content.find('"""', start + 19)
SYSTEM_PROMPT = content[start + 19:end]

# ============================================================
# QUOTA TABLE (from build plan §3.4)
# ============================================================
# Each seed will generate paraphrases per its variant_axes
# Total target: 2,400

QUOTA_TABLE = {
    'canonical': 300,
    'colloquial_placename': 450,
    'festival_seasonal_date': 300,
    'sensor_phrasing': 200,
    'change_event_phrasing': 300,
    'disaster_officer': 450,
    'adversarial_fallback': 250,
    'multilingual_hinglish': 150,
}

# Axis instructions per stratum
AXIS_INSTRUCTIONS = {
    'canonical': "Produce paraphrases using gazetteer-form place names and plain dates. Keep the same intent and StructuredQuery exactly.",
    'colloquial_placename': "Use colloquial/transliterated spellings of place names (e.g., Ghy for Guwahati, Koli for Kolkata, Bambai for Mumbai). Do NOT use gazetteer forms in the query text. The StructuredQuery location must remain the canonical gazetteer form.",
    'festival_seasonal_date': "Use festival-relative or seasonal date expressions (e.g., 'after Rongali Bihu', 'during Durga Puja', 'before monsoon', 'post-harvest'). The StructuredQuery dates must be correctly normalized from the expression.",
    'sensor_phrasing': "Use alternative sensor phrasings: 'radar', 'cloud-penetrating', 'SAR', 'night imagery', 'VV VH polarization'. The StructuredQuery sensor field must match the implied sensor.",
    'change_event_phrasing': "Use change/event phrasings BEYOND the current keyword list (e.g., 'shifted', 'expansion', 'breached', 'inundation', 'erosion', 'damaged', 'urban growth'). The StructuredQuery change_flag must be true and event/object must reflect the change.",
    'disaster_officer': "Use disaster-officer phrasing: waterlogging, embankment breach, access-safety ('is it safe to access', 'can we reach'), urgency ('right now', 'as of today', 'as of now'). Include specific districts/gauges. StructuredQuery must have correct event/object/change_flag.",
    'adversarial_fallback': "Produce adversarial/ambiguous queries: wrong geo pairings, unresolvable dates, off-domain requests, double intents. The StructuredQuery MUST be the EXACT fallback output (location may be wrong/null, dates null, etc.). This teaches graceful failure.",
    'multilingual_hinglish': "Produce Hinglish/Hindi-script phrasings of the above intents (e.g., 'bhaatoi paani keman ase Guwahati', 'paani ka level dikhao'). The StructuredQuery must match the semantic intent.",
}

# ============================================================
# TEACHER LLM SETUP (NVIDIA API)
# ============================================================
import os
import requests

NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')
if not NVIDIA_API_KEY:
    raise RuntimeError("NVIDIA_API_KEY not set in environment")

TEACHER_MODEL = "nvidia/nemotron-3-ultra"
TEACHER_VERSION = "2024-10"
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Generation parameters
GEN_PARAMS = {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 512,
}

# RNG seeds for reproducibility
BASE_SEED = 20260925

# ============================================================
# GENERATION PROMPT TEMPLATE (from build plan §3.2)
# ============================================================
GENERATION_SYSTEM_PROMPT = """You generate training data for a satellite-image query parser. Given a seed query and its gold StructuredQuery JSON, produce N paraphrases. Rules:
1. Each paraphrase re-expresses the SAME intent and must parse to the SAME gold JSON, except where an axis explicitly says to alter one field (see axis instructions).
2. Use the colloquial/transliterated spelling, festival-relative dates, or phrasing style specified by the axis. Do not produce dictionary-form place names unless the axis says so.
3. Write like a real user: typos are allowed only in the place name, never in a way that changes intent. No meta-commentary, no numbering, no explanations.
4. Output strict JSON lines: {"query_text": "...", "structured_query": {...}}"""

def build_user_prompt(seed, axis_name, n):
    """Build the user prompt for a seed-axis pair."""
    axis_inst = AXIS_INSTRUCTIONS.get(axis_name, "")
    return f"""seed_id: {seed['seed_id']}
axis: {axis_name} — {axis_inst}
seed_query: {seed['query_text']}
gold_structured_query: {json.dumps(seed['structured_query'], ensure_ascii=False)}
n: {n}"""

def call_teacher_llm(system_prompt, user_prompt, seed_val):
    """Call NVIDIA API for generation."""
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": TEACHER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": GEN_PARAMS["temperature"],
        "top_p": GEN_PARAMS["top_p"],
        "max_tokens": GEN_PARAMS["max_tokens"],
        "seed": seed_val,
    }
    
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()

def parse_teacher_output(response_json):
    """Parse teacher LLM response into list of records."""
    content = response_json['choices'][0]['message']['content'].strip()
    
    # Handle potential markdown fences
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    
    records = []
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Try to find JSON objects
            pass
    
    return records

# ============================================================
# MAIN GENERATION LOOP
# ============================================================
print("=" * 60)
print("PHASE 1: SYNTHETIC GENERATION")
print("=" * 60)
print(f"Teacher model: {TEACHER_MODEL} (v{TEACHER_VERSION})")
print(f"Decoding params: {GEN_PARAMS}")
print(f"Base RNG seed: {BASE_SEED}")
print()

# Create output directories
gen_dir = r'D:\VxKex\c\Sat-query-AI\satquery\data\llm_finetune\generated'
os.makedirs(gen_dir, exist_ok=True)

raw_output_path = os.path.join(gen_dir, 'raw_teacher_output.jsonl')

# Load seeds
benchmark_seeds = [s for s in seeds if int(s['seed_id'][1:]) <= 35]
human_seeds = [s for s in seeds if int(s['seed_id'][1:]) >= 36]

# For each seed, determine which strata it contributes to
# Each seed family appears in >= 2 strata
# We'll distribute the quota across seeds

# First, group seeds by their primary axis
seed_by_axis = {}
for s in seeds:
    for axis in s.get('variant_axes', ['canonical']):
        if axis not in seed_by_axis:
            seed_by_axis[axis] = []
        seed_by_axis[axis].append(s)

# For axes not in seed variant_axes, we'll use benchmark seeds as canonical
if 'canonical' not in seed_by_axis:
    seed_by_axis['canonical'] = benchmark_seeds

# Calculate how many paraphrases per seed per axis
# Simple approach: distribute quota evenly across seeds in that axis
total_generated = 0
all_records = []

# Track generation metadata
gen_metadata = {
    "teacher_model": TEACHER_MODEL,
    "teacher_version": TEACHER_VERSION,
    "generation_params": GEN_PARAMS,
    "base_seed": BASE_SEED,
    "prompt_template": "v1 (per build plan §3.2)",
    "generated_at": datetime.utcnow().isoformat() + 'Z',
    "batches": []
}

# For simplicity, let's generate a sample first to verify the pipeline
print("Starting with a test batch (S036, colloquial_placename, n=5)...")

test_seed = [s for s in human_seeds if s['seed_id'] == 'S036'][0]
axis = 'colloquial_placename'
n_test = 5
rng_seed = BASE_SEED + 1

user_prompt = build_user_prompt(test_seed, axis, n_test)
print(f"User prompt:\n{user_prompt[:500]}...")
print()

try:
    response = call_teacher_llm(GENERATION_SYSTEM_PROMPT, user_prompt, rng_seed)
    records = parse_teacher_output(response)
    print(f"Generated {len(records)} records:")
    for r in records:
        print(f"  {r}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()