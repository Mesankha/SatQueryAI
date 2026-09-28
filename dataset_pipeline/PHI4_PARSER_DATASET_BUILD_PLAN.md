# SatQuery AI — Phi-4-mini Parser Fine-Tuning Dataset Build Reference
## v1 · 2026-09-24 · Implements plan §3, §9, §11.4 and workstream WS-2

**Purpose:** Turn plan Section 3's "build a synthetic training set of ~1,500–3,000 (query → StructuredQuery) pairs" into an exact, phase-gated pipeline that produces a **ready-to-train** dataset. No fine-tuning code is included here — training uses Microsoft's official `sample_finetune.py` for `microsoft/Phi-4-mini-instruct` (verified: it uses `trl.SFTTrainer` with a `messages` column and the model's chat template, LoRA `r=16, alpha=32, dropout=0.05, target_modules="all-linear"`). This document covers everything **before** `accelerate launch sample_finetune.py`.

**Status labels used throughout:**
- **[FACT]** — verified against repo documents or Microsoft's published script.
- **[ASSUMPTION]** — believed true, must be confirmed in Phase 0 before proceeding.
- **[PROPOSAL]** — a concrete choice; change it only by written amendment, not ad-hoc.

---

## 0. Objective and non-negotiables

**Objective:** A dataset of (natural-language query → `StructuredQuery` JSON) pairs that teaches Phi-4-mini *semantic field extraction* under the diversity axes the current 35-query benchmark under-represents, including queries labeled with the **correct fallback behavior** so the model fails gracefully.

**Non-negotiable rules (hard blockers):**
1. **[FACT]** The `StructuredQuery` Pydantic schema in `m1_parser` is **frozen**. This dataset is shaped to fit it. If the schema needs a field, that is a separate written amendment to the fix plan — never edit schema and dataset in the same change.
2. **[FACT]** No number, count, or quality claim in this pipeline may be fabricated. Every gate below is verified by a script that writes a machine-readable report file. A gate without its report file is **not passed**.
3. **[FACT]** Temporal/geographic leakage discipline (fix plan global rule 5) applies here too: the **real 35-query benchmark is evaluation-only** and must be decontaminated from training data (exact rule in §5.3).
4. **[FACT]** Training file format must be the `messages` chat format that Microsoft's `sample_finetune.py` consumes (verified in the HF model repo), with the system prompt **byte-identical** to the production parser's system prompt.
5. **[PROPOSAL — recommendation]** Target size: **2,400 total pairs** → 2,000 train / 200 validation / 200 test. Rationale: inside plan §3's 1,500–3,000 band; large enough for per-stratum quotas to be meaningful; small enough for a 100%-schema-validation + 10%-human-review pass to complete in one working day.

---

## 1. Deliverables — exact file tree

All under `data/llm_finetune/` (human-owned per the interface contract — the agent never edits this directory; the agent provides only the frozen schema, the fallback function, and validation scripts).

```
data/llm_finetune/
├── pairs.jsonl                  # canonical dataset, one record per line, plan §9 schema:
│                                #   query_text, structured_query, source, variant_type,
│                                #   reviewed_by   (+ seed_id, split — see §2.2, §6.1)
├── seeds/
│   ├── seed_queries.jsonl       # Phase 0 output: the curated seed pool
│   └── seed_report.md           # Phase 0 gate report
├── generated/
│   ├── raw_teacher_output.jsonl # Phase 1 raw teacher output (pre-validation)
│   ├── validated.jsonl          # Phase 1 output: schema-valid, deduped, labeled
│   ├── rejected.jsonl           # every rejected record + machine reason (auditable)
│   └── gen_report.md            # Phase 1 gate report
├── review/
│   ├── review_sample.jsonl      # Phase 2 stratified sample (exactly 10%)
│   ├── review_decisions.jsonl   # per-record human verdicts
│   └── review_report.md         # Phase 2 gate report
├── sft/
│   ├── train.jsonl              # FINAL training file: {"messages":[...]} format
│   ├── val.jsonl                # FINAL validation file
│   ├── test.jsonl               # FINAL test file (never seen in training)
│   └── PACKAGE_REPORT.md        # Phase 4 gate report + final acceptance summary
└── scripts/                     # agent-written, human-executed
    ├── validate_pairs.py        # Phase 1 machine validation
    ├── dedup_decontaminate.py   # Phase 1 dedup + Phase 3 decontamination
    ├── stratified_sample.py     # Phase 2 sampling
    ├── audit_distribution.py    # Phase 3 distribution audit
    └── package_sft.py           # Phase 4 chat-format conversion + package report
```

**Definition of "readymade":** when Phase 4's gate passes, `sft/train.jsonl` + `sft/val.jsonl` are the only files needed by the training step. `sft/test.jsonl` feeds the evaluation step (baseline vs. fine-tuned field-level exact match, per WS-2) — it does **not** go into `sample_finetune.py`.

---

## 2. PHASE 0 — Schema freeze & seed harvest
**Owner:** Team 1 (agent provides frozen schema + fallback function; human curates seeds)
**Duration:** ~0.5–1 day. **No dataset work may start before Gate 0 passes.**

### 2.1 Inputs to confirm (each is a checkable fact)

| # | Item | How to confirm | Blocking? |
|---|---|---|---|
| 0.1 | `StructuredQuery` schema is frozen and exported from one place in `m1_parser` | `grep -rn "class StructuredQuery" --include="*.py"` returns exactly one definition; git log shows no schema change since the freeze commit | **Yes** |
| 0.2 | Production parser system prompt is a single fixed string | It lives in config/code as one constant; copy it byte-for-byte into `seeds/system_prompt.txt` | **Yes** |
| 0.3 | The regex/keyword fallback function runs standalone | Import it and call it on 3 sample queries; record outputs in `seeds/fallback_probe.json` | **Yes** — adversarial labels depend on it |
| 0.4 | The real 35-query benchmark file is identified and hash-stamped | `sha256sum` the file, record hash + path in `seed_report.md` | **Yes** — decontamination baseline |
| 0.5 | Backend decision (Ollama vs `transformers.pipeline`) is made per plan §3 | Recorded in `DECISIONS.md` | No for data, **yes** before packaging (system prompt source differs) |

If any blocking item fails: **stop.** Fix it via the fix plan, then restart Phase 0. Do not "work around" a missing schema by inventing fields.

### 2.2 Build the seed pool (`seeds/seed_queries.jsonl`)

**Source:** the existing 35 benchmark queries **[FACT]** + new human-written seeds.
**Target:** **40–60 seed queries** (each seed will be paraphrased into a family of training pairs in Phase 1).

Each seed record:

```json
{
  "seed_id": "S017",
  "query_text": "exact seed question text",
  "structured_query": { "...": "StructuredQuery JSON, verified per §2.3" },
  "variant_axes": ["colloquial_placename", "festival_date"],
  "origin": "benchmark_seed" | "human_written"
}
```

**Human task H1 (owner: one Team 1 member, ~2h):**
1. Take the 35 benchmark queries verbatim (do not reword them). Label each by running it through the current parser/fallback and verifying the label by hand against the query's plain meaning. These 35 are marked `origin: benchmark_seed`.
2. Write **20–36 additional human seeds** covering the §3/§11.4 axes:
   - 8 colloquial/transliterated Assam place names (e.g., spellings a local would type, not gazetteer forms)
   - 6 festival/season-relative dates (Bihu, Durga Puja, monsoon, harvest season)
   - 4 sensor phrasing variants ("radar", "cloud-penetrating", "SAR", "night imagery")
   - 4 disaster-officer phrasings (waterlogging, embankment breach, "is it safe to access X", urgency: "right now / as of today")
   - 4 change/event phrasings beyond the current keyword list
   - 4 deliberate adversarial/ambiguous queries (wrong state+place pairing, date before sensor era, double location, off-domain request) — label per §4.3
3. **Label verification rule:** every seed's `structured_query` must validate against the frozen schema programmatically, and every label must be checked by the seed writer; a second person re-checks only the adversarial seeds (100% of them).

### 2.3 GATE 0 — exit criteria (all must be TRUE; report: `seeds/seed_report.md`)

- [ ] Item 0.1–0.5 confirmed, evidence (paths/hashes) recorded in `seed_report.md`
- [ ] 40 ≤ seed count ≤ 60; ≥ 4 adversarial seeds; ≥ 20% seeds are disaster-officer phrasing
- [ ] 100% of seed labels pass `StructuredQuery.model_validate()`
- [ ] 100% of adversarial seeds re-checked by a second person; disagreements resolved in writing in the report
- [ ] `system_prompt.txt` matches the production constant byte-for-byte (script-checked `diff`)

**On failure:** fix seeds and re-run. Gate 0 cannot be waived.

---

## 3. PHASE 1 — Synthetic generation & machine validation
**Owner:** human drives the teacher LLM; agent's `validate_pairs.py` + `dedup_decontaminate.py` do the checking.
**Duration:** ~1 day of active work (generation is API-call-bound, not human-bound).

### 3.1 Teacher LLM setup **[PROPOSAL]**

- **Teacher:** any strong general LLM the team already has API access to (do **not** pick a new vendor for this — use what exists). Record model name + version in `gen_report.md` (it becomes part of the dataset provenance, like `teacher_model` in the VLM schema).
- **Decoding:** temperature 0.8 for paraphrase diversity, top_p 0.95, max tokens 512. Seed the generator with a fixed integer per batch and record it. Same seed + same settings must be able to regenerate the identical raw output — this is the reproducibility requirement.
- **Never use Phi-4-mini itself as the teacher.** Self-generation would bake the current extraction errors into the labels instead of fixing them.

### 3.2 Generation prompt template (one batch call per seed; record the exact template in `gen_report.md`)

```
SYSTEM: You generate training data for a satellite-image query parser. Given a seed
query and its gold StructuredQuery JSON, produce N paraphrases. Rules:
1. Each paraphrase re-expresses the SAME intent and must parse to the SAME gold JSON,
   except where an axis explicitly says to alter one field (see axis instructions).
2. Use the colloquial/transliterated spelling, festival-relative dates, or phrasing
   style specified by the axis. Do not produce dictionary-form place names unless the
   axis says so.
3. Write like a real user: typos are allowed only in the place name, never in a way
   that changes intent. No meta-commentary, no numbering, no explanations.
4. Output strict JSON lines: {"query_text": "...", "structured_query": {...}}
USER: seed_id: {seed_id}
      axis: {axis_name} — {axis_instructions}
      seed_query: {seed.query_text}
      gold_structured_query: {seed.structured_query}
      n: {n}
```

Per-axis quotas below (§3.4) determine `n` per seed. Every generated record inherits the seed's `seed_id`, `variant_type = axis`, `source = "synthetic"`.

### 3.3 Machine validation — every generated record, no exceptions (`validate_pairs.py`)

For each record in `raw_teacher_output.jsonl`:

1. **JSON parse** of the whole line and of `structured_query`. Fail → `rejected.jsonl`, reason `json_parse`.
2. **Schema validation:** `StructuredQuery.model_validate(record["structured_query"])`. Fail → `rejected`, reason `schema`. (Target: < 3% rejection. Above 10% → teacher prompt is broken; fix prompt and regenerate the batch, do not hand-patch.)
3. **Field-consistency check vs seed gold:** for normal (non-adversarial) records, `structured_query` must equal the seed gold **exactly** — this is a *paraphrase* task, not a *relabeling* task. Mismatch → `rejected`, reason `label_drift`. (Some drift is expected and fine — but it must be rejected and optionally regenerated with a tightened prompt, never silently accepted, because an unreviewed drifted label is a label error with extra steps.)
4. **Query sanity:** non-empty, ≤ 300 chars, language English/Hinglish (the production user base), no gold JSON pasted into the query text.
5. **Dedup:** normalized query string (lowercase, collapse whitespace, strip punctuation) exact-dup → keep first, reject rest (`reason: dup_exact`). Near-dup within the same `seed_id + variant_type` (word-level Jaccard ≥ 0.9) → reject (`reason: dup_near`).

Accepted records are appended to `generated/validated.jsonl` with all §1 metadata fields.

### 3.4 Quota table — dataset composition (sums to 2,400)

| Stratum (`variant_type`) | Count | Notes |
|---|---|---|
| `canonical` (gazetteer-form place names, plain dates) | 300 | baseline competence — do not under-weight |
| `colloquial_placename` | 450 | transliteration variants; multiple spellings of the same place |
| `festival_seasonal_date` | 300 | Bihu, Durga Puja, monsoon, harvest; include unresolvable ones (label = fallback output) |
| `sensor_phrasing` | 200 | radar/cloud-penetrating/SAR/night |
| `change_event_phrasing` | 300 | beyond current keyword list |
| `disaster_officer` | 450 | WS-2 §11.4 set: waterlogging, embankment breach, access-safety, urgency |
| `adversarial_fallback` | 250 | labeled with **correct fallback output**, per §4.3 |
| `multilingual_hinglish` | 150 | Hinglish/Hindi-script phrasings of the above intents |
| **Total** | **2,400** | |

**Rules:** (a) every seed family appears in ≥ 2 strata unless the seed's nature forbids it; (b) no stratum may deviate from its quota by more than ±10% at Gate 1 — rebalance by regenerating the under-filled stratum, not by deleting over-filled ones (deleting biases toward easy strata).

### 3.5 GATE 1 — exit criteria (report: `generated/gen_report.md`)

- [ ] Total accepted ≥ 2,400; every stratum within ±10% of quota
- [ ] `schema` rejection rate < 10% overall; `label_drift` rate < 15% overall
- [ ] 100% of accepted records validate against the frozen schema (re-run validator on `validated.jsonl`; report line count of pass/fail)
- [ ] `rejected.jsonl` contains every rejected record with a machine reason — zero unexplained losses
- [ ] Teacher model name/version, prompt template version, decoding params, and RNG seeds recorded

**On failure:** regenerate the failing strata. **Do not enter Phase 2 with a below-quota stratum** — the Phase 2 review sample is stratified, and thin strata break the statistical meaning of Gate 2.

---

## 4. PHASE 2 — Human review (the only phase that uses human judgment at scale)
**Owner:** two named reviewers from Team 1 (reviewer A: seed-adjacent member; reviewer B: a second member who did not write those seeds).
**Duration:** ~4–5 reviewer-hours total. **This phase exists because schema validity is machine-guaranteed but label correctness is not.**

### 4.1 Sampling (`stratified_sample.py`)

- Exact **10%** of `validated.jsonl`, **stratified by `variant_type`** (so each stratum contributes ≥ 10% of its records, rounded up; adversarial stratum contributes **100%** — all 250).
- Expected sample: ~240 records + mandatory 100% adversarial coverage already inside that 10%? No — adversarial gets 100% (250) **plus** its 10% share would double-count; rule: sample = 10% of each stratum, then replace the adversarial 10% share with **100% of the adversarial stratum**. Total review burden ≈ 250 + ~215 ≈ **465 records**. Budget 4–5 hours (≈30–35 sec/record with a review UI or spreadsheet).
- `review_sample.jsonl` records the sample; everything not sampled remains `reviewed_by: null`.

### 4.2 Review rubric (printed at the top of the review file — reviewers must not improvise criteria)

For each record, verdict ∈ {`accept`, `fix`, `reject`} with a one-line reason for fix/reject:

1. **Intent match:** does the query, read as a real user would mean it, correspond to the labeled `structured_query`? If the phrasing is ambiguous between two readings, is the label the *more natural* reading? (If neither reading matches → `fix` or `reject`.)
2. **Field-level checks:** location string is the canonical gazetteer form the schema expects (colloquial spellings live in the *query*, not the label); date range is correctly normalized from the query's relative/festival expression and does not leak outside the sensor era mentioned or implied; sensor/modality field matches query phrasing; `change_flag` set iff the query asks about change.
3. **Fallback-labeled records (adversarial):** the label must be the *output the current regex/keyword fallback actually produces for this query* — verify by running the fallback on the query text, not by reasoning from memory. If the fallback's real output is wrong-but-deterministic, the label is still the fallback output (that is what "fail gracefully" means: the fine-tuned model learns to emit the same safe structure), and the case is logged in `review_report.md` as a fallback-improvement candidate for the fix plan.
4. **Language quality:** the query reads like something a real user would type — awkward but not nonsensical. Giberish → `reject`.

### 4.3 Adversarial labeling convention (restated — this is where vague work sneaks in)

Every adversarial query must fall into one of **four named cases**, and the case name is stored in the record:

| Case | Example shape | Label rule |
|---|---|---|
| `wrong_geo_pairing` | place name + wrong state/river | fallback output (usually location=null or best-effort), **not** a guessed location |
| `unresolvable_date` | "before Bihu" with no year context that resolves | fallback output; never an invented date range |
| `off_domain` | "book me a flight to Guwahati" | fallback output / empty StructuredQuery shape the fallback emits |
| `double_intent` | two locations or two events in one query | fallback output for the primary intent only, matching what the fallback actually returns |

A new case may be added only by written amendment, with its label rule appended to this table. "I'll just label it sensibly" is not a rule and is not permitted.

### 4.4 Gate computation

- Per-stratum label-correctness rate = accepted / reviewed in that stratum (fixed records count as correct only if the fix is verified correct by reviewer B).
- **Overall pass:** ≥ 98% label correctness, **and** ≥ 95% **in every stratum**, **and** 100% of `reject` decisions audited by reviewer B.
- **On failure of a stratum:** the **entire stratum** is regenerated in Phase 1 (not just the bad records), then re-reviewed at 10%. One regeneration cycle is expected; two consecutive failures of the same stratum means the axis definition is ambiguous — stop, rewrite the axis instructions, amend this document, restart Phase 1 for that stratum. Never proceed to Phase 3 with an un-passed stratum.

### 4.5 GATE 2 — exit criteria (report: `review/review_report.md`)

- [ ] Sample = 10% per stratum with adversarial at 100%; counts recorded per stratum
- [ ] Overall label correctness ≥ 98%; every stratum ≥ 95%
- [ ] All `fix` records patched in `validated.jsonl` and re-validated against schema (script re-run, new pass count in report)
- [ ] All fallback-improvement candidates listed and handed to the fix plan as a separate task (do not fix fallback code inside this pipeline)
- [ ] Reviewer names (or initials) recorded per record in `reviewed_by`; `review_decisions.jsonl` complete

---

## 5. PHASE 3 — Decontamination & distribution audit
**Owner:** agent scripts, human signs off.
**Duration:** < 1 hour of wall time; it is machine work plus one human read of a report.

### 5.1 Benchmark decontamination (the leakage blocker)

- The 35 benchmark queries (hash-stamped in Phase 0) are **evaluation-only** [FACT per plan §3 step 4].
- Rule: **no training/validation record may be a near-copy of any benchmark query.** Compute word-level Jaccard similarity between every record's `query_text` and every benchmark query. Threshold **[PROPOSAL]**: Jaccard ≥ 0.6 → move record to `sft/test.jsonl` only if its seed differs from the benchmark seed's family... **no — simpler and stricter:** Jaccard ≥ 0.6 → **drop from the dataset entirely** and log. Rationale: benchmark-adjacent paraphrases in training inflate the benchmark score without improving the thing judges care about (new phrasings).
- Also drop any record whose `query_text` contains a benchmark query as a substring after normalization.

### 5.2 Temporal/geographic split discipline (fix plan global rule 5, applied to data)

- Test set must contain **≥ 30% pairs from seed families that appear nowhere in train** (i.e., whole seed families held out). This is the data-level analog of "split by AOI and by time, never by random row": it tests extraction *generalization to unseen intent phrasings*, not memorized seed paraphrases.
- Enforced at split time in Phase 4, but audited here: `audit_distribution.py` reports how many seed families exist and confirms ≥ 20 families are available for holdout (if not, Phase 1 must add seeds — Phase 3 blocks on this).

### 5.3 Distribution audit (numbers, not vibes)

`audit_distribution.py` writes: record count per stratum; distinct place names; distinct date-expression types (absolute / relative / festival); sensor values histogram; `change_flag` true/false ratio; avg query length per stratum; train/test seed-family overlap (must be disjoint except where documented).

**Sanity targets [PROPOSAL]:** ≥ 15 distinct locations; ≥ 6 distinct festival/season anchors; `change_flag` ratio between 0.3–0.6; no single location > 15% of the dataset (prevents location memorization from masquerading as parsing skill).

### 5.4 GATE 3 — exit criteria (report: `audit_distribution.md`)

- [ ] Zero benchmark-near-dup records in train/val (script prints count; must be 0)
- [ ] ≥ 20 seed families available for test holdout
- [ ] All distribution sanity targets met
- [ ] Human has read the audit report and signed the line in it (name + date)

---

## 6. PHASE 4 — Split & package the readymade SFT dataset
**Owner:** `package_sft.py` (agent-written), human executes and verifies.

### 6.1 Split

- Hold out **2 seed families** (chosen at random with recorded RNG seed) entirely → all their records go to `test.jsonl`.
- Remaining records: **2,000 → train**, **200 → val**, remainder → `test.jsonl` (target ~200 test), all stratified by `variant_type`, split at the **family level** (records of one family stay in one split).
- `split` field (`train`/`val`/`test`) is stamped into every record of `pairs.jsonl` as the final canonical artifact.

### 6.2 Package to training format (verified against Microsoft's script)

Microsoft's `sample_finetune.py` **[FACT]** does: `load_dataset(...)`, reads the **`messages`** field, applies `tokenizer.apply_chat_template(..., add_generation_prompt=False)`, trains with `SFTTrainer(..., dataset_text_field="text", packing=True)`. Therefore each output line must be:

```json
{"messages": [
  {"role": "system", "content": "<exact production parser system prompt from seeds/system_prompt.txt>"},
  {"role": "user", "content": "<query_text>"},
  {"role": "assistant", "content": "<minified StructuredQuery JSON, keys sorted, UTF-8, no markdown fence>"}
]}
```

Rules:
- The system prompt is **byte-identical** across all records and equal to production (Gate 0 checked this; Gate 4 re-checks by hash).
- Assistant content is **pure JSON** — no ```json fences, no trailing commentary. (The production parser extracts JSON via regex; if training data contains fences, the model learns to emit fences.)
- `load_dataset("json", data_files={"train": "sft/train.jsonl", ...})` must load with **zero errors** — `package_sft.py` performs exactly this load as its final self-check.
- Extra metadata columns are fine (SFTTrainer ignores them with `remove_unused_columns=True`, which the sample script sets); the canonical provenance fields (`seed_id`, `variant_type`, `source`, `reviewed_by`, `split`) are **kept** in the SFT files for auditability.

### 6.3 Training handoff (no training code here — instructions only)

1. Copy `sample_finetune.py` from the `microsoft/Phi-4-mini-instruct` HF repo into a scratch training dir (do not vendor it into the app repo).
2. Change exactly two things: the dataset source (point at `sft/train.jsonl` / `sft/val.jsonl`) and hyperparameters per plan §3: **2–3 epochs** (start 2), LoRA **r 8–16** (script default 16 is acceptable), learning rate per script default 5e-6 is for full-scale SFT; **[PROPOSAL]** 1e-4–2e-4 typical for LoRA adapters — pick 1e-4, it's inside the band used in Microsoft's own ecosystem examples.
3. Environment pinned per script header: `accelerate==1.3.0, bitsandbytes, peft==0.14.0, transformers==4.48.1, trl, datasets, deepspeed`. Run `accelerate config` then `accelerate launch sample_finetune.py` on the RTX 3050/5070 (4-bit QLoRA if OOM; the script itself notes batch size / LoRA dim / target modules as the levers).
4. Baseline first **[FACT per WS-2]**: before training, run the base Phi-4-mini on `sft/test.jsonl` + the real 35-query benchmark through the evaluation harness and store the field-level exact-match table in `evals/` tagged `baseline_`. No before/after table → no claim, and the fine-tune is not "done."

### 6.4 GATE 4 — final acceptance (report: `sft/PACKAGE_REPORT.md`)

- [ ] `pairs.jsonl` line count = 2,400 (±0); train 2,000 / val 200 / test 200 (±5% after decontamination drops)
- [ ] 100% of SFT records: schema-valid label, correct system-prompt hash, pure-JSON assistant content
- [ ] `load_dataset("json", ...)` self-check exit 0 for all three files
- [ ] Train/val/test seed-family disjointness confirmed by script output
- [ ] All four prior gate reports linked; all gate failures and resolutions summarized
- [ ] Human sign-off line: name, date, "dataset approved for training"

---

## 7. Human work cards (explicit, unambiguous)

| Card | Task | Owner | Time | Done when |
|---|---|---|---|---|
| **H1** | Seed curation per §2.2 (35 benchmark + 20–36 new seeds, adversarial re-check) | 1 Team 1 member + 1 verifier | ~2h + 0.5h | Gate 0 passes |
| **H2** | Drive teacher generation: batch 60 seeds × quota, collect raw output | same member | ~1h active | `raw_teacher_output.jsonl` complete |
| **H3** | Review pass per §4 (≈465 records, rubric printed) | 2 reviewers | ~4–5h total | Gate 2 passes |
| **H4** | Adjudicate `fix`/`reject` edges + audit rejects | reviewer B | ~1h | Gate 2 report complete |
| **H5** | Read audit report + sign Gate 3 | Team 1 lead | 15 min | signature in `audit_distribution.md` |
| **H6** | Run packaging script, verify `PACKAGE_REPORT.md`, sign Gate 4 | Team 1 lead | 30 min | Gate 4 signed |
| **H7** | Kick off training + baseline eval per §6.3–6.4 (outside this doc's scope, listed so the pipeline end-state is explicit) | GPU owner (Team 2 pairing) | hours–overnight | `evals/` baseline + fine-tuned tables |

Total human effort: **~9–11 hours spread over 3–4 days**, matching WS-2's "Week 1–2" cadence slot.

---

## 8. Vague-work prevention rules (enforce these and the pipeline can't silently rot)

1. **Every phase has a machine-generated report and a numeric gate.** "Looks fine" is not a gate. If a script can't check it, it doesn't count as done.
2. **Every rejection is auditable.** `rejected.jsonl` must account for 100% of records that didn't make it. Unexplained loss > 0.5% → Phase 1 redone.
3. **Every human task has an owner, a time box, and an artifact.** Anything outside the H1–H7 cards is scope creep; add it only by amending this document.
4. **Every gate failure has exactly one prescribed response** (regenerate / re-review / stop-and-amend). Nobody chooses an ad-hoc response at gate time.
5. **No schema edits inside this pipeline.** Ever.
6. **The 35 benchmark queries are read-only** for the whole pipeline — input to Phase 0, decontamination target in Phase 3, evaluation set after training. They are never training rows.
7. **Any single record can be traced**: seed → teacher batch → validation → review decision → split → SFT line. If tracing breaks, the dataset is not readymade.

---

## Appendix A — worked example records (shape reference only; place/date values below are illustrative)

```json
{"query_text": "bhaatoi paani keman ase Guwahati'r xaikhote ei xomoy?", "structured_query": {"location": "Guwahati", "date_start": "<normalized from 'ei xomoy' per schema rules>", "sensor": "any", "object": "water", "event": "flood", "change_flag": true}, "source": "synthetic", "variant_type": "multilingual_hinglish", "seed_id": "S031", "split": "train", "reviewed_by": null}
{"query_text": "show me water levels near Kaziranga after this year's Rongali Bihu", "structured_query": {"location": "Kaziranga", "date_start": "<Bihu-normalized>", "sensor": "any", "object": "water", "event": "flood", "change_flag": false}, "source": "synthetic", "variant_type": "festival_seasonal_date", "seed_id": "S018", "split": "train", "reviewed_by": "AB"}
{"query_text": "is it safe to drive to Majuli right now, is the embankment broken?", "structured_query": {"location": "Majuli", "date_start": "<today-relative>", "sensor": "any", "object": "embankment", "event": "breach", "change_flag": true}, "source": "synthetic", "variant_type": "disaster_officer", "seed_id": "S040", "split": "train", "reviewed_by": null}
```

(The `"<...>"` values must be filled with the exact normalized forms the frozen schema/fallback produces — run the code, never hand-normalize.)

## Appendix B — amendment log

| Date | Change | Approved by |
|---|---|---|
| 2026-09-24 | v1 initial | (pending team review) |
| 2026-09-25 | Benchmark resolution: replaced all "24-query benchmark" references with "35-query benchmark" per repository evidence. The canonical benchmark is fixtures/fixtures_queries.json (35 entries), snapshotted at data/llm_finetune/benchmark/benchmark_35.json. AUDIT_REPORT.md updated to reflect 35 queries. | (resolved per benchmark decision) |
