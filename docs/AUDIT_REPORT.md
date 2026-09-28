# SatQuery AI — Adversarial Engineering Audit Report

**Date:** 2026-09-08  
**Auditor:** Kilo (automated adversarial audit)  
**Repository:** `satquery/`  
**Spec Documents:** `satquery_ai_production_scaleup_plan.md` (only spec found; `satellite_retrieval_hackathon_playbook.md` not found in repo)

---

## Executive Summary

**Overall Verdict: FAIL** — The system is **not truthfully described by its own documentation**. Critical Phase 0 blockers remain unfixed, mocks/stubs are wired into live code paths, the LLM branch is structurally unreachable due to timeout mismatch, SAR data is sent to the VLM (violating a hard architectural constraint), evaluation numbers are near-zero placeholders, and "on the horizon" features are documented but not built.

---

## Section A — Phase 0 Blockers (MUST-PASS)

### A1. Validator Timeout vs LLM Timeout
| Parameter | Value | Location |
|-----------|-------|----------|
| Validator timeout (passed to `parse_llm`) | **5.0 seconds** | `m1_parser/parser_validator.py:26` |
| LLM internal timeout (config) | **5 seconds** | `config.yaml:32` |
| M1 LLM pipeline load | **No GPU / torch not installed** | Runtime verified |

**Finding: FAIL** — The LLM branch is **structurally unreachable**. The validator calls `parse_llm(query_text, timeout_seconds=5.0)`, but:
- PyTorch is not installed in the environment (`ModuleNotFoundError: No module named 'torch'`)
- CUDA is unavailable → `_get_pipeline()` returns `None` immediately → `parse_llm` returns `None` in ~6ms
- Every query falls back to deterministic parser (`used_fallback=True` always)

**Evidence:** Runtime test confirms 6ms fallback every time. Fine-tuning Phi-4-mini is pointless until GPU + torch are available and timeout is increased to ≥30s.

---

### A2. M2 Retrieval Wired into M5 Dispatch?

**Finding: PARTIAL — Mock still in fallback path**

The dispatch table (`m5_controller/dispatch_table.py`) calls `m2_retrieve()` which:
1. **Tries** `_real_m2()` from `m5_controller/m2_adapter.py` (real M2 pipeline: Stage 1 filter → Stage 2 semantic rank → Stage 3 fusion)
2. **On any exception**, falls back to `load_fixture_scenes("fixtures/fixtures_scenes.json")` (line 56-76)

**Live mock still wired:** The fixture fallback is a **naive keyword filter** (lines 67-76), not the real M2 pipeline. Since M0 catalog is empty (0 scenes), every real query hits the fixture fallback.

**Evidence:** 
- `m0_catalog/catalog.py` count = 0 (verified)
- `real_m2_retrieve()` logs "M0 catalog is empty - falling back to fixtures"
- M2 pipeline (Stage 1→2→3) **never runs on real data** in current state

---

### A3. Catalog/Fixture File Paths — Real Data?

| File | Exists | Non-empty | Real Data |
|------|--------|-----------|-----------|
| `fixtures/fixtures_scenes.json` | ✅ Yes | ✅ 27 scenes | ⚠️ Synthetic fixtures (hardcoded UUIDs, fake file paths) |
| `fixtures/fixtures_queries.json` | ✅ Yes | ✅ 35 queries | ✅ Realistic benchmark queries |
| `fixtures/fixtures_results.json` | ✅ Yes | ✅ Expected results | ✅ Ground truth for eval |
| `data/catalog.db` | ✅ Yes | ❌ 0 rows | ❌ Empty |

**Finding: FAIL** — M0 catalog is empty. All live queries route through fixture fallback. No real Sentinel-1/2 data ingested.

---

### A4. End-to-End `/query` Candidate Count at Metadata-Filter Stage

**Test:** `Show me Sentinel-2 images of agriculture near Guwahati from June 2024`

| Stage | Candidate Count | Source |
|-------|-----------------|--------|
| M0 catalog (total) | 0 | SQLite `catalog.db` |
| Fixture fallback (total) | 27 | `fixtures_scenes.json` |
| After metadata filter (AOI ∩ date ∩ sensor) | **10** | Fixture naive filter |
| Final results returned | 5 | Dispatch table `pipeline_search` |

**Note:** The 10 candidates are from **fixture keyword matching**, not real metadata filtering on AOI/date/sensor. The real M2 `SatelliteMetadataFilter` never executes on real data.

---

### Section A Verdicts

| Blocker | Verdict | Evidence |
|---------|---------|----------|
| A1: Timeout mismatch makes LLM reachable | **FAIL** | Validator 5s, LLM 5s, torch missing → 100% fallback |
| A2: M2 wired into M5 (no mock in live path) | **FAIL** | Fixture fallback always triggers (M0 empty) |
| A3: Catalog/fixture paths resolve to real data | **FAIL** | Catalog empty; fixtures are synthetic placeholders |

---

## Section B — Query Parser (M1)

### B1. Outlines Grammar-Constrained Decoding
**Finding: NOT IMPLEMENTED** — No `outlines` library usage found anywhere in codebase.
- `parser_llm.py` uses raw `transformers.pipeline` with `do_sample=False, temperature=0.0`
- JSON extracted via regex (` ```json` block parsing), not constrained decoding
- `StructuredQuery` validation happens **post-hoc** via `model_validate()` (line 136), not during generation

### B2. Regex/Keyword Fallback Returns Valid StructuredQuery
**Finding: PASS** — `parser_fallback.py:parse_fallback()` always returns a schema-valid `StructuredQuery` (line 224-247). Never raises, never returns raw exception.

### B3. Fine-Tuned Phi-4-mini LoRA Checkpoint
**Finding: NOT LOADED / NOT EXISTENT**
- No LoRA adapter path exists at `./models/geochat_lora` (config.yaml:47)
- `parser_llm.py` loads base `microsoft/Phi-4-mini-instruct` only
- No eval script comparing fine-tuned vs base+outlines found
- Production plan Section 3 explicitly states: "Zero LoRA checkpoints completed for either VLM or LLM fine-tuning"

### B4. Actual Runtime Config vs Documented

| Parameter | Documented (config.yaml) | Actual Runtime |
|-----------|--------------------------|----------------|
| `num_ctx` | Not specified | N/A (transformers pipeline, not Ollama) |
| Quantization | `load_in_4bit: true` | Not loaded (torch missing) |
| Ollama flags | N/A | Not using Ollama |
| Model | `microsoft/Phi-4-mini-instruct` | Base model only, never loaded |

---

## Section C — Retrieval & Ranking (M2, M4)

### C1. Metadata Filter Runs Before Embedding/FAISS
**Finding: PASS (in code structure)** — `real_m2_retrieve()` in `m2_adapter.py` lines 387-400:
```python
# Stage 1: Metadata filtering
filtered = SatelliteMetadataFilter.filter_candidates(m2_scenes, m2_query)
# Stage 2: Semantic ranking
ranked, encoder_used = ranker.rank_candidates(filtered, m2_query, top_k=top_k * 2)
# Stage 3: Multi-factor score fusion
final = score_engine.score_and_rank_candidates(ranked, m2_query, top_k=top_k)
```
**But:** Never executes on real data (M0 empty → fixture fallback bypasses all 3 stages).

### C2. FAISS Index Populated with Real Embeddings
**Finding: FAIL** — `FAISSIndexFlatIPAdapter` size = 0, `_use_faiss = False` (verified). Deterministic hash-based fallback encoder used. No real embeddings stored.

### C3. Fusion/Ranking Formula Matches Playbook Section 6
**Finding: CODE MATCHES SPEC** — `scoring.py:SatelliteScoreEngine` implements:
```
S_composite = w_geo*S_geo + w_temp*S_temporal + w_sem*S_sem + w_mod*S_modality
```
Default weights (config.py): `w_geo=0.30, w_temp=0.25, w_sem=0.30, w_mod=0.15` ✅

**Change-flagged weight shift** (scoring.py:196-201):
```python
if extra.get("change_flag"):
    w_t = 0.40; w_s = 0.40; w_g = 0.10; w_m = 0.10
```
✅ Matches production plan Section 6 (w5=0.35 for change).

**Component score definitions:**
- `geo_score`: bbox overlap ratio (0-1) ✅
- `temporal_score`: 1.0 if within range, decay outside ✅
- `semantic_score`: from Stage 2 (0.5 default if no encoder) ✅
- `modality_score`: 1.0 match, 0.3 mismatch, ×0.5 if cloud > max ✅

**No hardcoded shortcuts or unused weights found.**

### C4. Change-Flagged Queries Use Alternate Weight Set
**Finding: PASS** — Branch at `scoring.py:197-201` confirmed. `change_flag` from `m1_query.change_flag` propagates via `extra_params`.

---

## Section D — SAR Handling (HARD CONSTRAINT)

### D1. SAR Data Paths to VLM (EarthDial)
**Finding: FAIL — VIOLATION FOUND**

**File: `m5_controller/dispatch_table.py` — `pipeline_fusion()` lines 395-411:**
```python
sar = next((c for c in candidates if c.modality == "sar"), None)
...
sar_caption = await asyncio.wait_for(
    _call_m3_caption(sar.file_path or ""),  # ← SAR SENT TO VLM
    timeout=35.0
)
```

**File: `m3_vlm/http_client.py` — `_call_m3_caption()`** calls M3 `/caption` endpoint with SAR image path.

**File: `m3_vlm/sar_prompts.py`** — Contains `SAR_CAPTION_TEMPLATE` and `select_prompt()` that **explicitly routes SAR to captioning** (lines 172-179).

**Production Plan Section 4 (hard rule):**
> "Do not attempt to LoRA-fine-tune SAR narrative capability into EarthDial... your playbook's golden path already routes SAR through deterministic metadata + change-score explanations with no VLM narrative claim."

**This is a direct architectural violation.** SAR images are sent to the VLM for caption generation in the fusion pipeline.

### D2. SAR Explanations Generated by Deterministic Template Only
**Finding: PARTIAL** — M4 change detection (`m4_changedetect/explanation/templates.py`) generates deterministic SAR explanations (log-ratio + metadata) ✅

**BUT** M5 fusion pipeline (`dispatch_table.py`) **also** calls VLM for SAR captions and synthesizes them → violates "exclusively deterministic template" requirement.

---

## Section E — Change Detection

### E1. NDVI/NDWI Delta on Real Sentinel-2 Bands
**Finding: PASS (code structure)** — `optical_analyzer.py` extracts `nir`, `red`, `green` bands and calls `ndvi.py`/`ndwi.py` mathematical functions. Uses real NumPy arrays.

**Verified:** Test with synthetic arrays produces real delta values (e.g., NDVI delta -0.54 → change_score 0.27).

### E2. SAR Log-Ratio on Real Sentinel-1 Data
**Finding: PASS (code structure)** — `sar_analyzer.py` extracts SAR intensity (VV/VH) and calls `log_ratio.py` functions.

### E3. Single Observation → change_score=null + Honest Explanation
**Finding: PASS** — `change_detector.py:66-70`:
```python
if image_t1 is None or image_t2 is None:
    return ChangeResult(
        change_score=None,
        change_description="Change detection unavailable: Missing bi-temporal image pair."
    )
```
**Runtime verified:** Single observation returns `change_score=None` with honest message.

---

## Section F — Explanation Generator / Scientific Honesty

### F1. Confirmatory Language Violations

| File:Line | String | Violation |
|-----------|--------|-----------|
| `m5_controller/dispatch_table.py:204` | `"These findings are consistent with the query criteria."` | **Borderline** — "consistent with" is hedged, but appended to every result regardless of actual evidence |
| `m4_changedetect/explanation/llm_polish.py` | Guardrails explicitly **forbid** "detects", "confirms", "identifies", "proves" | ✅ Good — but LLM polish is disabled by default (`enable_llm_polish: false` in config.yaml:55) |

**No hard violations of confirmatory language found in deterministic templates.** The M4 templates use "indicates", "observed", "shows" — appropriately hedged.

### F2. Factual Metadata vs Model-Derived Separation
**Finding: PARTIAL** — `dispatch_table.py:_synthesize_result()` (lines 189-205):
```python
explanation = (
    f"This result is ranked #{rank} with a composite score of {score:.2f}. "
    f"The image was acquired by {scene.sensor} on {scene.acquisition_time.isoformat()} over {scene.geometry_wkt[:30]}... "
    f"{model_summary} These findings are consistent with the query criteria."
)
```
- Metadata (sensor, date, location) and model outputs (caption, VQA, change_desc) are **concatenated in one paragraph**
- No structural separation in `ResultItem` schema — `explanation_text` is a single string
- `confidence_note` only mentions parser fallback, not model uncertainty

---

## Section G — Dispatch Matrix / Fallback Hardening

| Condition | Status | Code Location | Automated Test? |
|-----------|--------|---------------|-----------------|
| Optical query, Tier 1/2 AOI, parser succeeds → full pipeline | **Partial** | `pipeline_search` | ✅ `test_search_pipeline_returns_results` |
| SAR-relevant query → SAR golden path only (no VLM narrative) | **FAIL** | `pipeline_fusion` calls `_call_m3_caption` on SAR | ❌ No test for this |
| AOI outside Tier 1/2 → Tier 3 live STAC fallback | **Not Implemented** | No Tier 3 logic in code | ❌ |
| LLM parser fails/times out → regex fallback, valid output | **PASS** | `parser_validator.py:19-38` | ✅ Implicit (all tests use fallback) |
| VLM unavailable → deterministic explanation only | **Partial** | `_call_m3_*` return `UniformError`; `pipeline_*` handle | ✅ `test_vqa_pipeline_with_missing_image_returns_error_result` |
| Change-flagged, single observation → change_score=null, honest text | **PASS** | `change_detector.py:66-70` | ✅ `test_missing_second_image` |
| Fine-tuned Phi-4-mini fails → base → outlines → regex | **Not Implemented** | No fine-tuned model exists; no outlines | ❌ |

**Critical Gap:** No test exists for "SAR query → no VLM narrative". The fusion pipeline **explicitly violates** this.

---

## Section H — Disaster-Officer Surfaces (Production Plan Section 11)

| Feature | Status | Evidence |
|---------|--------|----------|
| Confidence bands (High/Med/Low) driven by quality flags | **Not Implemented** | No confidence band logic in code; `confidence_note` only mentions parser fallback |
| Data-freshness statement (acquisition timestamp + age) | **Not Implemented** | No freshness computation in response |
| Briefing/synthesis mode (template over verified fields) | **Not Implemented** | Only `mock_llm_synthesis` placeholder (line 160-162) returns hardcoded string |

**All three are documented but not built.**

---

## Section I — Data & Catalog

### I1. Live Database Row Counts
| Tier | AOIs | Scenes in Catalog |
|------|------|-------------------|
| Tier 1 (Assam 7 AOIs) | 7 | **0** (catalog empty) |
| Tier 2 (expansion) | 0 | **0** |

### I2. Gazetteer
**Finding: HARDCODED + NOMINATIM FALLBACK** — `m2_adapter.py:_geocode_location_to_bbox()` (lines 141-214):
- 30+ hardcoded Indian city bboxes (lines 152-190)
- Nominatim fallback for unknown locations (lines 202-212)
- **No invented/guessed coordinates found** — returns `None` for unknown, letting text search handle it ✅

---

## Section J — Evaluation & Reliability

### J1. Evaluation Harness Results (Last Real Run)

| Metric | Value | Notes |
|--------|-------|-------|
| Recall@5 | **0.167** | Only 1/6 queries had relevant scene in fixtures |
| BERTScore P/R/F1 | **~0.008** | Mock embeddings (hash-based), not real BERTScore |
| BLEU | **0.000** | No n-gram overlap with synthetic ground truth |
| IoU | **0.000** | Grounding returns no bbox |
| Avg Latency | **651 ms** | Mostly fixture lookup |
| Success Rate | **100%** | Pipeline never crashes (fallbacks everywhere) |

**All metrics are effectively placeholders** — mock embeddings, synthetic ground truth, fixture-only data.

### J2. Offline Demo-Replay Mode
**Finding: IMPLEMENTED & WORKING** — `m6_api/main.py:_cache_response()` / `_get_cached_response()` with LRU eviction to disk (`replay_cache/`). Verified working in test.

### J3. "Kill API Key / Kill Network" Resilience Tests
**Finding: NOT IMPLEMENTED** — No such tests exist. M3 HTTP client has retry logic but no test for network/API key failure scenarios.

---

## Section K — Frontend

**Finding: NO FRONTEND IN REPOSITORY** — No `frontend/`, `web/`, `ui/`, or similar directory. No HTML/JS/React/Vue files outside `venv/`. The 5 signature queries cannot be verified against a live UI.

---

## Section L — Nationals-Differentiation Gap Check

| "On the Horizon" Item (Production Plan) | Shipped & Working? | Evidence |
|----------------------------------------|-------------------|----------|
| Disaster-officer briefing mode | ❌ **No** | Only `mock_llm_synthesis` placeholder |
| Confidence bands (High/Med/Low) | ❌ **No** | No implementation |
| Data-freshness statements | ❌ **No** | No implementation |
| Tier 2 catalog expansion (15-25 AOIs) | ❌ **No** | Catalog empty (0 scenes) |
| Phi-4-mini LoRA fine-tune | ❌ **No** | No checkpoint, no training run, torch missing |
| EarthDial fine-tune (RGB) | ❌ **No** | No dataset bootstrap, no training |
| SAR narrative restriction (hard rule) | ❌ **Violated** | Fusion pipeline sends SAR to VLM |

**Verdict:** This reads as a **search demo with a disaster-themed pitch deck**, not a working disaster-response prototype. Core architectural violations (SAR→VLM), empty catalog, unreachable LLM, and zero "horizon" features shipped.

---

## Module-by-Module Status Table

| Module | Status | Evidence | Risk |
|--------|--------|----------|------|
| M0 Catalog | **Broken** | 0 scenes, empty DB, no ingestion run | Critical — blocks all retrieval |
| M1 Parser | **Degraded** | 100% fallback, no LLM, no outlines, no fine-tune | High — parser accuracy unverified |
| M2 Retrieval | **Untested on real data** | Pipeline code exists but never runs (M0 empty) | Critical — core retrieval unvalidated |
| M3 VLM | **Not integrated** | HTTP client exists, service not running, SAR→VLM violation | High — architectural violation |
| M4 Change Detection | **Working (unit tests pass)** | 97 tests pass, deterministic math verified | Low — solid implementation |
| M5 Controller | **Partial** | Dispatch wired, but fusion violates SAR rule; mocks in fallback | High — SAR violation |
| M6 API | **Working** | Endpoints respond, replay mode works, eval harness runs | Medium — eval metrics meaningless |

---

## Scientific-Honesty Violations

| File:Line | String | Issue |
|-----------|--------|-------|
| `m5_controller/dispatch_table.py:204` | `"These findings are consistent with the query criteria."` | Appended to every result regardless of actual match quality; implies validation that doesn't exist |
| `m5_controller/dispatch_table.py:426` | `"Optical observation: {opt_caption}. SAR observation: {sar_caption}."` | Presents VLM-generated SAR caption as factual observation (SAR→VLM violation) |

---

## Fabricated/Placeholder Evaluation Numbers

| Location | Metric | Value | Reality |
|----------|--------|-------|---------|
| `eval_results.json` | Recall@5 | 0.167 | Only 1/6 queries matched fixture IDs |
| `eval_results.json` | BERTScore | ~0.008 | Mock hash embeddings, not BERT |
| `eval_results.json` | BLEU | 0.0 | No n-gram overlap with synthetic refs |
| `eval_harness.py:51-63` | `MockEmbeddingModel` | — | Explicitly labeled "replace with real model" |

---

## Prioritized Fix List

### Must Fix Before Sept 20 (Pre-Nationals Submission)
1. **Fix M0 catalog ingestion** — Populate `catalog.db` with real Sentinel-1/2 scenes for 7 Assam AOIs (use `populate_m0_from_planetary` or `populate_m0_from_sar_lora`)
2. **Remove SAR→VLM path in fusion pipeline** — `pipeline_fusion` must not call `_call_m3_caption` on SAR modality; use deterministic template only
3. **Increase parser timeout to 30s + install torch/CUDA** — Make LLM branch reachable; validate Phi-4-mini loads
4. **Wire real M2 pipeline end-to-end** — Ensure `real_m2_retrieve` succeeds without fixture fallback (remove try/except or make fallback explicit test-only)
5. **Add automated test for "SAR query → no VLM narrative"** — Fail CI if SAR sent to VLM

### Should Fix Before Nationals If Shortlisted
6. **Implement confidence bands** — Map cloud %, coregistration quality, single/dual obs → High/Med/Low with reason string
7. **Add data-freshness statement** — Compute "most recent observation: [date], N days old" in response
8. **Build briefing/synthesis mode** — Template over verified fields (no free-form LLM generation)
9. **Run Phi-4-mini LoRA fine-tune** — Even 500 synthetic pairs + eval vs base+outlines
10. **Replace mock embeddings in eval harness** — Use real sentence-transformers for BERTScore

### Optional / Stretch
11. Tier 2 catalog expansion (15-25 AOIs) with live STAC fallback
12. EarthDial RGB LoRA fine-tune on bootstrapped Indian data
13. Frontend implementation (map, slider, explanation cards)
14. Offline demo hardening (network kill switch test)

---

## Final Answer: Is This System Truthfully Described by Its Own Documentation?

**No.** Specifically:

1. **The production plan's "Phase 0 blockers" (Section 2) are explicitly acknowledged as unfixed** — parser timeout mismatch, M2 not wired, catalog empty, zero LoRA checkpoints. The plan states "nothing in Sections 3–6 starts until items 1–3 above are closed" — yet the repo shows no evidence these are closed.

2. **A hard architectural rule is violated** — Production Plan Section 4: "Do not let the VLM narrate SAR evidence... SAR results get a deterministic, traceable explanation instead." The fusion pipeline in `dispatch_table.py` **explicitly sends SAR images to the VLM for captioning** and synthesizes the output.

3. **All "on the horizon" disaster-officer features (Section 11) are unimplemented** — confidence bands, freshness statements, briefing mode exist only as documentation.

4. **Evaluation numbers are synthetic** — Recall@5=0.167, BERTScore≈0.008 on mock embeddings with fixture data. The harness uses `MockEmbeddingModel` explicitly marked "replace with real model."

5. **The LLM parser is structurally unreachable** — 5s validator timeout, missing torch/CUDA, 100% fallback rate. Fine-tuning claims are moot.

The system is a **well-structured prototype with solid unit tests (M4: 97 passing) and clean module boundaries**, but it **does not function as described** on real data, violates its own stated SAR constraint, and has not shipped any of the differentiating disaster-officer features. It would not survive expert questioning at nationals in its current state.