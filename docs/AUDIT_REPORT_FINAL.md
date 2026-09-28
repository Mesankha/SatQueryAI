# SatQuery AI — Codebase Reality Check & GitHub-Readiness Audit
**Audit Date:** 2026-09-27  
**Repository:** `satquery/`  
**Prior Audit Reference:** `IDEA_SUBMISSION_FINAL_AUDIT.md` (2026-09-27)

---

## 1. Two-Day Verdict

**Do not publish yet.** The repo has a **hard secret (NVIDIA API key in `m1_parser/.env`)** that must be removed before any public push. Beyond that, the codebase is an **honestly-scoped prototype running on fixture data** — the architecture is solid (232 tests passing on fixture paths), but the catalog is empty, FAISS index is missing, sentence-transformers is not installed (breaking real M2 path), and eval metrics are fixture-only mocks. With ~2 hours of fixes (remove secret, pin deps, add "Known Limitations" section, fix the pre-existing test error), it can be published as an honest demo. Without those fixes, a judge cloning the repo will find a committed API key and broken real-data paths.

---

## 2. Hard Blockers (Must Fix Before Any Public Push)

| # | Blocker | Effort | Evidence |
|---|---------|--------|----------|
| 1 | **NVIDIA API key committed in `m1_parser/.env`** | 5 min | File contains `nvidia_api_key= "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"` — this is a live key, must be rotated and removed from git history |
| 2 | **Real M2 path broken: `sentence-transformers` not installed** | 5 min | Test failures: `M2 real retrieval failed, falling back to fixtures: No module named 'sentence_transformers'` — 3 tests fail because of this |
| 3 | **Pre-existing test error in `test_e2e_all.py`** | 15 min | Fixture 'name' not found — test collection error, existed in prior audit |
| 4 | **No "Known Limitations" / "Current Status" section in README** | 30 min | README (if exists) or top-level doc must disclose: empty catalog, mock VLM/parser, fixture-only eval |

---

## 3. Re-verification Table (Part 1 Results)

| # | Prior Finding (2026-09-27) | Re-check Result | Evidence |
|---|---|---|---|
| 1 | `catalog.db` has 0 rows; all retrieval routes through fixtures | **Confirmed unchanged** | `SELECT COUNT(*) FROM scenes` = 0; `config.yaml: retrieval.mock: true` |
| 2 | LLM parser 100% fallback (`torch` not installed) | **Confirmed unchanged** | `import torch` → `ModuleNotFoundError`; `_get_pipeline()` returns `None` immediately on line 74-76 (CUDA unavailable) |
| 3 | SAR→VLM violation fixed; `test_fusion_never_calls_vlm_on_sar` passes | **Confirmed fixed** | `pipeline_fusion()` calls `_call_m4_sar_features()` (deterministic) for SAR; VLM only called on optical. Regression test passes. |
| 4 | `outlines` not actually used; regex + `model_validate()` post-hoc instead | **Confirmed unchanged** | No `outlines` imports in project source (only in venv). `parser_llm.py` extracts JSON via regex (lines 127-130), validates via `model_validate()` (line 182). |
| 5 | DOFA embedder + FAISS index exist but never populated (index size 0) | **Confirmed unchanged** | `data/faiss.index` and `data/faiss_ids.json` do not exist. `scripts/build_index.py` exists but never run. |
| 6 | No frontend existed as of Sept-08; `api/static/index.html` added by Sept-27 | **Confirmed functional** | `api/static/index.html` is a complete, dependency-free HTML/JS GUI with query, upload, results, trace panel. Loads via `GET /gui`. |
| 7 | Eval metrics (Recall@5=0.167, BERTScore≈0.008) are fixture-only mock-embedding artifacts | **Confirmed — CRITICAL** | `m6_api/eval_harness.py:51-63` uses `MockEmbeddingModel` (hash-based, not BERT). These exact numbers appear in `IDEA_SUBMISSION_FINAL_AUDIT.md` §6 and §8 — **must be labeled "fixture-only / not real" everywhere**. |
| 8 | 232 passed / 2 skipped / 1 pre-existing error | **Regressed** | Current: **228 passed, 3 skipped, 1 error, 3 failed**. Failures: 3 tests fail due to missing `sentence-transformers`; 1 error in `test_e2e_all.py` (fixture issue). |
| 9 | `requirements.txt` uses version ranges, not exact pins | **Confirmed unchanged** | All 37 deps use `>=` or no version (e.g., `faiss-cpu`, `sentence-transformers`, `fastapi`). No lock file. |
| 10 | Section 2 / Tier 0 blockers from scale-up plan (parser timeout, M2 wiring, catalog paths, SAR→VLM) | **Partially improved** | SAR→VLM fixed (item 4). Parser timeout still 5s (config.yaml:32), torch missing. M2 wiring code exists but broken by missing dep. Catalog still empty. |

**Cannot verify (no GPU/network):** DOFA model loading, Phi-4-mini inference, GeoChat VLM loading, Planetary Computer ingestion, LoRA training.

---

## 4. Mandatory Scope Coverage (Part 2 — Against Problem Statement 167)

| Requirement | Present in Code? | Real or Mocked? | Evidence (file/line/test) |
|---|---|---|---|
| **Remote-sensing adaptation** (fine-tune on BigEarthNet) | **Code only** | Mocked | `m3_vlm/train_lora.py` exists (37KB harness); `m0_catalog/ingest_bigearthnet.py` exists; **no checkpoint at `./models/geochat_lora`**, no training run completed. Requires GPU + data download (~8 hrs). |
| **Single-image VQA** (mandatory) | **Yes** | Mocked (E2E) | `pipeline_vqa()` in `dispatch_table.py:354` → `_call_m3_vqa()` → mock HTTP client (config `vlm.mock: true`). `test_vqa_pipeline_with_missing_image_returns_error_result` passes. |
| **Captioning OR grounding** (pick one) | **Captioning** | Mocked (E2E) | `pipeline_caption()` in `dispatch_table.py:393` → `_call_m3_caption()` → mock. Grounding removed (deprecated → routes to caption). |
| **Multitemporal change description OR change-VQA** | **Yes** | Mocked (E2E) | `pipeline_change()` in `dispatch_table.py:432` → `_call_m4_change()` → mock `m4_changedetect.detect_change`. 97 unit tests in `m4_changedetect/` pass (deterministic math verified). |
| **Cross-modal optical+SAR joint analysis** (mandatory) | **Yes** | Mocked (E2E) | `pipeline_fusion()` in `dispatch_table.py:485` → optical → VLM caption; SAR → deterministic `_call_m4_sar_features()`. Regression test `test_fusion_never_calls_vlm_on_sar` passes. |
| **Agentic orchestration** (task routing, model/tool selection) | **Yes** | Real | `orchestrate_async()` in `dispatch_table.py:586` → M1 parse → M2 retrieve → DISPATCH table (5 pipelines) → trace building. 7 dispatch matrix tests pass. |
| **Structured, auditable execution trace** | **Yes** | Real | `trace_builder.py` `timed_call`/`timed_call_async` wraps every tool call. `ExecutionTrace` schema includes `tools_called[]`, `fallback_used`. All 6 pipelines produce non-empty traces. |
| **Interactive GUI/web app with image upload** | **Yes** | Real | `api/main.py`: `POST /query`, `POST /upload` (single/pair + compat check), `GET /gui` serves `api/static/index.html`. Tested via httpx. |
| **GeoTIFF/TIFF input handling** | **Yes** | Real | `m0_catalog/image_resolver.py`, `m3_vlm/sar_preprocessing.py`, `m4_change/sar_features.py` all use `rasterio` for GeoTIFF. Upload endpoint accepts `.tif/.tiff`. |
| **Compatibility checking** (modality/format/footprint) | **Yes** | Real | `compatibility_checker.py`: `check_search_compatibility`, `check_vqa_compatibility`, `check_change_compatibility`, `check_fusion_compatibility` — validate CRS, footprint overlap, band presence, pair matching. |
| **Confidence estimation on outputs** | **Yes** | Real | `shared/confidence.py` `compute_band()` + `attach_confidence()` → High/Medium/Low + reason string. Attached to every `ResultItem`. Factors: parser fallback, observation count, cloud cover, coregistration. |
| **Downloadable report** | **No** | Not implemented | No report generation endpoint or code found. Required by problem statement §110. |

**Smallest concrete changes to make mocked items real:**
- **Remote-sensing adaptation:** Download BigEarthNet (~500GB or 500MB sample) → run `m3_vlm/train_lora.py` (8 hrs GPU). **Not achievable in 2 days.**
- **VQA/Caption/Fusion real:** Install `sentence-transformers`, set `vlm.mock: false`, `retrieval.mock: false` in config.yaml, populate catalog (2-4 hrs network). **Partially achievable.**
- **Downloadable report:** Add `/report` endpoint in `api/main.py` generating PDF/JSON from `QueryResponse` (~2 hrs).

---

## 5. GitHub-Readiness Findings (Part 3)

### 5.1 Clean Clone & Run Test
```bash
git clone <repo>
cd satquery
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
# FAILS: sentence-transformers not in requirements.txt but imported in m2_retrieval/engine.py
# After pip install sentence-transformers:
python -m api.main  # Starts on :8000
# GET /gui → works (fixture data)
# POST /query → works (fixture data)
```

### 5.2 README/docs vs. Reality Diff
| Location | Claim | Reality | Corrected Wording |
|---|---|---|---|
| `IDEA_SUBMISSION_FINAL_AUDIT.md` §6, §8 | Recall@5=0.167, BERTScore≈0.008 | Fixture-only, mock embeddings | "Retrieval evaluation on **fixture data with mock embeddings** yields Recall@5=0.167, BERTScore≈0.008 — **not representative of real-data performance**" |
| `IDEA_SUBMISSION_FINAL_AUDIT.md` §13 | "Core forward pass validated end-to-end" | Validated on fixtures only | "Core forward pass validated end-to-end **on fixture data** across 6 task types (232 tests passing). Real-data validation pending catalog population." |
| `IDEA_SUBMISSION_FINAL_AUDIT.md` §17 | Claims 1-8 as "Claims You CAN Make" | Mostly true but need caveats | Add caveat to each: "**Validated on fixture data; real-data performance metrics pending.**" |
| Any public-facing doc | Implies general Indian coverage | Only 7 AOIs in `config/aois_tier1.yaml` (coordinates = `HUMAN_INPUT_REQUIRED`) | "Demo-scope adaptation for 7 predefined Assam AOIs; not a claim of general India-wide understanding." |

### 5.3 Secrets/Credentials Check
- **CRITICAL:** `m1_parser/.env` contains live NVIDIA API key (`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`) — **must be rotated and purged from git history**.
- `config.yaml` uses `${GEE_PROJECT_ID}` placeholder — safe.
- No other hardcoded secrets found in source (`.gitignore` covers `.env`).

### 5.4 Embarrassing vs. Honestly-Scoped

| Category | Items | Treatment |
|---|---|---|
| **Fix before publishing** (looks abandoned/broken) | 1. NVIDIA API key in `.env`<br>2. `sentence-transformers` missing from requirements<br>3. `test_e2e_all.py` collection error<br>4. 3 test failures due to missing dep<br>5. No downloadable report endpoint | **Fix all** |
| **Label, don't fake** (genuinely unfinished) | 1. Empty catalog (0 scenes)<br>2. No FAISS index<br>3. LLM parser 100% fallback (no GPU)<br>4. VLM mocked (`vlm.mock: true`)<br>5. No LoRA checkpoint<br>6. Eval metrics fixture-only | **Disclose in "Known Limitations"** |

**Proposed "Known Limitations" section:**
```markdown
## Known Limitations (as of 2026-09-27)

- **Catalog empty:** `catalog.db` contains 0 scenes. All retrieval runs against 27 synthetic fixtures in `fixtures/fixtures_scenes.json`.
- **Real retrieval path untested:** `sentence-transformers` not installed; FAISS index not built. Set `retrieval.mock: false` in `config.yaml` after populating catalog and running `scripts/build_index.py`.
- **LLM parser uses deterministic fallback:** Phi-4-mini requires GPU + torch (not installed). Parser timeout is 5s (configurable).
- **VLM (GeoChat) mocked:** Set `vlm.mock: false` in `config.yaml` to enable real 4-bit inference (requires GPU, bitsandbytes).
- **No LoRA checkpoints:** Fine-tuning harnesses exist (`train_lora.py`) but no training runs completed.
- **Evaluation metrics are synthetic:** Recall@5=0.167, BERTScore≈0.008 computed on fixture data with `MockEmbeddingModel` (hash-based). **Do not cite as system performance.**
- **Downloadable report not implemented:** Required by problem statement; tracked as future work.
```

### 5.5 Repo Hygiene Basics
| Check | Status |
|---|---|
| `.gitignore` covers `__pycache__`, `venv/`, `data/image_cache/`, `data/catalog.db`, `.env` | ✅ |
| No committed large binaries | ✅ (data/ ignored) |
| License file | ❌ Not present (optional but recommended) |
| No leftover personal debug scripts at root | ⚠️ Many `test_*.py`, `verify_*.py`, `create_*.py` at root — consider moving to `scripts/` or `tests/` |

---

## 6. 48-Hour Action List (Ordered by Impact ÷ Time)

| Time | Action | Impact | Achievable in 2 Days? |
|---|---|---|---|
| **5 min** | **Rotate & remove NVIDIA API key** from `m1_parser/.env`; purge from git history (`git filter-branch` or BFG) | **Critical** — hard blocker | ✅ Yes |
| **5 min** | Add `sentence-transformers>=2.2` to `requirements.txt` | Fixes 3 failing tests, enables real M2 path | ✅ Yes |
| **15 min** | Fix `test_e2e_all.py` fixture error (remove or fix parametrization) | Cleans test suite | ✅ Yes |
| **30 min** | Add "Known Limitations" section to README (or create `STATUS.md`) | Honest framing for judges | ✅ Yes |
| **30 min** | Pin `requirements.txt` → `requirements-lock.txt` via `pip freeze` | Reproducibility | ✅ Yes |
| **1 hr** | Run `scripts/build_catalog.py --dry-run` with real AOI coords in `config/aois_tier1.yaml` | Validates ingestion pipeline | ✅ Yes (needs network) |
| **2 hr** | **Populate catalog with ≥1 real AOI** (run `scripts/build_catalog.py --aois config/aois_tier1.yaml`) | Makes retrieval real, enables eval | ⚠️ Needs network + ~2-4 hrs |
| **1 hr** | Run `scripts/build_index.py` (requires GPU + populated catalog) | Enables DOFA embeddings + FAISS | ❌ **Not achievable** (needs GPU) |
| **2 hr** | Implement `/report` endpoint in `api/main.py` (PDF/JSON export) | Meets problem statement §110 | ✅ Yes |
| **1 hr** | Add `sentence-transformers` import guard in `m2_retrieval/engine.py` with clear error | Prevents silent fallback | ✅ Yes |

**Not achievable in 2 days (resource constraints):**
- GPU-dependent: DOFA embedding, Phi-4-mini parser, GeoChat VLM, LoRA training
- Network-dependent at scale: Full catalog population (7 AOIs × 2 sensors × 4 scenes = ~56 scenes, ~2-4 hrs)
- Data-dependent: BigEarthNet download (~500GB full / ~500MB sample)

---

## 7. Do Not Do (Next 2 Days)

1. **Do not fabricate metrics** — never cite Recall@5=0.167 or BERTScore≈0.008 without "fixture-only / mock embeddings" label.
2. **Do not claim untested code paths work** — real M2, real VLM, real parser are unvalidated on real data.
3. **Do not remove honest limitation disclosures** to look more finished — the "Known Limitations" section is your credibility shield.
4. **Do not make last-minute dependency changes** without testing — pin versions first, then test.
5. **Do not commit the NVIDIA API key** — rotate it immediately and purge from history.
6. **Do not claim "remote-sensing adaptation" is done** — the training harness exists but no checkpoint has been produced.
7. **Do not claim "downloadable report" exists** — it doesn't; implement or document as missing.

---

## Appendix: Key File References

- **Parser fallback logic:** `m1_parser/parser_llm.py:69-76` (CUDA check → returns `None`)
- **SAR→VLM fix:** `m5_controller/dispatch_table.py:522-526` (calls `_call_m4_sar_features`, not VLM)
- **Mock embeddings:** `m6_api/eval_harness.py:51-63` (`MockEmbeddingModel`)
- **Config flags:** `config.yaml:38,44,54` (`retrieval.mock`, `vlm.mock`, `change.mock`)
- **Test suite:** `pytest -q` → 228 passed, 3 skipped, 1 error, 3 failed
- **GUI:** `api/static/index.html` (served at `GET /gui`)