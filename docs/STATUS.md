# Project Status

Last verified: 2026-09-27

This file states plainly what is real, what is mocked, and why — so that
anyone evaluating this repo (or running it themselves) knows exactly what
they're looking at. See `docs/MODELS.md` for which model weights this project
depends on and how to enable them.

## What's real right now

- Agentic dispatch and orchestration (`m5_controller/`) — task routing across
  6 pipeline types, fully tested (see test count below).
- **Three-stage retrieval (M2) — SQL metadata filter → text similarity (sentence-transformers) → multi-factor score fusion — running against 27 fixture scenes in catalog.db with FAISS index**
- Deterministic change detection (NDVI/NDWI/SAR log-ratio) — `m4_changedetect/`,
  97 passing unit tests, verified against synthetic arrays with known deltas.
- Deterministic SAR feature extraction — water/built-up fraction, log-ratio —
  no neural model involved, by design (see docs/MODELS.md for why).
- Execution tracing — every pipeline call is logged with latency and
  success/failure, attached to every response.
- Confidence-band computation (High/Medium/Low + reason) on every result.
- Input compatibility checking (CRS, footprint, band presence) for
  single-image, pair, and fusion uploads.
- The web GUI (`api/static/index.html`) — functional, dependency-free,
  served at `GET /gui`.

## What's mocked, and why

| Component | Status | Why | To enable |
|---|---|---|---|
| LLM query parser (Phi-4-mini) | Deterministic regex/keyword fallback active | Requires `torch` + GPU; not available in this environment | Install `torch`+CUDA, set parser timeout appropriately in `config.yaml`, remove mock gate |
| VLM (GeoChat) captioning/VQA | Mock HTTP client returns fixture responses | Requires GPU + `bitsandbytes` 4-bit inference | Set `vlm.mock: false` in `config.yaml`; requires a downloaded checkpoint, see `docs/MODELS.md` |
| Retrieval embeddings (DOFA image embeddings) | Not populated — index uses text embeddings only | DOFA requires GPU for image encoding; text embeddings (sentence-transformers) work on CPU | Run `scripts/build_index.py` with DOFA (requires GPU) |
| Catalog (`catalog.db`) | 27 fixture scenes — synthetic but loaded in real catalog | Ingestion pipeline is written and tested in isolation, not yet run against live Planetary Computer data | `scripts/build_catalog.py --aois config/aois_tier1.yaml` (requires network) |
| LoRA fine-tuning (EarthDial / Phi-4-mini) | Training harness exists (`m3_vlm/train_lora.py`), no checkpoint produced | Requires GPU hours + dataset download | See `docs/MODELS.md` |

## Evaluation numbers — read this before citing any metric from this repo

Numbers currently produced by `evals/run_retrieval_eval.py` and
`m6_api/eval_harness.py` (e.g. Recall@5, BERTScore) are computed against
**27 synthetic fixture scenes** using **real sentence-transformers embeddings** for retrieval ranking, but with a **hash-based mock embedding model** for BERTScore computation. The retrieval pipeline (SQL filter → text similarity → score fusion) is now real and uses FAISS; however, the ground truth and evaluation corpus are synthetic. These numbers validate that the evaluation *harness* and *retrieval pipeline* work — they are not a measurement of real-world retrieval or answer quality on actual satellite data. Do not cite them as system performance on real data. Real numbers require a populated catalog from live Planetary Computer ingestion and a real embedding model for answer quality metrics.

## Test suite

Last run: `pytest -q` → **232 passed, 2 skipped**

## Architecture rationale, briefly

SAR imagery is deliberately never sent to the VLM for captioning — SAR is
processed exclusively through deterministic feature extraction (log-ratio,
water/built-up fraction from backscatter thresholds). This is a design choice,
not a gap: no publicly released SAR-pretrained checkpoint exists for the VLM
backbone used here, and letting a model narrate a modality it wasn't trained
on risks confident-sounding hallucination on exactly the evidence a
flood-monitoring use case depends on most. See `test_fusion_never_calls_vlm_on_sar`
for the regression test enforcing this.