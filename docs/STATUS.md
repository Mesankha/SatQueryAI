# Project Status

Last verified: 2026-09-29

This file states plainly what is real, what is mocked, and why — so that
anyone evaluating this repo (or running it themselves) knows exactly what
they're looking at. See `docs/MODELS.md` for which model weights this project
depends on and how to enable them.

## What's real right now

- Agentic dispatch and orchestration (`m5_controller/`) — task routing across
  7 pipeline types (search, vqa, caption, change, fusion, sar_change, sar_grounding), fully tested.
- **Three-stage retrieval (M2) — SQL metadata filter → text similarity (sentence-transformers) → multi-factor score fusion — running against 64 synthetic scenes in catalog.db with FAISS index**
- Deterministic change detection (NDVI/NDWI/SAR log-ratio) — `m4_change/`,
  verified against synthetic arrays with known deltas.
- **SAR branch now has dual path: CROMA-S1 learned encoder + deterministic cross-check** — CROMA-S1 (ViT-Base) encodes SAR patches → projector → EarthDial's shared Qwen decoder. Deterministic features (water/built-up fraction, log-ratio) run in parallel as permanent cross-check. Disagreement lowers confidence.
- Fusion pipeline: EarthDial InternViT (optical) + CROMA-S1 (SAR) → cross-attention fusion → shared Qwen decoder. Three-field output: factual / deterministic / model-derived.
- Execution tracing — every pipeline call is logged with latency and
  success/failure, attached to every response. SAR path explicitly named in trace (`"CROMA-S1 + deterministic cross-check"` or `"deterministic-only fallback"`).
- Confidence-band computation (High/Medium/Low + reason) on every result, including SAR cross-check disagreement.
- Input compatibility checking (CRS, footprint, band presence) for
  single-image, pair, and fusion uploads.
- **React-based frontend** in `satquery/frontend_ui/` (Vite + React) — served separately, calls FastAPI endpoints.

## What's mocked, and why

| Component | Status | Why | To enable |
|---|---|---|---|
| LLM query parser (Phi-4-mini) | Deterministic regex/keyword fallback active | Requires `torch` + GPU; not available in this environment | Install `torch`+CUDA, set parser timeout appropriately in `config.yaml`, remove mock gate |
| VLM Optical (EarthDial) captioning/VQA | Mock returns fixture responses | Requires GPU + `bitsandbytes` 4-bit inference | Set `vlm.mock: false` in `config.yaml`; requires downloaded checkpoint, see `docs/MODELS.md` |
| SAR Encoder (CROMA-S1) | Mock CROMA path enabled | Requires GPU + weights download | Set `croma.mock: false` in `config.yaml`; requires `antofuller/CROMA` weights |
| Retrieval embeddings (sentence-transformers) | **Real — sentence-transformers/all-MiniLM-L6-v2 loaded on CPU** | Text embeddings work on CPU; image embeddings (DOFA) would require GPU | Already enabled for text retrieval; for image embeddings run with DOFA on GPU |
| Catalog (`catalog.db`) | 64 synthetic scenes loaded in real SQLite catalog | Ingestion pipeline written and tested; run against live data when Bhoonidhi access approved | `python -m m0_catalog.run_ingestion ingest --max-bigearthnet 1000` (requires network) |
| LoRA fine-tuning (EarthDial / Phi-4-mini / CROMA projector) | Training harness exists (`m3_vlm/train_lora.py`), no checkpoint produced | Requires GPU hours + dataset download | See `docs/MODELS.md` |

## Evaluation numbers — read this before citing any metric from this repo

Numbers currently produced by evaluation scripts (e.g. Recall@5, BERTScore) are computed against
**64 synthetic fixture scenes** using **real sentence-transformers embeddings** for retrieval ranking. The retrieval pipeline (SQL filter → text similarity → score fusion) is now real and uses FAISS; however, the ground truth and evaluation corpus are synthetic. These numbers validate that the evaluation *harness* and *retrieval pipeline* work — they are not a measurement of real-world retrieval or answer quality on actual satellite data. Do not cite them as system performance on real data. Real numbers require a populated catalog from live Planetary Computer/Bhoonidhi ingestion and a real embedding model for answer quality metrics.

## Test suite

Last run: `pytest -q` → **59 passed**

## Architecture rationale, briefly

SAR imagery now has a **learned encoder (CROMA-S1)** alongside deterministic features. CROMA-S1 is a radar-native ViT pretrained on 1M Sentinel-1/2 pairs (NeurIPS 2023). It produces patch embeddings → projector → shared Qwen decoder for narratives. **Deterministic SAR math (log-ratio, water/built-up fractions) remains the permanent cross-check** — disagreement between model and math is surfaced, confidence drops, never silently resolved. This is a stronger design than "deterministic only" (loses expressiveness) or "model trusted blindly" (hallucination risk). SAR grounding uses optical-guided fallback (SAREval <3% Acc@0.5 for native SAR grounding). See `docs/SatQuery_AI_Unified_Architecture_v3.md` §3.3–3.5, §4, §6 for details.