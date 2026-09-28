# SatQuery AI

Agentic remote-sensing vision-language assistant — natural-language and
direct-image-upload analysis of satellite imagery (optical, SAR, and
optical+SAR pairs), built for SIH 2026 Problem Statement 167.

> **Status:** Working prototype, demo-scope. See [STATUS.md](STATUS.md) for
> exactly what's real vs. mocked, and why. See [docs/MODELS.md](docs/MODELS.md)
> for model dependencies.

## What it does

- Natural-language query → structured intent → routed to the right specialist
  pipeline (search, VQA, captioning, change detection, or optical–SAR fusion)
- Direct image/image-pair upload with compatibility checking
- Deterministic, auditable SAR analysis (no hallucination risk on a modality
  with no available pretrained VLM checkpoint)
- Every response includes a structured execution trace, confidence band, and
  data-freshness statement

## Quickstart

```bash
git clone <repo>
cd satquery
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements-lock.txt   # exact versions used for testing
python -m api.main
# → http://localhost:8000/gui
```

This runs the full pipeline against fixture data out of the box — no GPU or
API keys required for the demo path. See STATUS.md for what changes when real
models/data are enabled.

## Architecture

| Module | Purpose | Key Files |
|---|---|---|
| M0 Catalog | Scene metadata storage (SQLite), image caching, multi-source ingestion | `m0_catalog/`, `m0_ingest/` |
| M1 Parser | NL query → `StructuredQuery`; LLM-primary with regex/keyword fallback | `m1_parser/` |
| M2 Retrieval | **Three-stage real pipeline**: SQL metadata filter → text similarity (sentence-transformers + FAISS) → multi-factor score fusion | `m2_retrieval/` |
| M3 VLM | GeoChat 4-bit VLM: caption, VQA; lazy loading, mock-gated | `m3_vlm/` |
| M4 Change | Deterministic SAR feature extraction, optical/SAR change detection | `m4_change/`, `m4_changedetect/` |
| M5 Controller | Agentic dispatch: 5 pipelines, trace building, confidence bands | `m5_controller/` |
| M6 API | FastAPI server: POST /query, POST /upload, GET /gui; replay cache | `api/`, `m6_api/` |

## Testing

```bash
pytest -q
```
**232 passed, 2 skipped** (as of 2026-09-27)

## Problem statement mapping

| PS167 Requirement | Module | Status | Notes |
|---|---|---|---|
| Remote-sensing adaptation (BigEarthNet) | M3 VLM / M0 | Mocked | Training harness exists; no checkpoint produced |
| Single-image VQA (mandatory) | M3 VLM / M5 | Mocked | `pipeline_vqa` → mock HTTP client |
| Captioning (additional single-image) | M3 VLM / M5 | Mocked | `pipeline_caption` → mock HTTP client |
| Multitemporal change description | M4 Change / M5 | Real (deterministic) | NDVI/NDWI/SAR log-ratio, 97 unit tests |
| Cross-modal optical+SAR analysis | M3+M4 / M5 | Mocked+Real | Optical→VLM, SAR→deterministic features |
| Agentic orchestration | M5 Controller | Real | 6 pipelines, full trace, confidence bands |
| Input validation & compatibility | M5 Controller | Real | CRS, footprint, band checks |
| Structured execution trace | M5 Controller | Real | `ExecutionTrace` on every response |
| Confidence estimation | M5 Controller | Real | High/Med/Low bands on every result |
| Downloadable report | M6 API | **Missing** | Not yet implemented |
| Interactive GUI/web app | M6 API | Real | `api/static/index.html` at `GET /gui` |
| GeoTIFF/TIFF handling | M0, M3, M4 | Real | `rasterio`-based throughout |
| **Three-stage retrieval (SQL → text sim → fusion)** | **M2 Retrieval** | **Real (fixture data)** | **sentence-transformers + FAISS, 27 scenes** |

## License

MIT License — see [LICENSE](LICENSE)