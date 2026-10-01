# SatQuery AI

Agentic remote-sensing vision-language assistant — natural-language and
direct-image-upload analysis of satellite imagery (optical, SAR, and
optical+SAR pairs)

> **Status:** Working prototype, demo-scope. See [docs/STATUS.md](docs/STATUS.md) for
> exactly what's real vs. mocked, and why. See [docs/MODELS.md](docs/MODELS.md)
> for model dependencies. See [docs/SatQuery_AI_Unified_Architecture_v3.md](docs/SatQuery_AI_Unified_Architecture_v3.md)
> for full architecture and claims discipline.

## What it does

- **Natural-language query** → structured intent → routed to the right specialist
  pipeline (Retrieve/search, VQA, captioning, change detection, optical–SAR fusion, SAR change, SAR grounding)
- **Direct image/image-pair upload** with compatibility checking (CRS, footprint, bands)
- **SAR branch with dual path**: CROMA-S1 (radar-native ViT encoder) → projector → shared phi-3 decoder for narratives + **deterministic cross-check** (log-ratio, water/built-up fractions) running in parallel. Disagreement surfaces, confidence drops.
- **Optical branch**: EarthDial (InternViT + Phi-3 mini) with LoRA fine-tuning support
- **Fusion**: Cross-attention fusion of EarthDial optical tokens + CROMA SAR tokens → shared decoder
- **Every response** includes: structured execution trace, confidence band (High/Med/Low + reason), data-freshness statement, SAR path tracking

## Quickstart

```bash
git clone <repo>
cd satquery
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m api.main
# → API at http://localhost:8000
#   GET /health, /config-status, /catalog, /catalog/search
```

This runs the full pipeline against **synthetic fixture data** out of the box — no GPU or
API keys required for the demo path (all models in `mock: true` mode). See docs/STATUS.md for what
changes when real models/data are enabled.

## Frontend (React + Vite)

```bash
cd frontend_ui
npm install
npm run dev   # http://localhost:5173
npm run build # production build to dist/
```

The React frontend (`frontend_ui/`) provides the interactive UI.

## Architecture

| Module | Purpose | Key Files |
|---|---|---|
| **M0 Catalog** | Scene metadata storage (SQLite + FAISS), multi-source ingestion, synthetic test data generator | `m0_catalog/`, `m0_catalog/populate_synthetic.py` |
| **M1 Parser** | NL query → `StructuredQuery`; deterministic regex/keyword fallback (primary), Phi-4-mini LoRA (real) | `m1_parser/` |
| **M2 Retrieval** | **Three-stage real pipeline**: SQL metadata filter → text similarity (sentence-transformers + FAISS) → multi-factor score fusion | `m2_retrieval/` |
| **M3 VLM** | **Optical**: EarthDial `EarthDial_4B_RGB` (InternViT + Phi-3 mini). **SAR**: CROMA-S1 (ViT-Base/Large) → projector → shared decoder. **Fusion**: Cross-attention dual-encoder → shared phi-3 mini decoder | `m3_vlm/earthdial_loader.py`, `m3_vlm/croma_loader.py`, `m3_vlm/fusion_decoder.py`, `m3_vlm/projector.py` |
| **M4 Change** | direct vlm change detection based on modality(sar/optical) ,Deterministic change detection (NDVI/NDWI delta, SAR log-ratio) to cross verify the results. SAR Siamese (CROMA) placeholder | `m4_change/` |
| **M5 Controller** | Agentic dispatch: **7 pipelines** (search, vqa, caption, change, fusion, sar_change, sar_grounding), trace building, confidence bands with SAR cross-check, SAR path tracking | `m5_controller/` |
| **M6 API** | FastAPI server: POST /query, POST /upload, GET /report, GET /catalog, GET /config-status; replay cache | `api/` |

### Model Stack

| Component | Model | Source | Size | Mock Default |
|---|---|---|---|---|
| Query Parser (real) | Phi-4-mini-instruct | `microsoft/Phi-4-mini-instruct` | ~7.5GB (fp16) / ~2GB (4-bit) | ✅ Regex fallback |
| VLM Optical (real) | EarthDial `EarthDial_4B_RGB` | `hiyamdebary/EarthDial` | Multi-GB | ✅ Mock responses |
| SAR Encoder (real) | CROMA-S1 ViT-Base | `antofuller/CROMA` | ~350MB | ✅ Mock CROMA |
| Retrieval Embeddings | sentence-transformers/all-MiniLM-L6-v2 | PyPI / HF | ~80MB | **Real (CPU)** |

## Key Architecture Decisions (from Unified Architecture v3)

1. **SAR never hallucinates**: CROMA-S1 is a radar-native encoder (pretrained on 1M Sentinel-1/2 pairs). Deterministic SAR math (log-ratio, water/built-up fractions) is the **permanent cross-check** — not a stepping stone.
2. **Disagreement surfaced**: When model output ≠ deterministic math → confidence drops, both shown, never silently resolved.
3. **SAR grounding = optical-guided fallback**: SAREval <3% Acc@0.5 for native SAR grounding. Ground on co-registered optical, project box to SAR.
4. **Claims discipline**: No "zero hallucination" on SAR (learned encoder now in loop). No "general India-wide" claims — state exact AOI count/zones. No mock metrics as real.

## Testing

```bash
pytest -q
```
**59 passed** (as of 2026-09-29)

```bash
# Backend modules
pytest m0_catalog/test_catalog.py m3_vlm/test_croma.py m3_vlm/test_earthdial.py m5_controller/test_dispatch_matrix.py m5_controller/test_controller.py -v

# Frontend
cd frontend_ui && npm run lint
```

## Evaluation

```bash
# Retrieval evaluation (requires sentence-transformers model download on first run)
python -m evals.run_retrieval_eval --db ./data/catalog.db --splits ./data/eval_splits.json --tag baseline_202609
```

**Baseline metrics** (64 synthetic scenes, 2 held-out AOIs):
- Recall@5: 33.3%
- Precision@5: 36.3%
- mAP@5: 23.9%
- Latency p50: 142ms, p95: 5376ms

## To Enable Real Inference

1. Install GPU extras: `pip install torch bitsandbytes accelerate`
2. Set `mock: false` in `config.yaml`:
   ```yaml
   vlm:
     mock: false
   croma:
     mock: false
   retrieval:
     mock: false
   ```
3. First run downloads checkpoints via `transformers`/`huggingface_hub` (requires network + disk space)

## External Dependencies

- **Bhoonidhi** (`bhoonidhi.nrsc.gov.in`): ISRO/NRSC data access for Cartosat-2S/RISAT samples — required for real RISAT validation of CROMA 2-channel constraint
- **BigEarthNet.txt** (464K pairs, 9.6M annotations): Primary fine-tuning corpus for EarthDial + CROMA projector
- **VRSBench / RSVQA / CDVQA**: Evaluation benchmarks

## License

MIT License — see [LICENSE](LICENSE)