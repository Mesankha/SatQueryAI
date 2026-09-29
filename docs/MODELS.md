# Model Dependencies

This repository contains code and configuration only. Model weights are
**not committed** — they are large (multi-GB), publicly hosted, and pulled at
runtime or via a setup script. This is standard practice, not a gap.

| Component | Model | Source | Size | Required for |
|---|---|---|---|---|
| Query parser (real path) | Phi-4-mini-instruct | `microsoft/Phi-4-mini-instruct` (Hugging Face) | ~7.5GB (fp16) / ~2GB (4-bit) | Real NL parsing. Not required for demo — deterministic fallback parser is used by default and is fully tested. |
| VLM Optical (real path) | EarthDial `EarthDial_4B_RGB` | `hiyamdebary/EarthDial` (Hugging Face) | Multi-GB | Real optical captioning/VQA/fusion. Not required for demo — mock returns fixture-consistent responses. |
| SAR Encoder (real path) | CROMA-S1 (ViT-Base) | `antofuller/CROMA` (Hugging Face) | ~350MB (base) / ~1.2GB (large) | Real SAR encoding → projector → shared decoder. Not required for demo — mock CROMA path enabled by default. |
| Retrieval embeddings | sentence-transformers/all-MiniLM-L6-v2 | `sentence-transformers` PyPI / Hugging Face | ~80MB | Real semantic retrieval ranking. Fixture-based text matching is used by default. |

## Why weights aren't in this repo

- GitHub rejects files over 100MB without Git LFS, and these models are
  gigabytes each.
- Weights are already hosted canonically (Hugging Face) — recommitting them
  here would duplicate a multi-GB download for no benefit.
- The mock/fallback paths are not a placeholder hack — they're tested,
  documented pipeline branches, so the repo is fully runnable
  and demoable without downloading anything.

## To enable real inference

1. Install the GPU-dependent extras: `pip install torch bitsandbytes accelerate`
2. Set the relevant `mock: false` flags in `config.yaml`:
   - `vlm.mock: false` (EarthDial optical)
   - `croma.mock: false` (CROMA-S1 SAR encoder)
   - `retrieval.mock: false` (real M2 retrieval)
3. The first real run will download the checkpoint automatically via
   `transformers`/`huggingface_hub` (requires network + disk space per the
   table above)

## Architecture note: SAR now has a learned path

The SAR branch now uses **CROMA-S1** (radar-native ViT encoder) + **projector** + **EarthDial's shared Qwen decoder** for model-derived SAR narratives. Deterministic SAR features (log-ratio, water/built-up fractions) run as a permanent cross-check. When CROMA is unavailable, the system falls back to deterministic-only mode with explicit trace labeling. See `docs/SatQuery_AI_Unified_Architecture_v3.md` §3.3–3.5 for details.