# Model Dependencies

This repository contains code and configuration only. Model weights are
**not committed** — they are large (multi-GB), publicly hosted, and pulled at
runtime or via a setup script. This is standard practice, not a gap.

| Component | Model | Source | Size | Required for |
|---|---|---|---|---|
| Query parser (real path) | Phi-4-mini-instruct | `microsoft/Phi-4-mini-instruct` (Hugging Face) | ~7.5GB (fp16) / ~2GB (4-bit) | Real NL parsing. Not required for demo — deterministic fallback parser is used by default and is fully tested. |
| VLM (real path) | GeoChat / EarthDial (RGB checkpoint) | see project's model card references | Multi-GB | Real captioning/VQA. Not required for demo — mock HTTP client returns fixture-consistent responses. |
| Retrieval embeddings | sentence-transformers (text) + DOFA (image, optional) | `sentence-transformers` PyPI package / Hugging Face | Varies | Real semantic retrieval ranking. Fixture-based text matching is used by default. |

## Why weights aren't in this repo

- GitHub rejects files over 100MB without Git LFS, and these models are
  gigabytes each.
- Weights are already hosted canonically (Hugging Face) — recommitting them
  here would duplicate a multi-GB download for no benefit.
- The mock/fallback paths are not a placeholder hack — they're tested,
  documented pipeline branches (see `STATUS.md`), so the repo is fully runnable
  and demoable without downloading anything.

## To enable real inference

1. Install the GPU-dependent extras: `pip install torch bitsandbytes accelerate`
2. Set the relevant `mock: false` flags in `config.yaml`
3. The first real run will download the checkpoint automatically via
   `transformers`/`huggingface_hub` (requires network + disk space per the
   table above)