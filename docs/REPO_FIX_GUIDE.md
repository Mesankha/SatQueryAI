# SatQuery AI — Repo Fix & Publish Guide
Generated from `AUDIT_REPORT_FINAL.md`. Follow in order. Items marked **[BLOCKING]**
must be done before `git push`. Items marked **[POLISH]** make the repo read as
professionally engineered rather than a scramble — they involve zero fabrication,
only honest labeling and structure.

---

## STEP 0 — Do this before anything else, outside of git entirely

**Rotate the NVIDIA API key.** It is currently live in `m1_parser/.env` and has
now also been pasted into a shared document. Go to the NVIDIA NGC / API console,
revoke `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`,
issue a new one, and put the new one **only** in a local `.env` that is
gitignored. Do not proceed to git operations until this is done — if this repo
was ever `git add`ed with the old key present, the key is in your git history
even if you delete the file now (Step 1 covers purging history, but rotation
matters regardless of whether you succeed at purging).

---

## STEP 1 — Hard blockers **[BLOCKING]**

### 1.1 Purge the key from git history
If `m1_parser/.env` was ever committed:
```bash
# Confirm whether it's in history at all first:
git log --all --full-history -- "m1_parser/.env"
```
If that returns anything, use the `git filter-repo` tool (preferred over
`filter-branch`, which is deprecated and slower):
```bash
pip install git-filter-repo
git filter-repo --path m1_parser/.env --invert-paths
```
If it was **never** committed (only sitting on disk untracked), you just need
`.gitignore` to keep catching it — confirm:
```bash
git check-ignore -v m1_parser/.env
```
If nothing prints, add `*.env` and `m1_parser/.env` explicitly to `.gitignore`.

### 1.2 Fix the missing dependency
```bash
echo "sentence-transformers>=2.2.0" >> requirements.txt
```
Confirm the 3 failing tests now pass:
```bash
pytest -q m2_retrieval/
```

### 1.3 Fix the `test_e2e_all.py` collection error
The audit found a fixture `'name'` lookup failure. Have your agent locate the
exact fixture reference and either fix the fixture data or fix the test's
parametrization — do not just skip/xfail it silently, since a silently-skipped
test that used to run is worse for the "does this actually work" question than
a visible, understood failure.

### 1.4 Re-run the full suite and record the real number
```bash
pytest -q > test_output.txt
```
Whatever number comes out (228 passed / X failed / etc.) is the number that
goes in your README — not the old 232, not a rounded-up guess. This is the
easiest scientific-honesty win available to you: an exact, current, verifiable
test count.

---

## STEP 2 — Repo structure cleanup **[POLISH]**

A repo that looks professionally maintained has a predictable shape. Right now
you have `test_*.py`, `verify_*.py`, `create_*.py` scattered at root — this is
the single biggest visual "this was a scramble" signal, and it's also the
cheapest to fix.

```
satquery/
├── README.md                    # see Step 4
├── STATUS.md                    # see Step 3 — Known Limitations, promoted to its own file
├── LICENSE                      # pick one — MIT is standard for hackathon repos
├── requirements.txt
├── requirements-lock.txt        # pip freeze output, see 2.3
├── .gitignore
├── .env.example                 # see 2.2 — NOT .env
├── config.yaml
├── config/
├── data/                        # gitignored contents, .gitkeep to preserve the dir
│   └── .gitkeep
├── m0_catalog/ ... m6_api/       # existing modules, unchanged
├── fixtures/
├── evals/
├── scripts/
│   ├── build_catalog.py
│   ├── build_index.py
│   ├── verify_setup.py          # move verify_*.py here
│   └── ...
├── tests/                        # move test_*.py at root here if they aren't
│                                  # already inside their module's own test dir
└── docs/
    ├── ARCHITECTURE.md           # optional — see Step 5
    └── MODELS.md                 # see Step 6 — answers your weights question
```

Have your agent do the moves with `git mv`, not `mv`, so history is preserved
per-file:
```bash
git mv verify_setup.py scripts/verify_setup.py
```

### 2.2 Add `.env.example`
Never commit `.env`. Commit a template instead — this is the standard,
expected pattern and it's a positive signal, not a workaround:
```bash
# m1_parser/.env.example
nvidia_api_key=""
# Obtain from https://build.nvidia.com — required only if running the real
# Phi-4-mini path. Not required for the mock/fallback path used in the demo.
```

### 2.3 Pin dependencies
```bash
pip freeze > requirements-lock.txt
```
Keep `requirements.txt` as loose ranges for readability, but reference
`requirements-lock.txt` in the README as "exact versions used for testing" —
this directly answers the audit's reproducibility flag.

---

## STEP 3 — `STATUS.md`: the honesty document that actually helps you **[POLISH]**

Don't bury this as a README subsection — give it its own file and link it
prominently from the README. A dedicated status file reads as "this team
tracks their own state rigorously," which is a stronger signal than a buried
disclaimer.

```markdown
# Project Status

Last verified: <date you actually run this>

This file states plainly what is real, what is mocked, and why — so that
anyone evaluating this repo (or running it themselves) knows exactly what
they're looking at. See `docs/MODELS.md` for which model weights this project
depends on and how to enable them.

## What's real right now

- Agentic dispatch and orchestration (`m5_controller/`) — task routing across
  6 pipeline types, fully tested (see test count below).
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
| Retrieval embeddings (DOFA + FAISS) | Not populated — index size 0 | Requires populated catalog + GPU embedding pass | Run `scripts/build_catalog.py`, then `scripts/build_index.py` |
| Catalog (`catalog.db`) | 0 rows — all retrieval runs against `fixtures/fixtures_scenes.json` (27 synthetic scenes) | Ingestion pipeline is written and tested in isolation, not yet run against live data | `scripts/build_catalog.py --aois config/aois_tier1.yaml` |
| LoRA fine-tuning (EarthDial / Phi-4-mini) | Training harness exists (`m3_vlm/train_lora.py`), no checkpoint produced | Requires GPU hours + dataset download | See `docs/MODELS.md` |

## Evaluation numbers — read this before citing any metric from this repo

Numbers currently produced by `evals/run_retrieval_eval.py` and
`m6_api/eval_harness.py` (e.g. Recall@5, BERTScore) are computed against
**27 synthetic fixture scenes** using a **hash-based mock embedding model**,
not a real sentence-transformer or vision-language embedding. These numbers
validate that the evaluation *harness* works — they are not a measurement of
real-world retrieval or answer quality. Do not cite them as system
performance. Real numbers require a populated catalog and a real embedding
model (see table above).

## Test suite

Last run: `pytest -q` → **[fill in the exact number from Step 1.4]**

## Architecture rationale, briefly

SAR imagery is deliberately never sent to the VLM for captioning — SAR is
processed exclusively through deterministic feature extraction (log-ratio,
water/built-up fraction from backscatter thresholds). This is a design choice,
not a gap: no publicly released SAR-pretrained checkpoint exists for the VLM
backbone used here, and letting a model narrate a modality it wasn't trained
on risks confident-sounding hallucination on exactly the evidence a
flood-monitoring use case depends on most. See `test_fusion_never_calls_vlm_on_sar`
for the regression test enforcing this.
```

---

## STEP 4 — README rewrite **[POLISH, but grounded in the audit's exact findings]**

Structure that reads as professional without overclaiming:

```markdown
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

\`\`\`bash
git clone <repo>
cd satquery
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements-lock.txt   # exact versions used for testing
python -m api.main
# → http://localhost:8000/gui
\`\`\`

This runs the full pipeline against fixture data out of the box — no GPU or
API keys required for the demo path. See STATUS.md for what changes when real
models/data are enabled.

## Architecture

[one diagram or the module table — pull from your existing docs, don't
regenerate more claims than what's in the audit]

## Testing

\`\`\`bash
pytest -q
\`\`\`
[exact current number]

## Problem statement mapping

[table: PS167 requirement → which module implements it → real/mocked status
→ link to STATUS.md row — this directly shows a judge you've read your own
rubric, which is a strong signal on its own]

## License

[pick one]
```

---

## STEP 5 — `docs/MODELS.md`: answers your weights question directly **[POLISH]**

This is the file that resolves "do I need to upload the VLM/parser weights."
You don't, and this file is how you say so without it looking like an
omission:

```markdown
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
```

---

## STEP 6 — On "generating sample data so it looks tested"

Here's the version of this I'll help with, and why it's different from what
might have been in your head:

**Do this:** generate additional *clearly-labeled synthetic fixtures* —
more entries in `fixtures/fixtures_scenes.json`, more benchmark queries in
`fixtures/fixtures_queries.json` — so the demo has more variety to show off
during a live walkthrough, and so the test suite covers more edge cases.
Label them exactly like the existing 27: as fixtures, generated, for offline
testing. This is genuinely useful — it makes your test coverage claim
stronger and your live demo richer, and it's honest because the file's own
name and location already say "fixture."

**Don't do this:** creating output logs, screenshots, or "results" that are
framed as if a real model run happened when it didn't. If your agent proposes
anything like a fabricated training-run log or a captioning output attributed
to "GeoChat" that's actually hand-written, stop and don't include it — beyond
the ethical issue, it's a bigger risk to you than the gap it's covering,
because SIH judges routinely ask "walk me through this specific number" live,
and a fabricated one falls apart immediately under that question while an
honest "this is fixture-validated, here's the real-data path" survives it.

If you want richer *demo* material specifically, the honest version is: run
the mock/fallback paths against a wider variety of real natural-language
queries (not pre-scripted ones) live in front of whoever's evaluating you —
this shows the deterministic fallback architecture actually working under
unscripted input, which is a real, creditable thing to demonstrate.

---

## Execution order for your agent

1. Step 0 (you, manually, right now)
2. Step 1.1–1.4 (blocking fixes)
3. Step 2 (structure + `.env.example` + lockfile)
4. Step 3 (`STATUS.md`)
5. Step 5 (`docs/MODELS.md`)
6. Step 4 (README, last, since it links to STATUS.md and MODELS.md)
7. Step 6, if time remains — additional labeled fixtures only
8. Final: re-run `pytest -q` one more time, confirm the number in README/STATUS
   matches reality, then `git add`, review the diff for anything sensitive
   one more time, commit, push.
