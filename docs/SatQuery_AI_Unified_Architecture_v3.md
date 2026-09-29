# SatQuery AI — Unified Architecture & Codebase Upgrade Plan (v3)
## Reconciles: deep-research-report.md, production_scaleup_plan_v2.md, and the CROMA-S1 encoder change
### Team: ZENXNOVA — SIH 2026, PS 167 (ISRO/SAC)

---

## 0. Why this document exists, and the one rule that governs everything in it

Three sources fed into this plan, and they disagree in places:

- **`deep-research-report.md`** — describes the system in generic terms ("a vision-language model," "GLIP or Grounding DINO," "BLIP or ViT+Transformer"). It is a good description of *what each module must do*, but it never commits to specific models, so it is safe to treat as the **functional spec**, not the implementation.
- **`production_scaleup_plan_v2.md`** — commits to specific decisions (Phi-4-mini-instruct for parsing, EarthDial for the VLM, `transformers.pipeline` vs. Ollama, specific AOI lists) based on what the codebase *actually did* as of its audit date (2026-09-08).
- **The CROMA-S1 change** — replaces "no SAR narrative capability at all" (the v2 plan's stance, taken because no SAR checkpoint existed for EarthDial) with "SAR narrative capability through a purpose-built radar-native encoder," because a real, open, pretrained option (CROMA) was found that didn't exist in the v2 plan's search.

**The governing rule for this document: no fact, model name, or number is hardcoded as permanent.** Every model choice below is written as *"current best-known option, replace if a better one is verified."* Where the v2 plan said "no SAR checkpoint exists, so don't build SAR narration," that conclusion is now **superseded**, not deleted — it's preserved below as a historical decision with the reason it changed, because the same reasoning process (verify before committing, prefer honest fallback over hallucination) is what produced both the original SAR restriction and its later reversal. That reasoning process — not any specific model name — is the actual invariant.

This document is written to be **re-run, not re-read once**. Section 1 gives a repeatable verification procedure. Anyone on the team — or a future you, six weeks from now — should be able to follow it to check whether any component in this stack is still the best available option, without needing this exact conversation's context.

---

## 1. The anti-hardcoding procedure (read this before touching any model choice)

Every model/library/dataset decision in Sections 2–5 was reached by asking the same five questions. When a "better" option is proposed later — by a teammate, a new paper, or your own research — run it through the same five questions before swapping anything in:

1. **Is it real and accessible?** Confirm the weights/dataset are actually downloadable today, not announced-but-unreleased (this is exactly the mistake the v2 plan's audit caught with the "full Indian dataset" ask, and exactly what the earlier EarthDial-SAR search caught — the paper *described* a SAR training stage, but no checkpoint was ever released).
2. **What can it actually do, precisely?** Don't infer capability from a paper's abstract. CROMA is a clear example: it is a **pure vision encoder** (contrastive + masked-autoencoder pretraining) — no language head, no VQA/captioning/grounding output. Confusing "has SAR in its pretraining data" with "can describe a SAR image in English" is the exact error this whole architecture is designed to avoid.
3. **What does it cost to integrate, on your actual hardware?** State the GPU memory footprint, the fine-tuning method (full FT vs. LoRA vs. QLoRA vs. frozen-encoder + trainable head), and whether it's a same-day job or a multi-day one. If this isn't known, it isn't a decision yet — it's a hope.
4. **What's the honest fallback if it underperforms or fails to load?** Every model in this stack must degrade to a deterministic, non-hallucinating fallback. If you can't state the fallback, the model isn't ready to route into the live dispatch table.
5. **What's the regression test that would catch a silent violation of the design rule this component exists to enforce?** (E.g., "SAR never reaches a model that wasn't pretrained on SAR" needs a test that fails if `pipeline_fusion()` ever calls an optical-only VLM on a SAR tile again.)

Anything that fails question 1 or 2 does not get named in a judge-facing claim, in code comments, or on a slide — regardless of how promising it looked in a search result. Anything that passes 1–2 but is unverified on 3–5 gets built behind a flag with a fallback, not shipped as the primary path.

---

## 2. Functional architecture (from the research report — the part that doesn't change)

This is the task-level spec every implementation choice below must satisfy. It is deliberately model-agnostic: swapping EarthDial for a different VLM, or CROMA for a different SAR encoder, must not require rewriting this section.

### 2.1 Input scope (fixed by the official problem statement — not negotiable)
- **Single image:** one optical/multispectral or SAR image → captioning, VQA, or text-guided region grounding.
- **Cross-modal pair:** co-registered optical + SAR of the same area → joint analysis.
- **Bi-temporal pair:** two images of the same area, different times → change detection, change description, change-VQA.
- **Formats:** GeoTIFF/TIFF for real imagery; PNG/JPEG accepted only for public benchmark inputs (VRSBench, RSVQA, CDVQA), never for the ISRO/SAC evaluation set.

### 2.2 Task modules (what must exist, independent of which model implements it)

| Module | Function | Mandatory? |
|---|---|---|
| M1 — Query Parser | NL query → structured task + parameters (location, time, sensor, object, task type) | Yes (implicit — nothing routes without it) |
| M2 — Retrieval | Structured query → ranked candidates from an indexed catalog | Only for the retrieval entry path (Path A); not required by the official rubric |
| M3 — VQA / Captioning | Single-image question answering (mandatory) and either captioning or grounding (pick one — team has committed to captioning) | Yes |
| M4 — Change Understanding | Bi-temporal change detection, change description, or change-VQA | Yes (at least one of description/VQA) |
| M5 — Agentic Controller | Interprets query, validates inputs, selects M2–M4/fusion module, executes, assembles output, emits execution trace | Yes |
| M6 — Optical–SAR Fusion | Joint reasoning over a co-registered optical+SAR pair | Yes |
| M7 — Reporting/GUI | Upload, query box, results display, confidence, execution trace, downloadable report | Yes |

### 2.3 Preprocessing requirements (unchanged from the research report; these are format/physics constraints, not model choices)
- Ingest via GDAL/rasterio; validate projection + geotransform metadata for co-registration.
- Optical: reflectance normalization, cloud masking/flagging.
- SAR: radiometric calibration → intensity/dB scale (never raw digital numbers fed to any model).
- Cross-modal pairs: resample to matching resolution, or explicit multi-resolution fusion.
- Bi-temporal pairs: confirm same footprint/CRS; register precisely if not already aligned.
- Tile to model-appropriate sizes — **this is now genuinely model-dependent, not a single fixed number** (see Section 3.3 — CROMA and EarthDial want different input sizes, and this must stay true even if the specific models change again).
- Format validation: GeoTIFF/TIFF primary; reject or explicitly convert anything else; verify image count matches the detected task (e.g., reject a change-detection request with only one image, not silently proceed).

### 2.4 Output contract (mandatory, from the official problem statement — Section 10.2 of the scale-up plan)
Every response, from either entry path, must carry four **separately labeled** fields — never concatenated into one paragraph:
1. **Textual answer** — the model-derived response.
2. **Spatial evidence** — overlays, masks, bounding boxes, or change heatmaps, where applicable.
3. **Confidence estimate** — a calibrated score or High/Medium/Low band with a one-line reason.
4. **Execution trace** — task selected, model/tool names and versions, key parameters — structured data, not prose.

Within the textual answer, three sub-categories must stay visibly distinct: **factual metadata** (e.g., capture date), **deterministic sensor-derived statements** (e.g., SAR threshold math), and **AI-model-derived statements** (e.g., VLM captioning output). This separation is a named audit finding (F2) in the scale-up plan and a mandatory-scope item — it is not optional polish.
## 3. Current model stack (best-known-option-as-of-today — expected to change)

Each entry below follows the same shape: **what it is → why it's currently the pick → its hard constraints → its fallback → the test that guards it.** This shape is the reusable part. When a component is replaced, replace the whole block, not just the model name.

### 3.1 Query Parser (M1)

- **Current pick:** Phi-4-mini-instruct, LoRA fine-tuned on a synthetic (query → StructuredQuery JSON) dataset.
- **Why:** Microsoft ships an official LoRA recipe for this exact model; native structured tool-call format matches the `StructuredQuery` output target; QLoRA-feasible on 8GB+ GPUs.
- **Open implementation question (unresolved — resolve before building on top of it):** the codebase's actual generation mechanism is raw `transformers.pipeline` + regex JSON extraction + post-hoc Pydantic validation, not the constrained-decoding library (`outlines`) that documentation and prior claims described. Two honest paths, pick one and make every downstream doc/slide match:
  - (a) Actually integrate `outlines` (or an equivalent grammar-constrained decoder) — restores the stronger structural guarantee, moderate effort.
  - (b) Keep the current mechanism, correct all claims to "schema-validated post-hoc, with a deterministic regex fallback guaranteeing valid output."
  Do not let this stay ambiguous — it changes what timeout/latency fix is even correct (see Section 6.1).
- **Fallback chain (three deep, all must be tested independently):** fine-tuned Phi-4-mini → base Phi-4-mini (+ `outlines` or regex, per whichever is chosen above) → deterministic regex/keyword extraction. Every fallback tier must still return a valid `StructuredQuery`, never a raw error.
- **Guard test:** one test per fallback tier, forcing each upstream tier to fail and confirming the next tier fires and still produces schema-valid output.
- **Replace this when:** a smaller/faster/more accurate instruction model with native structured output ships and passes the Section 1 checklist — don't replace on hype; replace after checking hardware fit and running the same field-level accuracy comparison used to validate Phi-4-mini.

### 3.2 Vision-Language Backbone — Optical Branch (part of M3, M4-optical, M6)

- **Current pick:** EarthDial (CVPR 2025), `EarthDial_4B_RGB` checkpoint, LoRA fine-tuned.
- **Why:** purpose-built remote-sensing VLM with a released RGB checkpoint; InternViT-based vision encoder + Qwen language decoder already integrated as one model, so you get encoder+LLM together rather than having to wire them yourself.
- **Hard constraint verified, not assumed:** only three checkpoints are public — `EarthDial_4B_RGB`, `EarthDial_4B_MS`, `EarthDial_4B_Methane_UHI`. **No SAR checkpoint is public**, even though the paper describes an internal "Stage 3: Multispectral and SAR fine-tuning" stage. This is why SAR narration was previously excluded from EarthDial entirely — not a design preference, a hard fact about what's downloadable.
- **Zero-shot generalization warning (verified via an independent benchmark, Landsat30-AU, arXiv 2508.03127):** out-of-the-box EarthDial scored only 0.07 SPIDEr captioning and 0.48 VQA accuracy on held-out Australian imagery. This justifies fine-tuning but is also a hard warning against overclaiming — describe fine-tuned performance as "adapted to N named AOIs across M zones," never "general India-wide understanding."
- **Fallback:** deterministic explanation generator (template over precomputed NDVI/NDWI/change-score fields) — this is already the system's primary output path, so falling back to it costs nothing structurally if the fine-tuned VLM underperforms or fails to load.
- **Guard test:** go/no-go checkpoint — if fine-tuned RGB captions aren't measurably better than base-checkpoint captions by a fixed date, the system falls back to the deterministic generator for the demo, no exception.
- **Replace this when:** EarthDial releases a public SAR-capable checkpoint (re-run Section 1 against it before assuming it's better than the CROMA-S1 route below — a same-family SAR checkpoint might integrate more simply, but only if it's real and actually radar-native, not an optical checkpoint fine-tuned on SAR-as-photos), or when a stronger general remote-sensing VLM with an open RGB checkpoint appears.

### 3.3 Vision Encoder — SAR Branch (part of M3, M4-SAR, M6) — the actual change in this revision

- **Current pick:** CROMA-S1 (Contrastive Radar-Optical Masked Autoencoders, NeurIPS 2023), ViT-B or ViT-L, `antofuller/CROMA` weights.
- **Why this replaces "deterministic-math-only SAR":** the earlier plan (Section 4 of the scale-up plan) routed SAR through deterministic signal processing *only*, with an explicit rule that no VLM ever narrates SAR — because no validated SAR-pretrained checkpoint existed for EarthDial. CROMA changes the premise: it is a **radar-native encoder**, pretrained contrastively + with masked-autoencoding on 1M paired Sentinel-1/Sentinel-2 tiles, and it is real, open, and lightweight (base ≈350MB, large ≈1.2GB; feasible on a single consumer GPU for both inference and fine-tuning). This is exactly the kind of option Section 1's checklist exists to catch when it appears — a previously-correct restriction becomes upgradable once a genuinely-fitting, verified option exists.
- **What CROMA is NOT — state this explicitly everywhere it's mentioned:** a pure visual encoder. No language head. No captioning, VQA, or grounding capability by itself. It produces patch-token embeddings — nothing more. All downstream language capability is built by your own projector + decoder training, described in 3.4.
- **Hard input constraints (design around these, don't discover them at integration time):**
  - Input: 120×120 px tiles, Sentinel-1 must be exactly 2 channels (VV, VH).
  - RISAT compatibility gap: if RISAT imagery is single-polarization, you must duplicate the channel or zero-pad to 2 channels — **this is a stated assumption in every report/slide that touches RISAT inputs, not a silent implementation detail.**
  - Normalization: per-channel mean ± 2σ, rescaled to uint8 or float 0–1 (SatMAE/SeCo convention) — SAR must be fed in calibrated dB scale (gamma-naught), never raw digital numbers.
  - Resolution mismatch with EarthDial (120×120 vs. 448+): each branch runs at its own native tile size; do not force a shared resolution. CROMA outputs per-patch tokens that can be projected regardless of the source tile size mismatch.
- **Fallback (unchanged in spirit from the old rule, updated in mechanism):** the design invariant was never "no SAR encoder" — it was **"no hallucinated SAR narrative from a model that was never trained on radar physics."** CROMA satisfies that invariant by being radar-native, so the *narrative* fallback shifts from "always deterministic" to "deterministic math as a live cross-check, escalating to deterministic-only if CROMA/projector/decoder is unavailable or its output disagrees with the deterministic signal." See 3.5 for exactly how this fusion and disagreement-check works.
- **Guard test:** the regression test's assertion changes from *"SAR image never reaches any VLM call"* to *"SAR image never reaches an encoder that was not pretrained on SAR"* — i.e., the test must assert the SAR tile's forward pass goes through CROMA-S1 specifically, and fail loudly if a future edit routes it through EarthDial's optical InternViT branch instead. This is a stricter, more specific test than the one it replaces, not a weaker one.
- **Replace this when:** a stronger radar-native encoder is released and passes Section 1 (real weights, clear capability boundary, known integration cost, known fallback, testable guard) — e.g., a future CROMA version, or a different contrastive SAR-optical pretraining effort. Do not replace on the basis of a paper alone; require downloadable weights and a reproduced sanity check first.

### 3.4 Bridging the SAR encoder into the language interface

This is new work this revision introduces — it exists nowhere in the v2 plan because the old design never needed a SAR→language bridge.

- **Mechanism:** CROMA-S1 patch tokens (768-dim, ViT-B) → a small trainable projector (2-layer MLP) → a shared embedding space (e.g., 1024-dim) that also receives EarthDial's InternViT optical tokens (1024-dim) via their own projector. Fused tokens (concatenation + cross-attention, or CROMA's own pretrained `joint_encodings` when both modalities are present) feed into EarthDial's existing Qwen decoder.
- **What you are and aren't retraining:** you train the projector(s) and, if needed, LoRA-adapt the decoder's cross-attention to attend over the new SAR-derived tokens. You are **not** retraining EarthDial's optical path, and you are **not** retraining CROMA's encoder from scratch — frozen-or-LoRA CROMA + trainable projector is the cheap, correct starting point, exactly as with the optical LoRA work already validated.
- **Per-task recipes (each is its own fine-tuning run — do not conflate them):**
  - **Captioning:** CROMA-S1 → global-average-pool → projector → decoder, teacher-forced on `BigEarthNet.txt` captions + VRSBench + the India caption set. Start with CROMA frozen, train only projector+decoder, as the cheap sanity check before spending GPU-hours on anything larger.
  - **VQA:** same dual-encoder + fused-token setup → decoder with LoRA, trained on `BigEarthNet.txt` VQA pairs + CDVQA (using the SAR branch specifically for SAR-input questions).
  - **Grounding — set expectations honestly, in advance, not after a bad demo:** an independent benchmark (SAREval) found under 3% Acc@0.5 for *all* current VLMs attempting SAR visual grounding. CROMA does not change this by itself. If grounding on SAR is attempted at all, use patch tokens (not pooled) → box regression head, and plan an **optical-guided fallback** (ground on the co-registered optical image, project the box onto the SAR tile) as the honest, disclosed alternative — not a claim of native SAR grounding.
  - **Change detection (SAR branch):** CROMA-S1 as a Siamese encoder — shared weights across T1/T2 tiles, then a difference/correlation layer → a small decoder (e.g., U-Net-style) for the change mask. This is additive to, not a replacement for, the existing optical NDVI/NDWI-delta change path.

### 3.5 Optical–SAR fusion (M6) — the mandatory cross-modal requirement, now with two real signal sources instead of one

The official scope requires genuine joint reasoning over co-registered optical+SAR pairs (a representative query: *"use the optical and SAR images together to identify built-up and water-covered regions"*). The fusion design must keep three things separately labeled in every response — this was already a hard rule under the old deterministic-only design, and it becomes more important, not less, now that a learned SAR signal exists alongside the deterministic one:

1. **Factual metadata** (capture dates, sensor IDs, resolution) — never inferred, always read from source metadata.
2. **Deterministic sensor-derived statements** — VV/VH log-ratio, threshold-based water/built-up indicators, NDVI/NDWI delta. These remain fully explainable and reproducible from raw pixel math, independent of any learned model.
3. **AI-model-derived statements** — EarthDial's optical captioning/VQA output, *and now* CROMA-S1-routed SAR understanding through the shared decoder.

**The new cross-check role of deterministic SAR math:** previously, deterministic math was the *only* SAR signal, so there was nothing to cross-check it against. Now, deterministic thresholds/ratios serve as an independent verification signal against CROMA-routed model output. When the two disagree — e.g., the model's SAR-derived statement claims "no water" but the VV/VH ratio crosses the water threshold — **the confidence band drops and the disagreement is surfaced explicitly**, never silently resolved in favor of either signal. This is a stronger, more defensible design than either "deterministic only" (loses expressiveness) or "model output trusted blindly" (reintroduces exactly the hallucination risk the original SAR restriction existed to prevent).

- **Guard test:** construct a synthetic or real case where deterministic SAR math and CROMA-routed model output disagree; assert the response surfaces both values, flags the disagreement, and lowers confidence — never assert one silently overwrites the other.
## 4. Agentic controller (M5) — dispatch matrix, updated for the CROMA branch

This extends the existing dispatch-matrix table (scale-up plan Section 6) with the rows that change or are newly needed because SAR now has a model-derived path, not just a deterministic one. Rows unaffected by the CROMA change are carried over unchanged; only the SAR-touching rows are rewritten.

| Condition | Required behavior |
|---|---|
| Optical-only query, single image | EarthDial VQA/captioning path (unchanged) |
| SAR-only query, single image | **Changed:** CROMA-S1 → projector → shared decoder produces the model-derived statement; deterministic VV/VH math runs in parallel as a cross-check; both are shown, separately labeled; disagreement lowers confidence (was: deterministic-only, no model statement at all) |
| Optical+SAR pair, cross-modal query | EarthDial-optical + CROMA-S1-SAR, fused via the controller into one response with three separately-labeled fields (factual / deterministic / model-derived) — **no change to the three-field rule, but the "model-derived" field can now legitimately include SAR content, which it could not before** |
| Bi-temporal optical pair, change query | NDVI/NDWI delta (unchanged) |
| Bi-temporal SAR pair, change query | **New capability:** CROMA-S1 Siamese branch (3.4) can now contribute a learned change signal alongside the existing log-ratio deterministic method; until the Siamese branch is trained and validated, this row's behavior stays at the deterministic-only fallback — do not enable the learned path in the dispatch table until its own guard test (Section 3.3) passes |
| Grounding requested on a SAR image | **New, explicit:** route to the optical-guided fallback (ground on the paired optical image, project box to SAR) if a co-registered optical image exists; otherwise state plainly that SAR-native grounding is not attempted, citing the SAREval finding — never silently return a low-confidence box as if it were reliable |
| Direct image upload (any modality/pair), bypassing retrieval | Compatibility check → routes straight to the relevant specialist path above — unchanged structurally, but the specialist path it lands on for SAR now includes the CROMA route |
| CROMA/projector/decoder fails to load | Falls back to deterministic-SAR-only mode (the entire previous design) — this must be a true drop-in fallback, meaning the deterministic math path must never be allowed to depend on the model path being available |
| Any response involving SAR, at any stage | Execution trace must name which SAR path fired: `"CROMA-S1 + deterministic cross-check"` or `"deterministic-only fallback"` — a judge or auditor must be able to tell which mode produced any given answer without inspecting code |

**Regenerate this table's test suite whenever a row changes.** One test per row is the standing rule from the scale-up plan (Section 6) — it applies identically here; the CROMA change adds rows, it doesn't relax the one-test-per-row requirement.

---

## 5. Codebase upgrade plan — what to install, when, and how to verify it

This section is deliberately sequenced so nothing is installed before its prerequisite is confirmed necessary. Each step names the **install command, the verification step, and what "done" means** — not just "add this library."

### 5.1 New dependencies this revision introduces

| Package | Purpose | Install | Verify |
|---|---|---|---|
| `einops` | Required by CROMA's official code (`use_croma.py`) for tensor reshaping | `pip install einops` | `python -c "import einops; print(einops.__version__)"` |
| CROMA weights (`CROMA_base.pt` or `CROMA_large.pt`) | The SAR/optical/joint encoders themselves | Download from `huggingface.co/antofuller/CROMA` | Load with `torch.load`, confirm the state dict keys match `use_croma.py`'s expected model class before wiring into the pipeline |
| CROMA source (`use_croma.py`) | Official model definition — do not reimplement from the paper alone | Pull from `github.com/antofuller/CROMA` | Run the repo's own example/inference snippet on a dummy tensor first, before feeding it real Sentinel-1 tiles |

No other new top-level dependencies are required — `torch` is already a dependency of the existing EarthDial pipeline (subject to the open Tier-0 blocker below about whether `torch` is actually installed and reachable in the live environment).

### 5.2 Existing dependency/blocker status this revision does NOT change

These are carried over verbatim from the scale-up plan's audit findings because they are prerequisites for *everything*, including the CROMA work — a SAR encoder that can't load is exactly as broken as an SAR encoder that was never written:

1. **Confirm `torch` is actually installed and loading on the real GPU boxes**, not just missing in the audit's sandbox. The CROMA branch cannot be tested, let alone shipped, if this is still unresolved. This is a 10-minute check, not a assumption to carry forward silently.
2. **Resolve the Ollama vs. `transformers.pipeline` divergence for the parser** before doing any GPU/timeout work — the CROMA integration will compete for the same GPU memory budget, so this decision should be locked before adding a second model (CROMA) onto boxes that may already be memory-constrained by the parser + EarthDial.
3. **M0 catalog population** and **M2 wiring into M5** remain blockers for the *retrieval* entry path (Path A) — they do not block the CROMA/SAR work, which lives entirely in the *direct-upload* entry path (Path B). These are parallel-safe, not sequential.

### 5.3 Sequencing the CROMA integration itself (safe order — each step has a checkpoint before the next starts)

1. **Environment check.** Confirm GPU memory headroom for a second encoder (CROMA-base ≈350MB weights, larger with activations) alongside whatever EarthDial and the parser already occupy. If memory-constrained, CROMA-base (ViT-B), not CROMA-large, is the default — don't reach for the larger checkpoint until base is validated and headroom is confirmed.
2. **Standalone CROMA sanity check.** Load `CROMA_base.pt`, run it on a synthetic 2-channel 120×120 tensor, confirm output shape matches documented patch-token dimensions. Do this **before** touching any real Sentinel-1/RISAT data — isolate "does the model load and run" from "does it work on our actual imagery."
3. **Real-data forward pass, no training yet.** Feed a real calibrated Sentinel-1 tile (dB scale, 2-channel VV/VH) through CROMA, confirm no shape/dtype errors, and manually inspect the output embeddings look non-degenerate (not all-zero, not NaN). Do the same for one RISAT sample once Bhoonidhi access is approved (Section 5.5), specifically to catch the single-polarization channel-count problem before it becomes a live-demo surprise.
4. **Projector training (cheap sanity run).** Freeze CROMA, train only the CROMA-side projector + a frozen-or-lightly-adapted decoder head on a small captioning slice, and confirm loss decreases and outputs are not degenerate before committing to a full training run.
5. **Full per-task fine-tuning runs** (captioning, then VQA, then — if pursued — grounding/change-Siamese), each gated by its own go/no-go checkpoint exactly like the existing EarthDial-RGB pattern: if a fine-tuned run isn't measurably better than the frozen-projector baseline by a fixed date, ship the frozen-projector version or fall back further, rather than slipping the timeline.
6. **Wire into the dispatch table** (Section 4) only after step 5's checkpoint passes for at least the captioning task — VQA and grounding can lag behind and ship as "in progress" honestly, per the existing feasibility-table pattern.
7. **Add the regression test** from Section 3.3 (SAR tile must route through CROMA-S1, never through EarthDial's optical encoder) **before** merging the dispatch-table wiring, not after — this is the test that would have caught the original `pipeline_fusion()` violation, so it must exist prior to the new code path going live, not as an afterthought.

### 5.4 What this upgrade does NOT require (protect against scope creep)

- Does **not** require retraining EarthDial's optical path.
- Does **not** require a multi-GPU setup — CROMA-base/large are both single-GPU-feasible for fine-tuning, per the source material's own hardware note.
- Does **not** require abandoning the deterministic SAR math — it remains the cross-check and the fallback, permanently, not a stepping-stone to delete later.
- Does **not** retroactively invalidate the Phi-4-mini parser work, the retrieval engine (M2), or the change-detection module (M4-optical) — none of those interact with the SAR encoder choice.

### 5.5 Data access dependency (unchanged from the scale-up plan, restated because it now also gates CROMA validation)

Bhoonidhi (`bhoonidhi.nrsc.gov.in`) registration remains the only real path to Cartosat-2S/RISAT sample imagery, and approval lead time is unstated. This now gates **two** things, not one: (a) confirming the general ingestion pipeline doesn't assume Sentinel-only conventions, and (b) confirming CROMA's 2-channel input assumption against real RISAT polarization data specifically. Register early regardless of CROMA status — this dependency existed before this revision and doesn't get any faster by being reprioritized.
## 6. Claims discipline — what you can and cannot say, and why (this is the anti-hallucination section)

The instruction to "make sure no hallucination happens and the output is generalized" applies at two levels: the **model's** outputs to end users, and the **team's** claims to judges/teammates about the system. Both are covered by the same discipline.

### 6.1 Model-output honesty (enforced by architecture, not just policy)
- No VLM — EarthDial or the CROMA-routed decoder — ever produces a SAR-derived or optical-derived statement without the corresponding deterministic cross-check running alongside it. This is enforced by the dispatch table (Section 4), not left to prompt instructions.
- Any disagreement between model output and deterministic math is surfaced, never resolved silently. Confidence bands exist specifically to carry this signal to the end user.
- If a component fails to load (parser tier, EarthDial, CROMA/projector), the system degrades to the next fallback tier and the execution trace states which tier actually ran — a user or judge must never be shown a "confident" answer that was secretly produced by the weakest fallback without that being visible.

### 6.2 Team-claim honesty (what NOT to tell judges, teammates, or put on a slide)
Carrying forward the scale-up plan's non-goals, extended for the CROMA change:

- Don't claim CROMA gives captioning/VQA/grounding "out of the box" — it is an encoder only; every language capability is something your team trained.
- Don't claim SAR grounding works — cite the SAREval <3% Acc@0.5 finding and the optical-guided fallback explicitly, proactively, before a judge asks.
- Don't claim "zero hallucination" on SAR anymore — that was accurate under the deterministic-only design and is **no longer accurate** once a learned encoder is in the loop. The correct, still-strong claim is "cross-checked by deterministic signal math, with disagreement surfaced rather than hidden."
- Don't claim general India-wide (or general-anywhere) understanding from any fine-tuned checkpoint — state the exact AOI count and zone diversity the fine-tuning actually covered.
- Don't claim `outlines`-based structural guarantees unless it is actually integrated (Section 3.1) — describe the real mechanism if it isn't.
- Don't present retrieval metrics (Recall@K, MRR, nDCG) as part of the official judged rubric — they validate your own catalog engine but are not named anywhere in the ISRO/SAC judging criteria.
- Don't report any number produced by a mock evaluation harness (e.g., a `MockEmbeddingModel`-based Recall@5/BERTScore) as if it were real — pull these immediately from any judge-facing material and replace with "not yet measured" until a real sentence-transformer is wired in.

### 6.3 "Generalized, usable everywhere" — what this can and cannot honestly mean here
The request for a "generalized" output deserves a precise answer, not a vague one: **generalization is a property you test and report, not a property you assert.** Concretely:
- The **architecture** (parser → controller → specialist branches → fusion → trace) is domain-general by design — it does not hardcode Assam, India, or any specific AOI list into its logic. AOI lists, gazetteers, and fine-tuning corpora are *data*, swappable without touching the controller.
- The **fine-tuned models** are not, and should not be claimed to be, generally capable beyond what they were trained/validated on. A LoRA adapter trained on 20–30 AOIs is demo-scope-adapted, not nationally general — this is the same honest framing the scale-up plan already established for EarthDial-RGB, and it applies identically to the new CROMA-routed SAR path.
- The path to *actual* broader generalization is the live Tier-3 fallback (Section 5 of the scale-up plan): for any input outside the curated/fine-tuned scope, the system explicitly labels its answer as "live, uncached, not pre-validated" rather than silently pretending equal confidence everywhere. This labeled-degradation pattern **is** the generalization story — not a claim that the fine-tuned models themselves generalize perfectly, but that the system as a whole has an honest, non-crashing answer for inputs it wasn't specifically tuned on.

---

## 7. Mandatory-scope checklist, updated for this revision

Carried forward from the scale-up plan's Section 10.2, with the SAR-related rows updated to reflect that a model-derived SAR path now exists alongside the deterministic one. Status tiers are illustrative — replace with your team's actual state before presenting.

| Requirement | Status | What changed this revision |
|---|---|---|
| Remote-sensing adaptation (`BigEarthNet.txt`) | Unchanged — satisfy via `BigEarthNet.txt` directly | No change — still the fastest, lowest-risk path to this checkbox |
| Single-image VQA | Unchanged | No change from this revision |
| Captioning (chosen 2nd single-image task) | Unchanged | No change from this revision |
| Cross-modal optical+SAR analysis | **Upgraded** | Previously deterministic-only; now deterministic + CROMA-routed model signal, cross-checked |
| Change understanding (bi-temporal) | **Partially upgraded** | Optical NDVI/NDWI delta unchanged; SAR Siamese-CROMA change branch is new and gated behind its own guard test — do not mark this "done" for the SAR side until 3.4's change-detection recipe is validated |
| Agentic orchestration + execution trace | **Extended** | Trace must now name which SAR path fired (CROMA+cross-check vs. deterministic-only fallback) |
| Interactive GUI | Unchanged | No change from this revision |
| GeoTIFF/TIFF input handling | **At added risk, must re-verify** | CROMA's fixed 2-channel/120×120 input constraint is a new failure point specifically for real RISAT imagery — re-test ingestion against this constraint, don't assume the existing pipeline handles it |

---

## 8. Execution order for this specific change (fits inside the existing Tier system from the scale-up plan)

This does not replace the scale-up plan's Section 14 master execution order — it inserts the CROMA work into it at the tier where it actually belongs, based on cost and risk, not on how exciting it is.

- **Fits in Tier 0.5/1 (parallel, low-cost, start now):** environment check (5.3 step 1), standalone CROMA sanity check (5.3 step 2) — cheap, no data dependency, de-risks the rest.
- **Fits in Tier 1 (after Bhoonidhi access, real-data dependent):** real-data forward pass including the RISAT single-polarization check (5.3 step 3) — this is exactly the kind of "confirm before the official evaluation does it for you" check the scale-up plan already prioritizes for the general ingestion pipeline; the CROMA-specific version of that check rides along with it.
- **Fits in Tier 3 (mandatory-scope build-out):** projector training and per-task fine-tuning (5.3 steps 4–5), dispatch-table wiring and its regression test (5.3 steps 6–7) — this is new mandatory-adjacent capability, sequenced alongside the existing direct-upload-path and execution-trace work, not ahead of it.
- **Never in Tier 5 (do not defer past national screening) unless the whole EarthDial-LoRA track is also deferred:** the SAR path is additive to, and shares infrastructure with, the optical fine-tuning track — deferring one without the other creates an inconsistent system where optical is adapted and SAR isn't, which is a worse story than deferring both together.

---

## 9. Sources for the claims in this document

- CROMA (Fuller et al., NeurIPS 2023) — official repo `github.com/antofuller/CROMA`, weights at `huggingface.co/antofuller/CROMA`.
- EarthDial (CVPR 2025) — `hiyamdebary/EarthDial` GitHub, confirmed public checkpoint list.
- "Landsat30-AU" (arXiv 2508.03127) — independent zero-shot benchmark of EarthDial.
- SAREval — cited finding on SAR visual grounding performance across current VLMs (<3% Acc@0.5); verify the exact citation/venue before quoting the number to judges if it will be printed on a slide.
- `BigEarthNet.txt` (arXiv:2603.29630) — 464,044 co-registered Sentinel-1/2 pairs, 9.6M annotations.
- VRSBench (arXiv:2406.12384, NeurIPS 2024 D&B track).
- RSVQA (Lobry et al., IEEE TGRS 2020).
- Bhoonidhi (`bhoonidhi.nrsc.gov.in`) — ISRO/NRSC data access path for Cartosat-2S/RISAT samples.
- Official SIH 2026 Problem Statement 167 (ISRO/SAC) — source of truth for all mandatory-scope claims.
- `production_scaleup_plan_v2.md` (this team, v2, audit-informed) — source for all pre-CROMA architectural decisions and their stated reasoning, carried forward except where explicitly superseded above.

---

## 10. One-paragraph summary, if you need to explain this whole revision in one breath

*The system's SAR-handling principle never changed: don't let a model narrate radar imagery it wasn't trained to understand. What changed is that a model trained specifically to understand radar imagery — CROMA-S1 — was found, verified as real and lightweight, and integrated as a second encoder feeding the same shared language decoder EarthDial already uses for optical. Deterministic SAR math didn't get removed; it became the permanent cross-check against the new model signal, with disagreement surfaced rather than hidden. Every other module — the parser, retrieval, change detection, the controller, the GUI, the execution trace — is unaffected and unchanged.*
