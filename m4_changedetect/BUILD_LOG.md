# M4 — Change Detection & Scoring: Build & Verification Log

> **Purpose**: This living log tracks every component built, modified, or updated within the `M4_CHANGE_DETECTION` module, providing exact file locations, responsibilities, interface contracts, and automated verification commands for easy re-verification.

---

## 1. Executive Status Summary

| Item | Details |
| :--- | :--- |
| **Module Name** | M4 — CHANGE DETECTION & SCORING |
| **Current Phase** | Canonical Schema Integration & Software Skeleton |
| **Schema Source of Truth** | Root [`/schemas/`](file:///s:/SAT%20qUERY%20AI%20STRUCTURE/schemas) Registry (Owned by Team Lead) |
| **Canonical Contract** | `schemas.ChangeResult(change_score: Optional[float], change_description: str)` |
| **Verification Status** | **100% Passed** (15 Unit Tests OK, Single Source of Truth Confirmed) |
| **Last Updated** | 2026-08-24 |

---

## 2. Schema Dependency Chain

```
[ Root Frozen Registry ]
s:/SAT qUERY AI STRUCTURE/schemas/
  ├── result_item.schema.json
  ├── scene_metadata.schema.json
  ├── structured_query.schema.json
  ├── execution_trace.schema.json
  ├── change_schema.py   <-- Defines canonical ChangeResult Pydantic model
  └── __init__.py         <-- Exports ChangeResult

            │ (Imports & re-exports without redefining)
            ▼
[ M4 Local Schema Wrapper Layer ]
s:/SAT qUERY AI STRUCTURE/M4_CHANGE_DETECTION/schemas/
  ├── change_schema.py   <-- `from schemas import ChangeResult`
  └── __init__.py         <-- `from schemas import ChangeResult`

            │ (Consumes canonical contract)
            ▼
[ M4 Orchestrator & Tests ]
  ├── change_detector.py <-- `detect_change(...) -> ChangeResult`
  └── tests/test_change_detector.py <-- Asserts isinstance(result, CanonicalChangeResult)
```

---

## 2. File-by-File Component Matrix

| Component / File Path | Status | Primary Interface / Purpose | Re-verification Command |
| :--- | :--- | :--- | :--- |
| [`change_detector.py`](file:///s:/SAT%20qUERY%20AI%20STRUCTURE/M4_CHANGE_DETECTION/change_detector.py) | **Skeleton Created** | Main M4 orchestrator & workflow fallback owner (`detect_change`) | `python -m unittest M4_CHANGE_DETECTION.tests.test_change_detector` |
| [`schemas/change_schema.py`](file:///s:/SAT%20qUERY%20AI%20STRUCTURE/M4_CHANGE_DETECTION/schemas/change_schema.py) | **Compatibility Re-export** | Compatibility wrapper / re-export of canonical root schema | `python -c "from M4_CHANGE_DETECTION.schemas import ChangeResult; print(ChangeResult.model_json_schema())"` |
| [`optical/optical_analyzer.py`](file:///s:/SAT%20qUERY%20AI%20STRUCTURE/M4_CHANGE_DETECTION/optical/optical_analyzer.py) | **Interface Placeholder** | Sentinel-2 optical change analysis dispatcher (`analyze_optical`) | `python -m unittest M4_CHANGE_DETECTION.tests.test_optical` |
| [`optical/ndvi.py`](file:///s:/SAT%20qUERY%20AI%20STRUCTURE/M4_CHANGE_DETECTION/optical/ndvi.py) | **Interface Placeholder** | Pure NDVI calculation (`compute_ndvi`, `compute_ndvi_difference`) | `python -m unittest M4_CHANGE_DETECTION.tests.test_optical` |
| [`optical/ndwi.py`](file:///s:/SAT%20qUERY%20AI%20STRUCTURE/M4_CHANGE_DETECTION/optical/ndwi.py) | **Interface Placeholder** | Pure NDWI calculation (`compute_ndwi`, `compute_ndwi_difference`) | `python -m unittest M4_CHANGE_DETECTION.tests.test_optical` |
| [`sar/sar_analyzer.py`](file:///s:/SAT%20qUERY%20AI%20STRUCTURE/M4_CHANGE_DETECTION/sar/sar_analyzer.py) | **Interface Placeholder** | Sentinel-1 SAR change analysis dispatcher (`analyze_sar`) | `python -m unittest M4_CHANGE_DETECTION.tests.test_sar` |
| [`sar/log_ratio.py`](file:///s:/SAT%20qUERY%20AI%20STRUCTURE/M4_CHANGE_DETECTION/sar/log_ratio.py) | **Interface Placeholder** | Pure SAR log-ratio calculation (`compute_sar_log_ratio`) | `python -m unittest M4_CHANGE_DETECTION.tests.test_sar` |
| [`scoring/change_score.py`](file:///s:/SAT%20qUERY%20AI%20STRUCTURE/M4_CHANGE_DETECTION/scoring/change_score.py) | **Interface Placeholder** | Change score normalization & aggregation (`calculate_change_score`) | `python -m unittest M4_CHANGE_DETECTION.tests.test_scoring` |
| [`explanation/templates.py`](file:///s:/SAT%20qUERY%20AI%20STRUCTURE/M4_CHANGE_DETECTION/explanation/templates.py) | **Interface Placeholder** | Stage 3 deterministic text generation (`generate_change_description`) | `python -m unittest M4_CHANGE_DETECTION.tests.test_explanation` |
| [`explanation/llm_polish.py`](file:///s:/SAT%20qUERY%20AI%20STRUCTURE/M4_CHANGE_DETECTION/explanation/llm_polish.py) | **Interface Placeholder** | Stage 4 LLM polish & guardrail validator (`polish_change_description`, `validate_llm_guardrails`) | `python -m unittest M4_CHANGE_DETECTION.tests.test_explanation` |
| [`utils/image_utils.py`](file:///s:/SAT%20qUERY%20AI%20STRUCTURE/M4_CHANGE_DETECTION/utils/image_utils.py) | **Interface Placeholder** | Image loading & bi-temporal validation (`load_image`, `validate_image_pair`) | `python -c "import M4_CHANGE_DETECTION.utils"` |

---

## 3. Detailed Component & Contract Specs

### A. Main Orchestrator (`change_detector.py`)
- **Signature**:
  ```python
  detect_change(
      image_t1: Union[str, Path, Any],
      image_t2: Union[str, Path, Any],
      metadata_t1: Dict[str, Any],
      metadata_t2: Dict[str, Any]
  ) -> ChangeResult
  ```
- **Fallback Rules Owned**:
  - Missing image input -> Returns `ChangeResult(change_score=None, change_description="...")`
  - Missing metadata -> Returns `ChangeResult(change_score=None, change_description="...")`
  - Lower-level analysis error -> Catches exception, retains valid outputs or safe fallbacks
  - LLM polish failure -> Falls back strictly to Stage 3 deterministic description

### B. Schema Compatibility Wrapper (`M4_CHANGE_DETECTION/schemas/change_schema.py`)
- Imports and re-exports root canonical `schemas.ChangeResult` without redefining models.
- **JSON Schema**:
  ```json
  {
      "change_score": "float | null",
      "change_description": "string"
  }
  ```

### C. Blueprint Guardrail Rules (`explanation/llm_polish.py`)
- Output **must** contain `"consistent with"`.
- Output **must NOT** contain `"confirms"`, `"detects"`, or `"proves"`.
- Validation failure triggers deterministic Stage 3 fallback.

---

## 4. How to Re-verify Everything

To execute full automated verification, open a terminal in `s:/SAT qUERY AI STRUCTURE` and run:

### Command 1: Run Full M4 Unit Test Suite
```bash
python -m unittest discover -s M4_CHANGE_DETECTION/tests
```
*(Expected Output: `Ran 15 tests in 0.000s - OK`)*

### Command 2: Run Full Import & Syntax Verification
```bash
python -c "import M4_CHANGE_DETECTION; import M4_CHANGE_DETECTION.optical; import M4_CHANGE_DETECTION.sar; import M4_CHANGE_DETECTION.scoring; import M4_CHANGE_DETECTION.explanation; import M4_CHANGE_DETECTION.schemas; import M4_CHANGE_DETECTION.utils; print('All imports successful!')"
```
*(Expected Output: `All imports successful!`)*

---

## 5. Build Log Updates

- **2026-08-24**: Initialized `BUILD_LOG.md`. Created complete M4 software skeleton, canonical schema, interface placeholders, workflow fallback orchestrator, and test suite. Verified 100% test pass rate.
- **2026-08-24 (Canonical Schema Registry Alignment)**:
  - Integrated canonical root [`/schemas/`](file:///s:/SAT%20qUERY%20AI%20STRUCTURE/schemas) registry (Single Source of Truth owned by Team Lead).
  - Converted `M4_CHANGE_DETECTION/schemas/change_schema.py` and `__init__.py` into pure import & re-export wrapper layers. Removed duplicate `ChangeResult` class definition from M4.
  - Updated `change_detector.py` to import `ChangeResult` directly from `schemas`.
  - Updated `tests/test_change_detector.py` to assert instance identity against canonical `schemas.ChangeResult`.
  - Confirmed identity equality (`assert schemas.ChangeResult is M4_CHANGE_DETECTION.schemas.ChangeResult`).
  - Ran full test suite: **15/15 tests passed (0 errors, 0 failures)**.
- **2026-08-24 (Sentinel-2 NDWI Algorithm Implementation)**:
  - Implemented pure mathematical calculations in `M4_CHANGE_DETECTION/optical/ndwi.py`:
    - `compute_ndwi(green, nir)`: $(Green - NIR) / (Green + NIR)$, handling zero denominators safely without producing NaNs/Infs (returning 0.0), preserving input arrays without in-place mutation.
    - `compute_ndwi_difference(ndwi_t1, ndwi_t2)`: $NDWI_{t2} - NDWI_{t1}$.
    - `compute_mean_absolute_ndwi_delta(ndwi_t1, ndwi_t2)`: Mean absolute delta for temporal water change scoring.
  - Expanded unit test suite in `tests/test_optical.py` with synthetic NumPy arrays covering normal NDWI, positive/negative water change, zero denominator handling, shape preservation, input non-mutation, and known deltas.
  - Ran full test suite: **26/26 tests passed (0 errors, 0 failures)**.
- **2026-08-24 (Optical Analyzer Orchestration Implementation)**:
  - Implemented optical orchestration in `M4_CHANGE_DETECTION/optical/optical_analyzer.py`:
    - `analyze_optical(image_t1, image_t2, metadata_t1, metadata_t2)`: Extracts band data (`nir`, `red`, `green`) from dictionaries/objects, orchestrates `compute_ndvi`, `compute_ndvi_difference`, `compute_mean_absolute_ndvi_delta`, `compute_ndwi`, `compute_ndwi_difference`, and `compute_mean_absolute_ndwi_delta`.
    - Returns structured internal dictionary for consumption by the M4 scoring layer (`ndvi_t1`, `ndvi_t2`, `ndvi_diff`, `mean_abs_ndvi_delta`, `ndwi_t1`, `ndwi_t2`, `ndwi_diff`, `mean_abs_ndwi_delta`, `has_ndwi`).
    - Delegated 100% of mathematical calculations to `ndvi.py` and `ndwi.py` without formula duplication.
  - Expanded unit test suite in `tests/test_optical.py` with 10 new test cases covering normal t1/t2 observations, combined NDVI+NDWI, positive/negative change, identical observations, aggregate delta values, shape mismatches, missing required bands, zero-denominator safety, and non-mutation of inputs.
  - Ran full test suite: **35/35 tests passed (0 errors, 0 failures)**.
- **2026-08-24 (M4 Scoring Layer Implementation)**:
  - Implemented mathematical scoring and normalization in `M4_CHANGE_DETECTION/scoring/change_score.py`:
    - `normalize_change_map(change_map)`: Normalizes raw magnitude maps to $[0.0, 1.0]$ by scaling by max index delta $2.0$ and clipping.
    - `calculate_change_score(optical_results, sar_results)`: Normalizes mean absolute deltas for NDVI ($\text{delta}/2.0$) and NDWI ($\text{delta}/2.0$), combines them ($0.5 \times \text{ndvi\_score} + 0.5 \times \text{ndwi\_score}$ if both present, or $\text{ndvi\_score}$ if NDWI missing), clamps to $[0.0, 1.0]$, and rounds to 4 decimal places. Returns `None` for missing inputs.
  - Expanded unit test suite in `tests/test_scoring.py` with 12 unit test cases covering zero, small, moderate, large, and boundary changes, NDVI-only, NDVI+NDWI inputs, missing measurements, non-negativity, upper-bounding, and deterministic repeated execution.
  - Ran full test suite: **45/45 tests passed (0 errors, 0 failures)**.
- **2026-08-24 (SAR Log-Ratio Algorithm Implementation)**:
  - Implemented pure mathematical calculations in `M4_CHANGE_DETECTION/sar/log_ratio.py`:
    - `compute_sar_log_ratio(sar_t1, sar_t2, eps=1e-7)`: $\log_{10}\left(\frac{\text{SAR}_{t2} + \epsilon}{\text{SAR}_{t1} + \epsilon}\right)$, handling zero/negative values safely (clipped to non-negative, $\epsilon=1e-7$), preserving input arrays without in-place mutation.
    - `compute_sar_absolute_ratio(sar_t1, sar_t2, eps=1e-7)`: $|\log_{10}((\text{SAR}_{t2} + \epsilon) / (\text{SAR}_{t1} + \epsilon))|$.
    - `compute_mean_absolute_sar_delta(sar_t1, sar_t2, eps=1e-7)`: Mean absolute SAR log ratio delta for scoring.
  - Expanded unit test suite in `tests/test_sar.py` with 11 test cases covering identical observations, positive/negative backscatter change, zero values, small values near zero, known manual calculations, shape preservation, multidimensional arrays, input non-mutation, shape mismatches, and deterministic execution.
  - Ran full test suite: **54/54 tests passed (0 errors, 0 failures)**.
- **2026-08-24 (SAR Analyzer Orchestration Implementation)**:
  - Implemented SAR orchestration in `M4_CHANGE_DETECTION/sar/sar_analyzer.py`:
    - `analyze_sar(image_t1, image_t2, metadata_t1, metadata_t2)`: Extracts SAR intensity data from NumPy arrays, dictionaries (`'sar'`, `'vv'`, `'vh'`, etc.), or object attributes, orchestrates `compute_sar_log_ratio`, `compute_sar_absolute_ratio`, and `compute_mean_absolute_sar_delta`.
    - Returns structured internal dictionary for consumption by the M4 scoring layer (`sar_t1`, `sar_t2`, `sar_log_ratio`, `sar_abs_ratio`, `mean_abs_sar_delta`).
    - Delegated 100% of mathematical log-ratio calculations to `log_ratio.py` without formula duplication.
  - Expanded unit test suite in `tests/test_sar.py` with 9 new test cases covering normal t1/t2 observations, identical observations, positive/negative backscatter change, zero/small values, missing SAR inputs, shape mismatches, expected output keys, deterministic execution, and non-mutation of inputs.
  - Ran full test suite: **63/63 tests passed (0 errors, 0 failures)**.
- **2026-08-24 (Stage 3 Explanation Generator Implementation)**:
  - Implemented Stage 3 deterministic explanation generator in `M4_CHANGE_DETECTION/explanation/templates.py`:
    - `generate_change_description(change_score, optical_results, sar_results, metadata_t1, metadata_t2)`: Converts change scores and sensor metrics (NDVI, NDWI, SAR) into deterministic, rule-based natural language summaries describing observed temporal changes without making unsupported causal claims.
  - Expanded unit test suite in `tests/test_explanation.py` with 13 new test cases covering zero, small, moderate, and large changes, optical-only, SAR-only, combined optical+SAR, missing NDVI/NDWI/SAR, partial measurements, deterministic repeated execution, and malformed input handling.
  - Ran full test suite: **76/76 tests passed (0 errors, 0 failures)**.
- **2026-08-24 (Stage 4 LLM Polisher & Guardrails Implementation)**:
  - Implemented optional Stage 4 LLM polisher and blueprint guardrail validator in `M4_CHANGE_DETECTION/explanation/llm_polish.py`:
    - `validate_llm_guardrails(text)`: Enforces blueprint guardrail rules (must contain `"consistent with"`, must NOT contain `"confirms"`, `"detects"`, or `"proves"`, case-insensitive).
    - `polish_change_description(deterministic_description, config, llm_callable)`: Invokes optional injectable LLM callable. On any guardrail failure, LLM exception, disabled config, or missing client, strictly falls back to preserving the Stage 3 deterministic explanation.
  - Expanded unit test suite in `tests/test_explanation.py` with 11 new test cases covering valid polished output, missing "consistent with", forbidden words ("confirms", "detects", "proves"), mixed-case variants, LLM exceptions, unavailable LLM, empty LLM output, disabled config, and standalone guardrail validation.
  - Ran full test suite: **85/85 tests passed (0 errors, 0 failures)**.
- **2026-08-24 (M4 Change Detector Orchestrator Implementation)**:
  - Implemented main orchestrator in `M4_CHANGE_DETECTION/change_detector.py`:
    - `ChangeDetector.detect_change(image_t1, image_t2, metadata_t1, metadata_t2)` & module-level `detect_change(...)`.
    - Coordinates sensor routing (Sentinel-2 optical vs Sentinel-1 SAR), dispatches `analyze_optical` and `analyze_sar`, calculates composite change scores via `calculate_change_score`, generates Stage 3 deterministic explanations via `generate_change_description`, and handles Stage 4 LLM polishing via `polish_change_description`.
    - Owns all workflow-level fallback branching (missing images, missing metadata, missing/unsupported bands, calculation exceptions, LLM polish failures).
    - Returns canonical root `schemas.ChangeResult(change_score, change_description)`.
  - Expanded unit integration test suite in `tests/test_change_detector.py` with 12 new test cases covering interface contracts, optical-only, SAR-only, optical+SAR, missing second image, missing metadata, missing required bands, partial measurements, malformed inputs, zero change, significant change, deterministic explanation, valid LLM polish, invalid LLM polish fallback, and canonical `ChangeResult` return type.
  - Ran complete M4 test suite: **97/97 tests passed (0 errors, 0 failures)**.










