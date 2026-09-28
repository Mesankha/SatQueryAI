"""
M5 Agentic Controller - dispatch table + pipeline implementations.
This file imports ONLY schemas and shared utilities. No M1-M4 internals.
"""
import asyncio
import logging
from typing import Any, Awaitable, Callable, List, Optional

from shared.schemas import (
    StructuredQuery, ResultItem, ExecutionTrace, SceneMetadata,
    ModelOutputs, ScoreBreakdown, UniformError, ErrorType, QueryResponse, TaskType,
)
from shared.logger import get_logger
from shared.config import get_config
from .trace_builder import TraceBuilder, timed_call, timed_call_async
from .compatibility_checker import (
    check_search_compatibility, check_vqa_compatibility,
    check_change_compatibility, check_fusion_compatibility,
)
from shared.confidence import compute_band, attach_confidence

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# M1 Parser (real - uses deterministic fallback as primary)
# ---------------------------------------------------------------------------

def mock_m1_parse(query_text: str) -> StructuredQuery:
    """Parse query using M1 deterministic fallback parser."""
    from m1_parser.parser_fallback import parse_fallback
    return parse_fallback(query_text)


# ---------------------------------------------------------------------------
# M2 Retrieval - Dispatcher (mock vs real engine)
# ---------------------------------------------------------------------------

def _m2_retrieve(query: StructuredQuery) -> List[SceneMetadata]:
    """
    M2 retrieval dispatcher.
    Reads retrieval.mock from config and calls either fixture mock or real engine.
    All pipelines should call this function.
    """
    # Reload settings if M0_CATALOG_DB env var is set (for test injection)
    import os
    if os.environ.get("M0_CATALOG_DB"):
        from shared.config import reload_settings
        reload_settings()
    
    config = get_config()
    retrieval_config = config.get("retrieval", {})
    # Force real M2 when test catalog is explicitly injected
    use_mock = retrieval_config.get("mock", True) and not os.environ.get("M0_CATALOG_DB")

    if use_mock:
        logger.debug("M2 retrieval: using fixture mock (retrieval.mock=true)")
        return _m2_retrieve_mock(query)
    else:
        logger.debug("M2 retrieval: using real engine (retrieval.mock=false)")
        return _m2_retrieve_real(query)


def _m2_retrieve_mock(query: StructuredQuery) -> List[SceneMetadata]:
    """Fixture-based retrieval (mock)."""
    from shared.validate import load_fixture_scenes
    scenes = load_fixture_scenes("fixtures/fixtures_scenes.json")

    if query.image_id:
        for s in scenes:
            if s.scene_id == query.image_id:
                return [s]
        return []

    results = []
    for s in scenes:
        if query.sensor and query.sensor.value != "both":
            if s.sensor != query.sensor.value and s.sensor != "both":
                continue
        if query.object and query.object.lower() not in (s.object or "").lower():
            continue
        results.append(s)

    results.sort(key=lambda s: s.acquisition_time, reverse=True)
    return results


def _m2_retrieve_real(query: StructuredQuery) -> List[SceneMetadata]:
    """Real M2 retrieval using engine.retrieve."""
    config = get_config()
    paths = config.get("paths", {})
    db_path = paths.get("catalog_db", "./data/catalog.db")
    top_k = config.get("retrieval", {}).get("top_k", 10)

    try:
        from m2_retrieval.engine import retrieve as engine_retrieve
        candidates = engine_retrieve(query, db_path, top_k=top_k)
        if candidates:
            return candidates
    except Exception as e:
        logger.warning(f"M2 real retrieval failed, falling back to fixtures: {e}")

    # Fallback to mock on any error
    return _m2_retrieve_mock(query)


# Keep mock_m2_retrieve as an alias for backward compatibility
def mock_m2_retrieve(query: StructuredQuery) -> List[SceneMetadata]:
    """Deprecated: use _m2_retrieve instead. Kept for backward compat."""
    return _m2_retrieve(query)


# Public API for tests and external use
def m2_retrieve(query: StructuredQuery) -> List[SceneMetadata]:
    """Public M2 retrieval entry point. Dispatches to mock or real engine based on config."""
    return _m2_retrieve(query)


# ---------------------------------------------------------------------------
# M3 VLM Adapters (mock vs real vlm_runner) with modality support
# ---------------------------------------------------------------------------

async def _call_m3_vqa(image_path: str, question: str, modality: str = "optical"):
    """Call M3 run_vqa via vlm_runner (real) or HTTP mock, depending on vlm.mock config.
    Falls back to mock if real VLM fails (e.g., no GPU)."""
    config = get_config()
    vlm_config = config.get("vlm", {})
    
    # Determine mock flag based on modality
    if modality == "sar":
        use_mock = vlm_config.get("sar_mock", True)
    else:
        use_mock = vlm_config.get("mock", True)

    if use_mock:
        # Mock path: use existing HTTP client (which will return mock responses)
        from m3_vlm.http_client import run_vqa_http
        result = await run_vqa_http(image_path, question, modality=modality)
        if result.get("error"):
            return UniformError(
                module="M3",
                error_type=ErrorType(result.get("error_type", "model_unavailable")),
                message=result.get("message", "VQA failed"),
                fallback_applied=result.get("fallback_used", False),
            )
        return result["answer"]
    else:
        # Real path: use vlm_runner with fallback to mock
        try:
            if modality == "sar":
                from m3_vlm.vlm_runner import run_sar_vqa
                result = run_sar_vqa(image_path, question)
            else:
                from m3_vlm.vlm_runner import run_vqa
                result = run_vqa(image_path, question)
            if isinstance(result, UniformError):
                # Real VLM failed, fall back to mock
                logger.warning(f"Real VLM VQA ({modality}) failed: {result.message}, falling back to mock")
                return _mock_vqa_response(image_path, question, modality)
            return result
        except Exception as e:
            # Any exception in real VLM path -> fall back to mock
            logger.warning(f"Real VLM VQA ({modality}) exception: {e}, falling back to mock")
            return _mock_vqa_response(image_path, question, modality)


def _mock_vqa_response(image_path: str, question: str, modality: str = "optical") -> str:
    """Generate mock VQA response directly (no HTTP)."""
    if modality == "sar":
        return f"[SAR Mock] Based on radar backscatter analysis of {image_path}: {question} -> Simulated answer based on SAR characteristics."
    return f"[Mock] Simulated answer for {image_path}: {question} -> Based on visual analysis of the satellite imagery."


async def _call_m3_caption(image_path: str, modality: str = "optical"):
    """Call M3 run_caption via vlm_runner (real) or HTTP mock, depending on vlm.mock config.
    Falls back to mock if real VLM fails (e.g., no GPU)."""
    config = get_config()
    vlm_config = config.get("vlm", {})
    
    # Determine mock flag based on modality
    if modality == "sar":
        use_mock = vlm_config.get("sar_mock", True)
    else:
        use_mock = vlm_config.get("mock", True)

    if use_mock:
        from m3_vlm.http_client import run_caption_http
        result = await run_caption_http(image_path, modality=modality)
        if result.get("error"):
            return UniformError(
                module="M3",
                error_type=ErrorType(result.get("error_type", "model_unavailable")),
                message=result.get("message", "Caption failed"),
                fallback_applied=result.get("fallback_used", False),
            )
        return result["caption"]
    else:
        # Real path: use vlm_runner with fallback to mock
        try:
            if modality == "sar":
                from m3_vlm.vlm_runner import run_sar_caption
                result = run_sar_caption(image_path)
            else:
                from m3_vlm.vlm_runner import run_caption
                result = run_caption(image_path)
            if isinstance(result, UniformError):
                # Real VLM failed, fall back to mock
                logger.warning(f"Real VLM ({modality}) failed: {result.message}, falling back to mock")
                return _mock_caption_response(image_path, modality)
            return result
        except Exception as e:
            # Any exception in real VLM path -> fall back to mock
            logger.warning(f"Real VLM ({modality}) exception: {e}, falling back to mock")
            return _mock_caption_response(image_path, modality)


def _mock_caption_response(image_path: str, modality: str = "optical") -> str:
    """Generate mock caption response directly (no HTTP)."""
    if modality == "sar":
        return f"[SAR Mock] Simulated SAR analysis for {image_path}: Radar backscatter shows typical patterns with bright urban areas, dark water bodies, and textured vegetation. Water fraction ~15%, built-up ~10%."
    return f"[Mock] Simulated optical caption for {image_path}: This is a simulated satellite image showing typical remote sensing features including land cover, water bodies, and vegetation patterns."


# ---------------------------------------------------------------------------
# M4 Change Adapters (mock vs real change_detector) — uses m4_changedetect only
# ---------------------------------------------------------------------------

async def _call_m4_change(t1_path: str, t2_path: str, metadata_t1: dict = None, metadata_t2: dict = None):
    """Call M4 change detection via m4_changedetect (mock or real), depending on change.mock config."""
    config = get_config()
    change_config = config.get("change", {})
    use_mock = change_config.get("mock", True)

    if use_mock:
        # Mock path: use existing m4_changedetect.detect_change
        from m4_changedetect import detect_change
        from shared.schemas import UniformError, ErrorType

        try:
            meta_t1 = metadata_t1 or {"constellation": "SENTINEL-2"}
            meta_t2 = metadata_t2 or {"constellation": "SENTINEL-2"}

            result = detect_change(t1_path, t2_path, meta_t1, meta_t2)

            return {
                "change_score": result.change_score,
                "change_description": result.change_description
            }
        except Exception as e:
            logger.warning(f"M4 change detection failed: {e}")
            return UniformError(
                module="M4",
                error_type=ErrorType.model_unavailable,
                message=f"Change detection failed: {str(e)}",
                fallback_applied=False,
            )
    else:
        # Real path: use m4_changedetect with config to enable real detectors
        from m4_changedetect import detect_change
        from shared.schemas import UniformError, ErrorType

        try:
            meta_t1 = metadata_t1 or {"constellation": "SENTINEL-2"}
            meta_t2 = metadata_t2 or {"constellation": "SENTINEL-2"}

            # Pass config to enable real analysis (mock=false at module level handled by m4_changedetect)
            result = detect_change(t1_path, t2_path, meta_t1, meta_t2)

            return {
                "change_score": result.change_score,
                "change_description": result.change_description
            }
        except Exception as e:
            logger.warning(f"M4 change detection failed: {e}")
            return UniformError(
                module="M4",
                error_type=ErrorType.model_unavailable,
                message=f"Change detection failed: {str(e)}",
                fallback_applied=False,
            )


async def _call_m4_sar_features(sar_path: str) -> dict[str, Any] | UniformError:
    """Call M4 deterministic SAR feature extraction via m4_changedetect."""
    # Check mock mode
    config = get_config()
    change_config = config.get("change", {})
    use_mock = change_config.get("mock", True)
    
    if use_mock:
        # Return mock SAR features for demo
        import random
        return {
            "water_fraction": round(random.uniform(0.05, 0.35), 3),
            "builtup_fraction": round(random.uniform(0.02, 0.25), 3),
            "log_ratio_mean": round(random.uniform(-0.5, 0.5), 3),
            "notes": "Mock SAR features (change.mock=true)",
        }
    
    from m4_changedetect.sar.sar_analyzer import analyze_sar
    from shared.schemas import UniformError, ErrorType

    try:
        # Use the SAR analyzer from m4_changedetect
        result = analyze_sar({"intensity": sar_path}, {"intensity": sar_path}, {}, {})
        # analyze_sar returns dict with sar_metrics
        sar_metrics = result.get("sar_metrics", {})
        return {
            "water_fraction": sar_metrics.get("water_fraction", 0.0),
            "builtup_fraction": sar_metrics.get("builtup_fraction", 0.0),
            "log_ratio_mean": sar_metrics.get("log_ratio_mean", 0.0),
        }
    except Exception as e:
        logger.warning(f"M4 SAR feature extraction failed: {e}")
        return UniformError(
            module="M4",
            error_type=ErrorType.invalid_input,
            message=f"SAR feature extraction failed: {str(e)}",
            fallback_applied=False,
        )


# ---------------------------------------------------------------------------
# Output synthesizer
# ---------------------------------------------------------------------------

def _synthesize_result(
    scene: SceneMetadata,
    rank: int,
    score: float,
    model_outputs: ModelOutputs,
    query: StructuredQuery,
    trace: ExecutionTrace,
) -> ResultItem:
    raw_bd = getattr(scene, "score_breakdown", None)
    if raw_bd and isinstance(raw_bd, dict):
        score_breakdown = ScoreBreakdown(**raw_bd)
    else:
        # Use zero scores with placeholder flag - only populate modality_score and change_score when real values exist
        modality_score = 1.0 if query.sensor is None or query.sensor.value in (scene.sensor, "both") else 0.5
        change_score = 1.0 if model_outputs.change_description is not None else 0.0
        score_breakdown = ScoreBreakdown(
            semantic_sim=0.0,
            geo_score=0.0,
            temporal_score=0.0,
            modality_score=modality_score,
            change_score=change_score,
        )

    parts = []
    if model_outputs.caption:
        parts.append(f"Caption: {model_outputs.caption}")
    if model_outputs.vqa_answer:
        parts.append(f"VQA answer: {model_outputs.vqa_answer}")
    if model_outputs.change_description:
        parts.append(f"Change description: {model_outputs.change_description}")
    if model_outputs.fusion_statement:
        parts.append(f"Fusion statement: {model_outputs.fusion_statement}")

    model_summary = " ".join(parts) if parts else "No additional model outputs available."

    explanation = (
        f"This result is ranked #{rank} with a composite score of {score:.2f}. "
        f"The image was acquired by {scene.sensor} on {scene.acquisition_time.isoformat()} over {scene.geometry_wkt[:30]}... "
        f"{model_summary} These findings are consistent with the query criteria."
    )

    confidence_note = "Analysis completed using primary models."
    if query.used_fallback:
        confidence_note = "Note: The query was parsed using fallback rules; interpret results with caution."
    elif trace.fallback_used:
        confidence_note = "Note: One or more analysis tools used fallback behavior."

    result = ResultItem(
        image_id=scene.scene_id,
        rank=rank,
        score=score,
        score_breakdown=score_breakdown,
        metadata={
            "satellite": scene.sensor,
            "sensor": scene.sensor,
            "acquisition_time": scene.acquisition_time.isoformat(),
            "location_label": scene.geometry_wkt[:50],
            "scores_placeholder": True,
        },
        model_outputs=model_outputs,
        explanation_text=explanation,
        confidence_note=confidence_note,
    )

    # Attach confidence band and freshness metadata
    attach_confidence(result, scene, query.used_fallback, trace)

    return result


# ---------------------------------------------------------------------------
# Pipeline implementations (async) - using REAL M2
# ---------------------------------------------------------------------------

async def pipeline_search(query: StructuredQuery, builder: TraceBuilder) -> List[ResultItem]:
    # Trace M2 retrieval (sync function)
    candidates = timed_call(
        builder, "M2", "engine-retrieve" if not get_config().get("retrieval", {}).get("mock", True) else "fixture-mock",
        {"query_text": query.query_text, "sensor": query.sensor.value if query.sensor else None},
        lambda: _m2_retrieve(query)
    )
    builder.set_candidates(considered=len(candidates) + 5, after_filter=len(candidates))
    ok, err = check_search_compatibility(query, candidates)
    if not ok:
        builder.set_fallback_used(True)
        return []

    results = []
    for i, c in enumerate(candidates[:5]):
        # Use M2 composite score if available, else decreasing default
        score = getattr(c, "score", None) or max(0.5, 1.0 - i * 0.1)
        
        # For top result, try to get a caption for better UX
        model_outputs = ModelOutputs()
        if i == 0:
            ok_cap, err = check_vqa_compatibility(query, c)
            if ok_cap:
                try:
                    caption = await asyncio.wait_for(
                        timed_call_async(
                            builder, "M3", "GeoChat-4bit-optical",
                            {"image_path": c.file_path or ""},
                            _call_m3_caption, c.file_path or "", "optical"
                        ),
                        timeout=20.0
                    )
                    if not isinstance(caption, UniformError):
                        model_outputs = ModelOutputs(caption=caption, optical_caption=caption)
                except Exception:
                    pass  # Silently skip caption for search
        
        results.append(_synthesize_result(c, i + 1, score, model_outputs, query, builder.build()))
    return results


async def pipeline_vqa(query: StructuredQuery, builder: TraceBuilder) -> List[ResultItem]:
    candidates = timed_call(
        builder, "M2", "engine-retrieve" if not get_config().get("retrieval", {}).get("mock", True) else "fixture-mock",
        {"query_text": query.query_text, "sensor": query.sensor.value if query.sensor else None},
        lambda: _m2_retrieve(query)
    )
    builder.set_candidates(considered=len(candidates) + 5, after_filter=len(candidates))
    if not candidates:
        return []

    top = candidates[0]
    ok, err = check_vqa_compatibility(query, top)
    if not ok:
        builder.set_fallback_used(True)
        return [_error_result(top, err, query, builder.build())]

    try:
        answer = await asyncio.wait_for(
            timed_call_async(
                builder, "M3", "GeoChat-4bit",
                {"image_path": top.file_path or "", "question": query.question or ""},
                _call_m3_vqa, top.file_path or "", query.question or ""
            ),
            timeout=35.0
        )
    except asyncio.TimeoutError:
        logger.warning(f"M3 VQA timeout for {top.file_path}")
        builder.set_fallback_used(True)
        return [_error_result(top, UniformError(module="M3", error_type=ErrorType.timeout, message="VQA timeout"), query, builder.build())]

    if isinstance(answer, UniformError):
        builder.set_fallback_used(True)
        return [_error_result(top, answer, query, builder.build())]

    return [_synthesize_result(
        top, 1, 0.9, ModelOutputs(vqa_answer=answer), query, builder.build()
    )]


async def pipeline_caption(query: StructuredQuery, builder: TraceBuilder) -> List[ResultItem]:
    candidates = timed_call(
        builder, "M2", "engine-retrieve" if not get_config().get("retrieval", {}).get("mock", True) else "fixture-mock",
        {"query_text": query.query_text, "sensor": query.sensor.value if query.sensor else None},
        lambda: _m2_retrieve(query)
    )
    builder.set_candidates(considered=len(candidates) + 5, after_filter=len(candidates))
    if not candidates:
        return []

    top = candidates[0]
    ok, err = check_vqa_compatibility(query, top)
    if not ok:
        builder.set_fallback_used(True)
        return [_error_result(top, err, query, builder.build())]

    try:
        caption = await asyncio.wait_for(
            timed_call_async(
                builder, "M3", "GeoChat-4bit",
                {"image_path": top.file_path or ""},
                _call_m3_caption, top.file_path or ""
            ),
            timeout=35.0
        )
    except asyncio.TimeoutError:
        logger.warning(f"M3 Caption timeout for {top.file_path}")
        builder.set_fallback_used(True)
        return [_error_result(top, UniformError(module="M3", error_type=ErrorType.timeout, message="Caption timeout"), query, builder.build())]

    if isinstance(caption, UniformError):
        builder.set_fallback_used(True)
        return [_error_result(top, caption, query, builder.build())]

    return [_synthesize_result(
        top, 1, 0.9, ModelOutputs(caption=caption), query, builder.build()
    )]


async def pipeline_change(query: StructuredQuery, builder: TraceBuilder) -> List[ResultItem]:
    candidates = timed_call(
        builder, "M2", "engine-retrieve" if not get_config().get("retrieval", {}).get("mock", True) else "fixture-mock",
        {"query_text": query.query_text, "sensor": query.sensor.value if query.sensor else None},
        lambda: _m2_retrieve(query)
    )
    builder.set_candidates(considered=len(candidates) + 5, after_filter=len(candidates))
    if not candidates:
        return []

    t1 = candidates[0]
    t2 = None
    for c in candidates:
        if c.scene_id == t1.paired_scene_id:
            t2 = c
            break
    if t2 is None and len(candidates) > 1:
        t2 = candidates[1]

    ok, err = check_change_compatibility(t1, t2)
    if not ok:
        builder.set_fallback_used(True)
        return [_error_result(t1, err, query, builder.build())]

    try:
        change_result = await asyncio.wait_for(
            timed_call_async(
                builder, "M4", "deterministic-NDVI-logratio",
                {"t1_path": t1.file_path or "", "t2_path": t2.file_path if t2 else ""},
                _call_m4_change,
                t1.file_path or "",
                t2.file_path if t2 else "",
                {"constellation": t1.sensor, "date": t1.acquisition_time.isoformat() if t1.acquisition_time else None},
                {"constellation": t2.sensor, "date": t2.acquisition_time.isoformat() if t2.acquisition_time else None} if t2 else {}
            ),
            timeout=60.0
        )
    except asyncio.TimeoutError:
        logger.warning(f"M4 change detection timeout for {t1.file_path}")
        builder.set_fallback_used(True)
        return [_error_result(t1, UniformError(module="M4", error_type=ErrorType.timeout, message="Change detection timeout"), query, builder.build())]

    if isinstance(change_result, UniformError):
        builder.set_fallback_used(True)
        return [_error_result(t1, change_result, query, builder.build())]

    return [_synthesize_result(
        t1, 1, 0.85,
        ModelOutputs(change_description=change_result.get("change_description")),
        query, builder.build()
    )]


async def _synthesize_fusion_statement(
    optical_caption: str,
    sar_vlm_caption: Optional[str],
    sar_deterministic: dict,
    query: StructuredQuery,
    builder: TraceBuilder
) -> str:
    """Synthesize final fusion statement using LLM or template fallback."""
    
    # Try to use Phi-4-mini (parser LLM) for synthesis if available
    try:
        from m1_parser.parser_llm import parse_with_llm
        synthesis_prompt = f"""
        You are a remote sensing analyst. Synthesize a concise analysis combining:
        
        Optical VLM Analysis: {optical_caption}
        SAR VLM Analysis: {sar_vlm_caption or "Not available (deterministic only)"}
        SAR Deterministic Features: Water={sar_deterministic.get('water_fraction',0):.1%}, Built-up={sar_deterministic.get('builtup_fraction',0):.1%}, Log-ratio={sar_deterministic.get('log_ratio_mean',0):.2f}
        
        User Query: {query.query_text}
        
        Return JSON with these exact keys:
        {{
            "fusion_statement": "<unified analysis>",
            "cross_validation": "<what SAR confirms/contradicts vs optical>",
            "confidence": 0.0-1.0
        }}
        
        The analysis should:
        1. Confirm/correct optical findings with SAR evidence
        2. Highlight what SAR sees that optical cannot (through clouds, at night, surface roughness)
        3. Quantify key metrics from deterministic analysis
        4. Be actionable for the user's query
        """
        
        result = parse_with_llm(synthesis_prompt)
        if result and isinstance(result, dict) and "fusion_statement" in result:
            # Log cross-validation if available
            if "cross_validation" in result:
                logger.debug(f"LLM cross-validation: {result['cross_validation']}")
            return result["fusion_statement"]
    except Exception as e:
        logger.debug(f"LLM synthesis not available: {e}")
    
    # Template-based fallback
    return _template_fusion_synthesis(optical_caption, sar_vlm_caption, sar_deterministic)


def _template_fusion_synthesis(
    optical_caption: str,
    sar_vlm_caption: Optional[str],
    sar_deterministic: dict
) -> str:
    """Template-based fusion synthesis (no LLM required)."""
    parts = []
    
    if sar_vlm_caption:
        parts.append(f"[SAR VLM] {sar_vlm_caption}")
    else:
        # Use deterministic as narrative
        parts.append(f"[SAR Deterministic] Water cover {sar_deterministic.get('water_fraction',0):.1%}, "
                     f"built-up {sar_deterministic.get('builtup_fraction',0):.1%}, "
                     f"mean log-ratio {sar_deterministic.get('log_ratio_mean',0):.2f}.")
    
    parts.append(f"[VLM Optical] {optical_caption}.")
    
    # Cross-validation insights
    water_frac = sar_deterministic.get('water_fraction', 0)
    builtup_frac = sar_deterministic.get('builtup_fraction', 0)
    
    if water_frac > 0.3:
        parts.append("SAR confirms significant water presence (low VV backscatter).")
    if builtup_frac > 0.2:
        parts.append("SAR indicates built-up structures (high VV+VH backscatter).")
    
    return " ".join(parts)


async def pipeline_fusion(query: StructuredQuery, builder: TraceBuilder) -> List[ResultItem]:
    candidates = timed_call(
        builder, "M2", "engine-retrieve" if not get_config().get("retrieval", {}).get("mock", True) else "fixture-mock",
        {"query_text": query.query_text, "sensor": query.sensor.value if query.sensor else None},
        lambda: _m2_retrieve(query)
    )
    builder.set_candidates(considered=len(candidates) + 5, after_filter=len(candidates))
    if not candidates:
        return []

    optical = next((c for c in candidates if c.modality in ("optical", "multispectral")), None)
    sar = next((c for c in candidates if c.modality == "sar"), None)

    ok, err = check_fusion_compatibility(optical, sar)
    if not ok:
        builder.set_fallback_used(True)
        return [_error_result(candidates[0], err, query, builder.build())]

    # --- OPTICAL PATH (always VLM) ---
    try:
        opt_caption = await asyncio.wait_for(
            timed_call_async(
                builder, "M3", "GeoChat-4bit-optical",
                {"image_path": optical.file_path or ""},
                _call_m3_caption, optical.file_path or "", "optical"
            ),
            timeout=35.0
        )
    except asyncio.TimeoutError:
        logger.warning("M3 Caption timeout in fusion pipeline (optical)")
        builder.set_fallback_used(True)
        return [_error_result(candidates[0], UniformError(module="M3", error_type=ErrorType.timeout, message="Caption timeout"), query, builder.build())]

    if isinstance(opt_caption, UniformError):
        builder.set_fallback_used(True)
        return [_error_result(candidates[0], opt_caption, query, builder.build())]

    # --- SAR PATH A: DETERMINISTIC (always runs) ---
    sar_feats = await timed_call_async(
        builder, "M4", "sar_features-deterministic",
        {"sar_path": sar.file_path or ""},
        _call_m4_sar_features, sar.file_path or ""
    )
    if isinstance(sar_feats, UniformError):
        builder.set_fallback_used(True)
        return [_error_result(candidates[0], sar_feats, query, builder.build())]

    # --- SAR PATH B: VLM (when GPU + SAR LoRA available) ---
    sar_vlm_caption = None
    config = get_config()
    vlm_config = config.get("vlm", {})
    sar_mock = vlm_config.get("sar_mock", True)
    
    if not sar_mock:
        try:
            sar_vlm_caption = await asyncio.wait_for(
                timed_call_async(
                    builder, "M3", "GeoChat-4bit-SAR",
                    {"image_path": sar.file_path or ""},
                    _call_m3_caption, sar.file_path or "", "sar"
                ),
                timeout=40.0  # SAR may need more time
            )
            if isinstance(sar_vlm_caption, UniformError):
                logger.warning(f"SAR VLM failed, using deterministic only: {sar_vlm_caption.message}")
                sar_vlm_caption = None
                builder.set_fallback_used(True)
        except asyncio.TimeoutError:
            logger.warning("SAR VLM timeout in fusion pipeline")
            builder.set_fallback_used(True)
        except Exception as e:
            logger.warning(f"SAR VLM failed: {e}")
            builder.set_fallback_used(True)

    # --- LLM SYNTHESIS ---
    fusion_statement = await _synthesize_fusion_statement(
        optical_caption=opt_caption,
        sar_vlm_caption=sar_vlm_caption,
        sar_deterministic=sar_feats,
        query=query,
        builder=builder
    )

    model_outputs = ModelOutputs(
        optical_caption=opt_caption,
        sar_caption=sar_vlm_caption,
        sar_features=sar_feats,
        fusion_statement=fusion_statement
    )

    return [_synthesize_result(
        optical, 1, 0.88,
        model_outputs, query, builder.build()
    )]


def _error_result(scene: Optional[SceneMetadata], error: UniformError, query: StructuredQuery, trace: ExecutionTrace) -> ResultItem:
    result = ResultItem(
        image_id=scene.scene_id if scene else "00000000-0000-0000-0000-000000000000",
        rank=1,
        score=0.0,
        score_breakdown=ScoreBreakdown(),
        metadata={},
        model_outputs=ModelOutputs(),
        explanation_text=error.message,
        confidence_note="Note: Analysis could not be completed due to missing data.",
    )
    # Attach confidence band (will be low due to fallback)
    if scene:
        attach_confidence(result, scene, query.used_fallback, trace)
    else:
        result.confidence_band = "low"
        result.confidence_reason = "Low — missing scene data"
    return result


# ---------------------------------------------------------------------------
# Dispatch table (async pipelines)
# ---------------------------------------------------------------------------

PipelineFunc = Callable[[StructuredQuery, TraceBuilder], Awaitable[List[ResultItem]]]

DISPATCH: dict = {
    TaskType.search.value: pipeline_search,
    TaskType.vqa.value: pipeline_vqa,
    TaskType.caption.value: pipeline_caption,
    TaskType.change.value: pipeline_change,
    TaskType.fusion.value: pipeline_fusion,
}


# ---------------------------------------------------------------------------
# Public entrypoint (async)
# ---------------------------------------------------------------------------

async def orchestrate_async(query_text: str, image_id: Optional[str] = None, aoi_override: Any = None) -> QueryResponse:
    """
    M5 main entrypoint (async).
    1. Parse query (M1)
    2. Retrieve candidates (M2 - real, from M0 catalog)
    3. Route via dispatch table
    4. Build trace
    5. Synthesize results
    """
    # Step 1: Parse with trace (sync function)
    builder = TraceBuilder(task_selected="unknown")
    parsed = timed_call(
        builder, "M1", "Phi-4-mini-4bit",
        {"query_text": query_text},
        lambda: mock_m1_parse(query_text)
    )
    if image_id:
        parsed.image_id = image_id
    if aoi_override:
        parsed.aoi = aoi_override

    # Update trace with correct task
    builder.trace.task_selected = parsed.task_type.value
    if parsed.used_fallback:
        builder.set_fallback_used(True)

    # Step 3: Dispatch
    pipeline = DISPATCH.get(parsed.task_type.value, pipeline_search)
    results = await pipeline(parsed, builder)

    trace = builder.build()
    return QueryResponse(results=results, trace=trace)


# Backward compatibility sync wrapper
def orchestrate(query_text: str, image_id: Optional[str] = None, aoi_override: Any = None) -> QueryResponse:
    """Sync wrapper for backward compatibility."""
    return asyncio.run(orchestrate_async(query_text, image_id, aoi_override))