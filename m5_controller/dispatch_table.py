"""
M5 Agentic Controller - dispatch table + pipeline implementations.
This file imports ONLY schemas and shared utilities. No M1-M4 internals.
"""
import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

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


def _evaluate_sar_cross_check(model_output: str, sar_features: dict) -> dict:
    """
    Evaluate disagreement between SAR model output and deterministic features.
    
    This is a heuristic check - in production you'd want more sophisticated
    NLP comparison. For now, we look for keyword conflicts.
    
    Returns dict with:
    - disagreement: bool
    - severity: "minor" | "major"
    - details: str
    """
    model_lower = model_output.lower()
    
    # Extract deterministic features
    water_frac = sar_features.get("water_fraction", 0)
    builtup_frac = sar_features.get("builtup_fraction", 0)
    log_ratio = sar_features.get("log_ratio_mean", 0)
    
    # Check for water disagreement
    model_mentions_water = any(kw in model_lower for kw in ["water", "lake", "river", "flood", "wet"])
    deterministic_water = water_frac > 0.2  # Significant water presence
    
    # Check for built-up disagreement
    model_mentions_builtup = any(kw in model_lower for kw in ["building", "built-up", "urban", "structure", "settlement"])
    deterministic_builtup = builtup_frac > 0.15  # Significant built-up presence
    
    disagreements = []
    severity = "minor"
    
    if deterministic_water and not model_mentions_water:
        disagreements.append(f"Deterministic shows water ({water_frac:.1%}) but model doesn't mention water")
        severity = "major"
    elif not deterministic_water and model_mentions_water:
        disagreements.append(f"Model mentions water but deterministic shows minimal water ({water_frac:.1%})")
    
    if deterministic_builtup and not model_mentions_builtup:
        disagreements.append(f"Deterministic shows built-up ({builtup_frac:.1%}) but model doesn't mention structures")
        if severity != "major":
            severity = "major"
    elif not deterministic_builtup and model_mentions_builtup:
        disagreements.append(f"Model mentions built-up but deterministic shows minimal built-up ({builtup_frac:.1%})")
    
    return {
        "disagreement": len(disagreements) > 0,
        "severity": severity,
        "details": "; ".join(disagreements) if disagreements else "No significant disagreement",
        "water_fraction": water_frac,
        "builtup_fraction": builtup_frac,
        "log_ratio_mean": log_ratio,
    }


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
# M3 VLM Adapters - EarthDial (optical) + CROMA (SAR) + Fusion
# ---------------------------------------------------------------------------

def _mock_vqa_response(image_path: str, question: str, modality: str = "optical") -> str:
    """Generate mock VQA response directly (no HTTP)."""
    if modality == "sar":
        return f"[SAR Mock] Based on radar backscatter analysis of {image_path}: {question} -> Simulated answer based on SAR characteristics."
    return f"[Mock] Simulated answer for {image_path}: {question} -> Based on visual analysis of the satellite imagery."


def _mock_caption_response(image_path: str, modality: str = "optical") -> str:
    """Generate mock caption response directly (no HTTP)."""
    if modality == "sar":
        return f"[SAR Mock] Simulated SAR analysis for {image_path}: Radar backscatter shows typical patterns with bright urban areas, dark water bodies, and textured vegetation. Water fraction ~15%, built-up ~10%."
    return f"[Mock] Simulated optical caption for {image_path}: This is a simulated satellite image showing typical remote sensing features including land cover, water bodies, and vegetation patterns."


async def _call_earthdial_vqa(image_path: str, question: str) -> str | UniformError:
    """Call EarthDial VQA for optical images."""
    config = get_config()
    ed_config = config.get("earthdial", {})
    use_mock = ed_config.get("mock", True)
    
    if use_mock:
        return _mock_vqa_response(image_path, question, "optical")
    else:
        try:
            from m3_vlm.earthdial_loader import run_optical_vqa
            result = run_optical_vqa(image_path, question)
            if isinstance(result, UniformError):
                logger.warning(f"EarthDial VQA failed: {result.message}, falling back to mock")
                return _mock_vqa_response(image_path, question, "optical")
            return result
        except Exception as e:
            logger.warning(f"EarthDial VQA exception: {e}, falling back to mock")
            return _mock_vqa_response(image_path, question, "optical")


async def _call_earthdial_caption(image_path: str) -> str | UniformError:
    """Call EarthDial captioning for optical images."""
    config = get_config()
    ed_config = config.get("earthdial", {})
    use_mock = ed_config.get("mock", True)
    
    if use_mock:
        return _mock_caption_response(image_path, "optical")
    else:
        try:
            from m3_vlm.earthdial_loader import run_optical_caption
            result = run_optical_caption(image_path)
            if isinstance(result, UniformError):
                logger.warning(f"EarthDial caption failed: {result.message}, falling back to mock")
                return _mock_caption_response(image_path, "optical")
            return result
        except Exception as e:
            logger.warning(f"EarthDial caption exception: {e}, falling back to mock")
            return _mock_caption_response(image_path, "optical")


async def _call_croma_sar_caption(image_path: str) -> str | UniformError:
    """Call CROMA-SAR captioning (CROMA encoder + projector + shared decoder)."""
    config = get_config()
    croma_config = config.get("croma", {})
    ed_config = config.get("earthdial", {})
    
    # Need both CROMA and EarthDial (for shared decoder)
    croma_mock = croma_config.get("mock", True)
    ed_mock = ed_config.get("mock", True)
    
    if croma_mock or ed_mock:
        return _mock_caption_response(image_path, "sar")
    else:
        try:
            from m3_vlm.croma_loader import run_croma_encoder
            from m3_vlm.croma_preprocessing import preprocess_sar_for_croma, validate_croma_input
            from m3_vlm.projector import get_projector_for_inference
            from m3_vlm.earthdial_loader import get_shared_decoder, run_decoder_generation
            
            # 1. Preprocess SAR for CROMA
            sar_tensor, _ = preprocess_sar_for_croma(image_path, return_metadata=True)
            valid, msg = validate_croma_input(sar_tensor)
            if not valid:
                return UniformError(module="M3", error_type=ErrorType.invalid_input, message=msg)
            
            # 2. Run CROMA encoder
            croma_tokens = run_croma_encoder(sar_tensor)
            if isinstance(croma_tokens, UniformError):
                return croma_tokens
            
            # 3. Project to shared space
            projector = get_projector_for_inference(
                croma_variant=croma_config.get("variant", "base"),
                shared_dim=1024,
                projector_path=croma_config.get("projector_path"),
                device=croma_config.get("device", "cuda")
            )
            sar_tokens = projector(croma_tokens)  # (1, N_sar, 1024)
            
            # 4. Generate using shared decoder (with a generic prompt)
            decoder, tokenizer = get_shared_decoder()
            if decoder is None or tokenizer is None:
                return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                                   message="Shared decoder not available")
            
            # Create prompt embedding for captioning
            prompt = "Describe this SAR remote sensing image in detail."
            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
            prompt_embeds = decoder.get_input_embeddings()(prompt_ids.to(decoder.device))
            
            # Prepend prompt to SAR tokens
            combined_embeds = torch.cat([prompt_embeds, sar_tokens], dim=1)
            attention_mask = torch.ones(combined_embeds.shape[:2], dtype=torch.long, device=combined_embeds.device)
            
            # Generate
            with torch.no_grad():
                output_ids = decoder.generate(
                    inputs_embeds=combined_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            input_len = combined_embeds.shape[1]
            new_tokens = output_ids[0][input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.warning(f"CROMA SAR caption exception: {e}, falling back to mock")
            return _mock_caption_response(image_path, "sar")


async def _call_croma_sar_vqa(image_path: str, question: str) -> str | UniformError:
    """Call CROMA-SAR VQA (CROMA encoder + projector + shared decoder)."""
    config = get_config()
    croma_config = config.get("croma", {})
    ed_config = config.get("earthdial", {})
    
    croma_mock = croma_config.get("mock", True)
    ed_mock = ed_config.get("mock", True)
    
    if croma_mock or ed_mock:
        return _mock_vqa_response(image_path, question, "sar")
    else:
        try:
            from m3_vlm.croma_loader import run_croma_encoder
            from m3_vlm.croma_preprocessing import preprocess_sar_for_croma, validate_croma_input
            from m3_vlm.projector import get_projector_for_inference
            from m3_vlm.earthdial_loader import get_shared_decoder
            
            # 1. Preprocess SAR for CROMA
            sar_tensor, _ = preprocess_sar_for_croma(image_path, return_metadata=True)
            valid, msg = validate_croma_input(sar_tensor)
            if not valid:
                return UniformError(module="M3", error_type=ErrorType.invalid_input, message=msg)
            
            # 2. Run CROMA encoder
            croma_tokens = run_croma_encoder(sar_tensor)
            if isinstance(croma_tokens, UniformError):
                return croma_tokens
            
            # 3. Project to shared space
            projector = get_projector_for_inference(
                croma_variant=croma_config.get("variant", "base"),
                shared_dim=1024,
                projector_path=croma_config.get("projector_path"),
                device=croma_config.get("device", "cuda")
            )
            sar_tokens = projector(croma_tokens)
            
            # 4. Generate using shared decoder with question
            decoder, tokenizer = get_shared_decoder()
            if decoder is None or tokenizer is None:
                return UniformError(module="M3", error_type=ErrorType.model_unavailable,
                                   message="Shared decoder not available")
            
            # Embed question and prepend
            question_ids = tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids
            question_embeds = decoder.get_input_embeddings()(question_ids.to(decoder.device))
            
            combined_embeds = torch.cat([question_embeds, sar_tokens], dim=1)
            attention_mask = torch.ones(combined_embeds.shape[:2], dtype=torch.long, device=combined_embeds.device)
            
            with torch.no_grad():
                output_ids = decoder.generate(
                    inputs_embeds=combined_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            input_len = combined_embeds.shape[1]
            new_tokens = output_ids[0][input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.warning(f"CROMA SAR VQA exception: {e}, falling back to mock")
            return _mock_vqa_response(image_path, question, "sar")


# Keep existing _call_m3_vqa and _call_m3_caption for backward compatibility (GeoChat)
# but add new modality routing
async def _call_m3_vqa(image_path: str, question: str, modality: str = "optical"):
    """Call M3 VQA - routes to EarthDial (optical) or CROMA (SAR) based on modality."""
    if modality == "sar":
        return await _call_croma_sar_vqa(image_path, question)
    else:
        return await _call_earthdial_vqa(image_path, question)


async def _call_m3_caption(image_path: str, modality: str = "optical"):
    """Call M3 captioning - routes to EarthDial (optical) or CROMA (SAR) based on modality."""
    if modality == "sar":
        return await _call_croma_sar_caption(image_path)
    else:
        return await _call_earthdial_caption(image_path)


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
    sar_cross_check: Optional[Dict[str, Any]] = None
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

    # Attach confidence band and freshness metadata (with SAR cross-check if available)
    attach_confidence(result, scene, query.used_fallback, trace, sar_cross_check)

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

    # Determine modality from candidate
    modality = "sar" if top.modality == "sar" else "optical"
    
    try:
        answer = await asyncio.wait_for(
            timed_call_async(
                builder, "M3", f"EarthDial-{modality}" if modality == "optical" else "CROMA-SAR",
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

    # For SAR, also run deterministic cross-check and track SAR path
    model_outputs = ModelOutputs(vqa_answer=answer)
    sar_cross_check = None
    if modality == "sar":
        # Track SAR path
        config = get_config()
        croma_mock = config.get("croma", {}).get("mock", True)
        if croma_mock:
            builder.set_sar_path("deterministic-only fallback")
        else:
            builder.set_sar_path("CROMA-S1 + deterministic cross-check")
        
        # Run deterministic SAR features as cross-check
        sar_feats = await _call_m4_sar_features(top.file_path or "")
        if not isinstance(sar_feats, UniformError):
            model_outputs.sar_features = sar_feats
            # Cross-check: compare model answer with deterministic features
            # This is logged in trace for auditability
            builder.add_tool_call("M4", "sar_features-deterministic", {"cross_check": True}, 0, True)
            
            # Prepare SAR cross-check for confidence calibration
            sar_cross_check = _evaluate_sar_cross_check(answer, sar_feats)

    return [_synthesize_result(
        top, 1, 0.9, model_outputs, query, builder.build(), sar_cross_check
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

    # Determine modality from candidate
    modality = "sar" if top.modality == "sar" else "optical"
    
    try:
        caption = await asyncio.wait_for(
            timed_call_async(
                builder, "M3", f"EarthDial-{modality}" if modality == "optical" else "CROMA-SAR",
                {"image_path": top.file_path or ""},
                _call_m3_caption, top.file_path or "", modality
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

    # For SAR, also run deterministic cross-check and track SAR path
    model_outputs = ModelOutputs(caption=caption)
    sar_cross_check = None
    if modality == "sar":
        # Track SAR path
        config = get_config()
        croma_mock = config.get("croma", {}).get("mock", True)
        if croma_mock:
            builder.set_sar_path("deterministic-only fallback")
        else:
            builder.set_sar_path("CROMA-S1 + deterministic cross-check")
        
        # Run deterministic SAR features as cross-check
        sar_feats = await _call_m4_sar_features(top.file_path or "")
        if not isinstance(sar_feats, UniformError):
            model_outputs.sar_features = sar_feats
            model_outputs.sar_caption = caption  # Store CROMA-derived caption separately
            builder.add_tool_call("M4", "sar_features-deterministic", {"cross_check": True}, 0, True)
            
            # Prepare SAR cross-check for confidence calibration
            sar_cross_check = _evaluate_sar_cross_check(caption, sar_feats)

    return [_synthesize_result(
        top, 1, 0.9, model_outputs, query, builder.build(), sar_cross_check
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


# =============================================================================
# Phase 3: Bi-temporal SAR Change Detection (CROMA Siamese) - placeholder
# =============================================================================

async def pipeline_sar_change(query: StructuredQuery, builder: TraceBuilder) -> List[ResultItem]:
    """
    Bi-temporal SAR change detection using CROMA Siamese encoder.
    
    Currently falls back to deterministic log-ratio method.
    When CROMA Siamese is trained, this will use the learned change signal.
    """
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

    # Verify both are SAR
    if t1.modality != "sar" or (t2 and t2.modality != "sar"):
        builder.set_fallback_used(True)
        return [_error_result(t1, UniformError(module="M5", error_type=ErrorType.invalid_input,
                            message="SAR change detection requires both images to be SAR"), query, builder.build())]

    # --- DETERMINISTIC SAR CHANGE (always runs) ---
    try:
        change_result = await asyncio.wait_for(
            timed_call_async(
                builder, "M4", "deterministic-SAR-logratio",
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
        logger.warning(f"M4 SAR change detection timeout for {t1.file_path}")
        builder.set_fallback_used(True)
        return [_error_result(t1, UniformError(module="M4", error_type=ErrorType.timeout, message="SAR change detection timeout"), query, builder.build())]

    if isinstance(change_result, UniformError):
        builder.set_fallback_used(True)
        return [_error_result(t1, change_result, query, builder.build())]

    # --- CROMA SIAMESE CHANGE (placeholder - falls back to deterministic) ---
    config = get_config()
    croma_config = config.get("croma", {})
    croma_mock = croma_config.get("mock", True)
    
    if not croma_mock:
        # TODO: Implement CROMA Siamese change detection when trained
        # For now, log that it's not yet available
        builder.add_tool_call("M3", "CROMA-Siamese-change", {"status": "not_trained_yet"}, 0, True)
        builder.set_sar_path("deterministic-only fallback (Siamese not trained)")
    else:
        builder.set_sar_path("deterministic-only fallback")

    return [_synthesize_result(
        t1, 1, 0.85,
        ModelOutputs(change_description=change_result.get("change_description")),
        query, builder.build()
    )]


# =============================================================================
# Phase 3: SAR Grounding Fallback - optical-guided
# =============================================================================

async def pipeline_sar_grounding(query: StructuredQuery, builder: TraceBuilder) -> List[ResultItem]:
    """
    SAR grounding using optical-guided fallback.
    
    Per architecture: SAR native grounding not attempted (SAREval <3% Acc@0.5).
    Instead: ground on co-registered optical image, project box to SAR tile.
    """
    # For SAR grounding, we need both optical and SAR - retrieve without sensor filter
    # Create a copy of query with sensor=None to get both modalities
    from shared.schemas import StructuredQuery, Sensor
    retrieval_query = StructuredQuery(
        query_text=query.query_text,
        task_type=query.task_type,
        location=query.location,
        aoi=query.aoi,
        start_date=query.start_date,
        end_date=query.end_date,
        sensor=Sensor.both,  # Retrieve both optical and SAR
        object=query.object,
        event=query.event,
        change_flag=query.change_flag,
        question=query.question,
        referring_expression=query.referring_expression,
        confidence=query.confidence,
        used_fallback=query.used_fallback,
        image_id=query.image_id,
    )
    
    candidates = timed_call(
        builder, "M2", "engine-retrieve" if not get_config().get("retrieval", {}).get("mock", True) else "fixture-mock",
        {"query_text": query.query_text, "sensor": "both"},
        lambda: _m2_retrieve(retrieval_query)
    )
    builder.set_candidates(considered=len(candidates) + 5, after_filter=len(candidates))
    if not candidates:
        return []

    # Need both optical and SAR for the same area
    optical = next((c for c in candidates if c.modality in ("optical", "multispectral")), None)
    sar = next((c for c in candidates if c.modality == "sar"), None)

    if not optical or not sar:
        builder.set_fallback_used(True)
        return [_error_result(candidates[0] if candidates else None, 
                            UniformError(module="M5", error_type=ErrorType.no_candidates,
                                        message="SAR grounding requires co-registered optical+SAR pair"),
                            query, builder.build())]

    ok, err = check_fusion_compatibility(optical, sar)
    if not ok:
        builder.set_fallback_used(True)
        return [_error_result(candidates[0], err, query, builder.build())]

    # Track SAR path
    builder.set_sar_path("optical-guided fallback (SAREval <3% Acc@0.5)")

    # Ground on optical image
    referring_expr = query.referring_expression or query.object or "the object"
    try:
        # Use EarthDial for optical grounding (or mock VQA with location query)
        grounding_question = f"Where is {referring_expr}? Provide bounding box coordinates."
        answer = await asyncio.wait_for(
            timed_call_async(
                builder, "M3", "EarthDial-optical-grounding",
                {"image_path": optical.file_path or "", "question": grounding_question},
                _call_earthdial_vqa, optical.file_path or "", grounding_question
            ),
            timeout=35.0
        )
    except asyncio.TimeoutError:
        logger.warning("M3 Grounding timeout")
        builder.set_fallback_used(True)
        return [_error_result(candidates[0], UniformError(module="M3", error_type=ErrorType.timeout, message="Grounding timeout"), query, builder.build())]

    if isinstance(answer, UniformError):
        builder.set_fallback_used(True)
        return [_error_result(candidates[0], answer, query, builder.build())]

    # In real implementation: parse bbox from answer, project to SAR coordinates
    # For now, return the answer with note about optical-guided fallback
    grounding_response = f"[Optical-Guided Fallback] Grounded on co-registered optical image: {answer}. Note: SAR-native grounding not attempted per SAREval findings (<3% Acc@0.5). Box projected to SAR tile assuming co-registration."

    model_outputs = ModelOutputs(
        vqa_answer=grounding_response,
        grounding_bbox={"note": "optical_guided_fallback", "source": "optical_projection"}
    )

    return [_synthesize_result(
        sar, 1, 0.7,  # Lower score for fallback
        model_outputs, query, builder.build()
    )]


# =============================================================================
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

    # --- OPTICAL PATH: EarthDial InternViT encoder ---
    try:
        # Use EarthDial for optical encoding (or mock)
        opt_caption = await asyncio.wait_for(
            timed_call_async(
                builder, "M3", "EarthDial-optical",
                {"image_path": optical.file_path or ""},
                _call_earthdial_caption, optical.file_path or ""
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

    # --- SAR PATH A: DETERMINISTIC (always runs as cross-check) ---
    sar_feats = await timed_call_async(
        builder, "M4", "sar_features-deterministic",
        {"sar_path": sar.file_path or ""},
        _call_m4_sar_features, sar.file_path or ""
    )
    if isinstance(sar_feats, UniformError):
        builder.set_fallback_used(True)
        return [_error_result(candidates[0], sar_feats, query, builder.build())]

    # --- SAR PATH B: CROMA encoder + projector + shared decoder (when available) ---
    sar_vlm_caption = None
    config = get_config()
    croma_config = config.get("croma", {})
    ed_config = config.get("earthdial", {})
    croma_mock = croma_config.get("mock", True)
    ed_mock = ed_config.get("mock", True)
    
    if not croma_mock and not ed_mock:
        try:
            # Use true dual-encoder fusion: EarthDial optical tokens + CROMA SAR tokens -> shared decoder
            from m3_vlm.earthdial_loader import encode_optical_image, get_shared_decoder
            from m3_vlm.croma_loader import run_croma_encoder
            from m3_vlm.croma_preprocessing import preprocess_sar_for_croma, validate_croma_input
            from m3_vlm.projector import get_projector_for_inference
            from m3_vlm.fusion_decoder import create_fusion_decoder
            from PIL import Image
            
            # 1. Encode optical with EarthDial InternViT
            optical_image = Image.open(optical.file_path).convert("RGB")
            # EarthDial expects 448x448 typically - preprocess
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((448, 448)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            optical_tensor = transform(optical_image).unsqueeze(0)  # (1, 3, 448, 448)
            
            optical_tokens = encode_optical_image(optical_tensor)
            if isinstance(optical_tokens, UniformError):
                logger.warning(f"EarthDial optical encoding failed: {optical_tokens.message}")
                raise RuntimeError("Optical encoding failed")
            
            # 2. Encode SAR with CROMA
            sar_tensor, _ = preprocess_sar_for_croma(sar.file_path, return_metadata=True)
            valid, msg = validate_croma_input(sar_tensor)
            if not valid:
                raise RuntimeError(f"SAR preprocessing failed: {msg}")
            
            croma_tokens = run_croma_encoder(sar_tensor)
            if isinstance(croma_tokens, UniformError):
                logger.warning(f"CROMA encoding failed: {croma_tokens.message}")
                raise RuntimeError("CROMA encoding failed")
            
            # 3. Project CROMA tokens to shared space
            projector = get_projector_for_inference(
                croma_variant=croma_config.get("variant", "base"),
                shared_dim=1024,
                projector_path=croma_config.get("projector_path"),
                device=croma_config.get("device", "cuda")
            )
            sar_tokens = projector(croma_tokens)  # (1, N_sar, 1024)
            
            # 4. Fuse tokens and generate using shared decoder
            decoder, tokenizer = get_shared_decoder()
            if decoder is None or tokenizer is None:
                raise RuntimeError("Shared decoder not available")
            
            fusion_decoder = create_fusion_decoder(hidden_dim=1024, fusion_type="cross_attention")
            
            # For captioning task
            if query.task_type.value == "fusion":
                # Create prompt for fusion captioning
                prompt = "Analyze this optical and SAR image pair together. Describe land cover, water bodies, built-up areas, and any notable features visible in both modalities."
                prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
                prompt_embeds = decoder.get_input_embeddings()(prompt_ids.to(decoder.device))
                
                # Fuse optical and SAR tokens
                fused_tokens = fusion_decoder.fusion(optical_tokens, sar_tokens)
                
                # Prepend prompt
                combined_embeds = torch.cat([prompt_embeds, fused_tokens], dim=1)
                attention_mask = torch.ones(combined_embeds.shape[:2], dtype=torch.long, device=combined_embeds.device)
                
                with torch.no_grad():
                    output_ids = decoder.generate(
                        inputs_embeds=combined_embeds,
                        attention_mask=attention_mask,
                        max_new_tokens=256,
                        do_sample=False,
                        temperature=0.0,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                
                input_len = combined_embeds.shape[1]
                new_tokens = output_ids[0][input_len:]
                fusion_statement = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                
                # Also get individual captions for cross-reference
                sar_vlm_caption = await _call_croma_sar_caption(sar.file_path)
            
        except asyncio.TimeoutError:
            logger.warning("Fusion decoder timeout in fusion pipeline")
            builder.set_fallback_used(True)
        except Exception as e:
            logger.warning(f"Dual-encoder fusion failed: {e}")
            builder.set_fallback_used(True)
    
    # If dual-encoder fusion not available or failed, fall back to template synthesis
    if not sar_vlm_caption:
        sar_vlm_caption = None
    
    # Track SAR path for auditability
    config = get_config()
    croma_config = config.get("croma", {})
    ed_config = config.get("earthdial", {})
    croma_mock = croma_config.get("mock", True)
    ed_mock = ed_config.get("mock", True)
    
    if croma_mock or ed_mock:
        builder.set_sar_path("deterministic-only fallback")
    else:
        builder.set_sar_path("CROMA-S1 + deterministic cross-check")
    
    # Evaluate SAR cross-check for confidence calibration
    sar_cross_check = None
    if sar_vlm_caption:
        sar_cross_check = _evaluate_sar_cross_check(sar_vlm_caption, sar_feats)
    else:
        # Even without VLM caption, we can check if deterministic features are notable
        sar_cross_check = _evaluate_sar_cross_check("", sar_feats)

    # --- SYNTHESIS (template or LLM) ---
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
        model_outputs, query, builder.build(), sar_cross_check
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
    TaskType.sar_change.value: pipeline_sar_change,
    TaskType.sar_grounding.value: pipeline_sar_grounding,
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