"""
M4 Change Detection Orchestrator.

Responsibilities:
- Coordinate temporal comparison of t1 and t2 observations.
- Dispatch to optical (Sentinel-2) or SAR (Sentinel-1) analysis.
- Own M4 workflow & fallback branching (missing images, missing metadata, calculation failures, LLM polish failures).
- Return canonical ChangeResult (change_score: float | None, change_description: str).

Lower-level analysis modules remain focused strictly on mathematical computation
and do not implement workflow-level fallback decisions.
"""

from typing import Any, Callable, Dict, Optional, Union
from pathlib import Path
import numpy as np

from shared.schemas import ChangeResult
from m4_changedetect.optical.optical_analyzer import analyze_optical
from m4_changedetect.sar.sar_analyzer import analyze_sar
from m4_changedetect.scoring.change_score import calculate_change_score
from m4_changedetect.explanation.templates import generate_change_description
from m4_changedetect.explanation.llm_polish import polish_change_description


class ChangeDetector:
    """
    Main M4 Change Detector Orchestrator class.
    Manages workflow dispatching, sensor routing, and fallback decision hierarchy.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        llm_callable: Optional[Callable[[str], str]] = None
    ):
        """
        Initializes ChangeDetector with optional pipeline configuration and optional LLM callable.

        :param config: Dictionary containing pipeline configuration flags (e.g. {"enable_llm_polish": True}).
        :param llm_callable: Optional injectable LLM generation callable for Stage 4 polishing.
        """
        self.config = config or {}
        self.llm_callable = llm_callable

    def detect_change(
        self,
        image_t1: Union[str, Path, Any],
        image_t2: Union[str, Path, Any],
        metadata_t1: Optional[Dict[str, Any]] = None,
        metadata_t2: Optional[Dict[str, Any]] = None
    ) -> ChangeResult:
        """
        Main M4 interface for bi-temporal satellite change detection.

        Coordinates temporal comparison of t1 and t2 observations across Optical (Sentinel-2)
        and SAR (Sentinel-1) modalities, handles workflow fallbacks, and produces canonical ChangeResult.

        :param image_t1: Image input / bands container for observation t1.
        :param image_t2: Image input / bands container for observation t2.
        :param metadata_t1: Observation metadata dictionary for t1.
        :param metadata_t2: Observation metadata dictionary for t2.
        :return: Canonical ChangeResult containing change_score (float | None) and change_description (str).
        """
        # --- Fallback 1: Missing Image Inputs ---
        if image_t1 is None or image_t2 is None:
            return ChangeResult(
                change_score=None,
                change_description="Change detection unavailable: Missing bi-temporal image pair."
            )

        # --- Fallback 2: Missing Metadata ---
        if metadata_t1 is None or metadata_t2 is None or not isinstance(metadata_t1, dict) or not isinstance(metadata_t2, dict):
            return ChangeResult(
                change_score=None,
                change_description="Change detection unavailable: Missing metadata for temporal pair."
            )

        constellation_t1 = str(metadata_t1.get("constellation", "")).upper()
        constellation_t2 = str(metadata_t2.get("constellation", "")).upper()
        sensor_type = str(metadata_t1.get("sensor_type", "")).upper()

        optical_results: Optional[Dict[str, Any]] = None
        sar_results: Optional[Dict[str, Any]] = None

        # --- Fallback 3: Sensor Routing & Partial Computation ---
        is_optical = any(kw in constellation_t1 or kw in sensor_type for kw in ("SENTINEL-2", "S2", "OPTICAL"))
        is_sar = any(kw in constellation_t1 or kw in sensor_type for kw in ("SENTINEL-1", "S1", "SAR"))

        # If constellation is unspecified, auto-detect based on image keys/structure
        if not is_optical and not is_sar:
            if isinstance(image_t1, dict):
                if any(k in image_t1 for k in ("nir", "NIR", "red", "RED", "b8", "B8")):
                    is_optical = True
                if any(k in image_t1 for k in ("sar", "SAR", "vv", "VV", "vh", "VH", "intensity")):
                    is_sar = True
            elif isinstance(image_t1, (np.ndarray, list)):
                is_sar = True

        # Attempt Optical Analysis
        if is_optical:
            try:
                optical_results = analyze_optical(image_t1, image_t2, metadata_t1, metadata_t2)
            except Exception:
                optical_results = None

        # Attempt SAR Analysis
        if is_sar:
            try:
                sar_results = analyze_sar(image_t1, image_t2, metadata_t1, metadata_t2)
            except Exception:
                sar_results = None

        # If auto-detect failed or sensor type unspecified, attempt both safely
        if optical_results is None and sar_results is None:
            try:
                optical_results = analyze_optical(image_t1, image_t2, metadata_t1, metadata_t2)
            except Exception:
                pass

            try:
                sar_results = analyze_sar(image_t1, image_t2, metadata_t1, metadata_t2)
            except Exception:
                pass

        if optical_results is None and sar_results is None:
            return ChangeResult(
                change_score=None,
                change_description="Change detection unavailable: Insufficient or missing required band/sensor data."
            )

        # --- Fallback 4: Compute Normalized Composite Change Score ---
        change_score: Optional[float] = None
        try:
            change_score = calculate_change_score(optical_results, sar_results)
        except Exception:
            change_score = None

        # --- Fallback 5: Generate Stage 3 Deterministic Explanation ---
        try:
            change_description = generate_change_description(
                change_score=change_score,
                optical_results=optical_results,
                sar_results=sar_results,
                metadata_t1=metadata_t1,
                metadata_t2=metadata_t2
            )
        except Exception as err:
            change_description = f"Change description unavailable: {str(err)}"

        # --- Fallback 6: Optional Stage 4 LLM Polish with Guardrail Fallback ---
        if self.config.get("enable_llm_polish", False) or self.llm_callable is not None:
            try:
                polished_desc = polish_change_description(
                    deterministic_description=change_description,
                    config=self.config,
                    llm_callable=self.llm_callable
                )
                change_description = polished_desc
            except Exception:
                # Strictly preserve Stage 3 deterministic explanation on LLM polish failure
                pass

        return ChangeResult(
            change_score=change_score,
            change_description=change_description
        )


def detect_change(
    image_t1: Union[str, Path, Any],
    image_t2: Union[str, Path, Any],
    metadata_t1: Optional[Dict[str, Any]] = None,
    metadata_t2: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    llm_callable: Optional[Callable[[str], str]] = None
) -> ChangeResult:
    """
    Module-level convenience function exposing main M4 interface.
    """
    orchestrator = ChangeDetector(config=config, llm_callable=llm_callable)
    return orchestrator.detect_change(image_t1, image_t2, metadata_t1, metadata_t2)
