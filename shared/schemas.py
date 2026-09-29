"""
Shared Pydantic schemas for all SatQuery modules.
Defines the 5 core schemas: SceneMetadata, StructuredQuery, ResultItem, ExecutionTrace, UniformError.
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    search = "search"
    vqa = "vqa"
    caption = "caption"
    grounding = "grounding"
    change = "change"
    fusion = "fusion"
    sar_change = "sar_change"
    sar_grounding = "sar_grounding"


class Sensor(str, Enum):
    sentinel_1 = "Sentinel-1"
    sentinel_2 = "Sentinel-2"
    both = "both"


class ErrorType(str, Enum):
    no_candidates = "no_candidates"
    model_unavailable = "model_unavailable"
    invalid_input = "invalid_input"
    timeout = "timeout"
    unknown = "unknown"


# ---------------------------------------------------------------------------
# Schema 1: AOI (Area of Interest)
# ---------------------------------------------------------------------------

class AOI(BaseModel):
    type: str = "Polygon"
    coordinates: list[list[list[float]]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Schema 2: SceneMetadata (catalog entry)
# ---------------------------------------------------------------------------

class SceneMetadata(BaseModel):
    scene_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sensor: str  # "Sentinel-1", "Sentinel-2", "both"
    modality: str  # "optical", "sar", "multispectral"
    acquisition_time: datetime
    geometry_wkt: str  # WKT polygon
    file_path: Optional[str] = None
    cloud_cover: Optional[float] = None
    object: Optional[str] = None  # land cover label
    paired_scene_id: Optional[str] = None  # for bi-temporal pairs
    score: Optional[float] = None  # retrieval similarity score


# ---------------------------------------------------------------------------
# Schema 3: StructuredQuery (M1 output)
# ---------------------------------------------------------------------------

class StructuredQuery(BaseModel):
    query_text: str
    task_type: TaskType = TaskType.search
    location: Optional[str] = None
    aoi: Optional[AOI] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    sensor: Optional[Sensor] = None
    object: Optional[str] = None
    event: Optional[str] = None
    change_flag: bool = False
    question: Optional[str] = None
    referring_expression: Optional[str] = None
    confidence: float = 0.0
    used_fallback: bool = False
    image_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Schema 4: Score Breakdown & Model Outputs (sub-schemas of ResultItem)
# ---------------------------------------------------------------------------

class ScoreBreakdown(BaseModel):
    semantic_sim: float = 0.0
    geo_score: float = 0.0
    temporal_score: float = 0.0
    modality_score: float = 0.0
    change_score: float = 0.0


class ModelOutputs(BaseModel):
    caption: Optional[str] = None
    vqa_answer: Optional[str] = None
    grounding_bbox: Optional[dict[str, Any]] = None
    change_description: Optional[str] = None
    fusion_statement: Optional[str] = None
    sar_features: Optional[dict[str, Any]] = None      # deterministic SAR features
    sar_caption: Optional[str] = None                  # VLM SAR caption
    sar_vqa_answer: Optional[str] = None               # VLM SAR VQA answer
    optical_caption: Optional[str] = None              # VLM optical caption in fusion


# ---------------------------------------------------------------------------
# Schema 5: ResultItem (single ranked result)
# ---------------------------------------------------------------------------

class ResultItem(BaseModel):
    image_id: str
    rank: int = 1
    score: float = 0.0
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_outputs: ModelOutputs = Field(default_factory=ModelOutputs)
    explanation_text: str = ""
    confidence_note: str = ""
    confidence_band: str = "unknown"                # "high" | "medium" | "low" | "unknown"
    confidence_reason: str = ""                     # one line, e.g. "Medium — partial cloud cover, single observation"


# ---------------------------------------------------------------------------
# Schema 6: ToolCall & ExecutionTrace (M5 diagnostics)
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    tool_name: str
    model_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
    success: bool = True


class ExecutionTrace(BaseModel):
    task_selected: str = ""
    tools_called: list[ToolCall] = Field(default_factory=list)
    candidates_considered: int = 0
    candidates_after_filter: int = 0
    fallback_used: bool = False


# ---------------------------------------------------------------------------
# Schema 7: UniformError (cross-module error)
# ---------------------------------------------------------------------------

class UniformError(BaseModel):
    module: str
    error_type: ErrorType = ErrorType.unknown
    message: str = ""
    fallback_applied: bool = False


# ---------------------------------------------------------------------------
# Schema 8: ChangeResult (M4 output)
# ---------------------------------------------------------------------------

class ChangeResult(BaseModel):
    change_score: Optional[float] = None
    change_description: str = ""
    optical_metrics: Optional[Dict[str, Any]] = None
    sar_metrics: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Schema 9: QueryResponse (M5 final output)
# ---------------------------------------------------------------------------

class QueryResponse(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    results: list[ResultItem] = Field(default_factory=list)
    trace: ExecutionTrace = Field(default_factory=ExecutionTrace)
