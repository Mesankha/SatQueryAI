"""
Session and Context Management for SatQuery.

Provides conversation memory, execution state, and artifact caching
for context-aware follow-up queries.
"""
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ContextDependencies(BaseModel):
    reuse_location: bool = False
    reuse_time_range: bool = False
    reuse_aoi: bool = False
    reuse_satellite_results: bool = False
    reuse_embeddings: bool = False


class Intent(BaseModel):
    task_type: str = "search"
    primary_target: Optional[str] = None
    secondary_targets: List[str] = Field(default_factory=list)
    change_analysis: bool = False


class LocationInfo(BaseModel):
    raw_text: str
    place_name: Optional[str] = None
    geometry: Optional[Dict[str, Any]] = None
    resolution_required: bool = True


class TemporalInfo(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    relative_expression: Optional[str] = None
    resolution_required: bool = True


class AnalysisRequirements(BaseModel):
    semantic_retrieval: bool = True
    vqa: bool = False
    captioning: bool = False
    grounding: bool = False
    change_detection: bool = False


class ExecutionPlan(BaseModel):
    modules: List[str] = Field(default_factory=list)


class M1StructuredOutput(BaseModel):
    """Strict output contract for M1 parser."""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    is_followup: bool = False
    context_dependencies: ContextDependencies = Field(default_factory=ContextDependencies)
    intent: Intent = Field(default_factory=Intent)
    location: Optional[LocationInfo] = None
    temporal: Optional[TemporalInfo] = None
    sensor_preferences: List[str] = Field(default_factory=list)
    analysis_requirements: AnalysisRequirements = Field(default_factory=AnalysisRequirements)
    execution_plan: ExecutionPlan = Field(default_factory=ExecutionPlan)
    confidence: float = 0.0
    clarification_required: bool = False
    clarification_question: Optional[str] = None


class ActiveContext(BaseModel):
    aoi: Optional[Dict[str, Any]] = None
    time_range: Optional[Dict[str, str]] = None
    sensor_selection: List[str] = Field(default_factory=list)
    last_task: Optional[str] = None
    last_result_ids: List[str] = Field(default_factory=list)


class Artifacts(BaseModel):
    catalog_search_id: Optional[str] = None
    scene_ids: List[str] = Field(default_factory=list)
    embedding_index_ids: List[str] = Field(default_factory=list)
    analysis_result_ids: List[str] = Field(default_factory=list)


class SessionState(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    active_context: ActiveContext = Field(default_factory=ActiveContext)
    artifacts: Artifacts = Field(default_factory=Artifacts)
    query_history: List[str] = Field(default_factory=list)


class SessionManager:
    """Manages session state persistence and context resolution."""

    def __init__(self, storage_dir: str = "./data/sessions", ttl_hours: int = 24):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

    def _session_path(self, session_id: str) -> Path:
        return self.storage_dir / f"{session_id}.json"

    def create_session(self) -> SessionState:
        """Create a new session."""
        session = SessionState()
        self._save(session)
        return session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Load session by ID."""
        path = self._session_path(session_id)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            session = SessionState.model_validate(data)
            # Check TTL
            if datetime.utcnow() - session.updated_at > self.ttl:
                self.delete_session(session_id)
                return None
            return session
        except Exception:
            return None

    def _save(self, session: SessionState):
        """Persist session to disk."""
        session.updated_at = datetime.utcnow()
        path = self._session_path(session.session_id)
        path.write_text(session.model_dump_json(indent=2))

    def update_session(self, session: SessionState):
        """Update existing session."""
        self._save(session)

    def delete_session(self, session_id: str):
        """Delete a session."""
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()

    def resolve_context(
        self,
        session: SessionState,
        m1_output: M1StructuredOutput
    ) -> M1StructuredOutput:
        """
        Resolve follow-up context by merging session artifacts into M1 output.
        This is the key function that enables context-aware queries.
        """
        if not m1_output.is_followup:
            # New query - initialize context from M1 output
            if m1_output.location and m1_output.location.geometry:
                session.active_context.aoi = m1_output.location.geometry
            if m1_output.temporal and m1_output.temporal.start_date and m1_output.temporal.end_date:
                session.active_context.time_range = {
                    "start": m1_output.temporal.start_date.isoformat(),
                    "end": m1_output.temporal.end_date.isoformat()
                }
            if m1_output.sensor_preferences:
                session.active_context.sensor_selection = m1_output.sensor_preferences
            session.active_context.last_task = m1_output.intent.task_type
            return m1_output

        # Follow-up query - reuse context based on dependencies
        deps = m1_output.context_dependencies

        if deps.reuse_location and session.active_context.aoi:
            if not m1_output.location:
                m1_output.location = LocationInfo(
                    raw_text="previous location",
                    geometry=session.active_context.aoi
                )
            elif not m1_output.location.geometry:
                m1_output.location.geometry = session.active_context.aoi

        if deps.reuse_aoi and session.active_context.aoi:
            if not m1_output.location:
                m1_output.location = LocationInfo(raw_text="previous AOI")
            m1_output.location.geometry = session.active_context.aoi

        if deps.reuse_time_range and session.active_context.time_range:
            if not m1_output.temporal:
                m1_output.temporal = TemporalInfo()
            tr = session.active_context.time_range
            m1_output.temporal.start_date = datetime.fromisoformat(tr["start"])
            m1_output.temporal.end_date = datetime.fromisoformat(tr["end"])

        if deps.reuse_satellite_results and session.artifacts.scene_ids:
            # M2 will filter to these scene IDs
            pass

        if deps.reuse_embeddings and session.artifacts.embedding_index_ids:
            # M2 will reuse these embedding indices
            pass

        # Update last task
        session.active_context.last_task = m1_output.intent.task_type

        return m1_output

    def record_query(self, session: SessionState, query_text: str):
        session.query_history.append(query_text)
        if len(session.query_history) > 20:
            session.query_history = session.query_history[-20:]

    def record_artifacts(
        self,
        session: SessionState,
        scene_ids: List[str] = None,
        embedding_index_ids: List[str] = None,
        analysis_result_ids: List[str] = None,
        catalog_search_id: str = None
    ):
        if scene_ids:
            session.artifacts.scene_ids = scene_ids
        if embedding_index_ids:
            session.artifacts.embedding_index_ids = embedding_index_ids
        if analysis_result_ids:
            session.artifacts.analysis_result_ids = analysis_result_ids
        if catalog_search_id:
            session.artifacts.catalog_search_id = catalog_search_id


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager(config: Optional[Dict] = None) -> SessionManager:
    global _session_manager
    if _session_manager is None:
        storage_dir = config.get("storage_dir", "./data/sessions") if config else "./data/sessions"
        ttl_hours = config.get("ttl_hours", 24) if config else 24
        _session_manager = SessionManager(storage_dir=storage_dir, ttl_hours=ttl_hours)
    return _session_manager