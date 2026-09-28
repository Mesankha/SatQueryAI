"""
M6 FastAPI Server - Main entry point for SatQuery API.

Endpoints:
- POST /query - Main query endpoint
- GET /health - Health check (wired to M0 catalog + M3 service)
- POST /replay - Offline replay mode (with LRU eviction)
"""
import json
import uuid
import time
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
from pathlib import Path
from collections import OrderedDict

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from shared.schemas import (
    QueryResponse, StructuredQuery, TaskType, Sensor, AOI,
    ResultItem, ExecutionTrace, UniformError, ErrorType
)
from shared.session import SessionManager, M1StructuredOutput, get_session_manager
from m5_controller.dispatch_table import orchestrate_async
from shared.logger import get_logger
from shared.config import get_settings

# M0 catalog
from m0_catalog.catalog import get_catalog

# M3 HTTP client
from m3_vlm.http_client import get_m3_client, close_m3_client


logger = get_logger(__name__)

# Global session manager
_session_manager: Optional[SessionManager] = None

# Replay cache with LRU eviction
REPLAY_CACHE_DIR = Path("./replay_cache")
REPLAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MAX_REPLAY_ENTRIES = 100

# In-memory LRU cache for replay (mirrors disk)
_replay_cache: OrderedDict[str, QueryResponse] = OrderedDict()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _session_manager
    settings = get_settings()
    
    _session_manager = get_session_manager({
        "storage_dir": settings.session.storage_dir,
        "ttl_hours": settings.session.ttl_hours,
    })
    
    # Initialize M3 client
    get_m3_client({
        "base_url": f"http://{settings.vlm.service_host}:{settings.vlm.service_port}",
        "timeout": settings.vlm.inference_timeout,
    })
    
    # Clean up expired sessions on startup
    _cleanup_expired_sessions()
    
    logger.info("M6 API started", extra={"module_name": "M6"})
    yield
    
    # Cleanup
    await close_m3_client()
    logger.info("M6 API shutting down", extra={"module_name": "M6"})


app = FastAPI(
    title="SatQuery AI",
    description="Natural-Language Cross-Modal Earth Observation Retrieval and Analysis Engine",
    version="0.1.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    query_text: str
    session_id: Optional[str] = None
    image_id: Optional[str] = None
    aoi_override: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    catalog_count: int = 0
    models_loaded: bool = False
    session_count: int = 0
    m3_service_status: str = "unknown"


class ReplayRequest(BaseModel):
    query_id: Optional[str] = None
    query_text: Optional[str] = None


class ReplayResponse(BaseModel):
    query_id: str
    cached: bool
    response: Optional[QueryResponse] = None
    error: Optional[str] = None


def get_session_manager_dep() -> SessionManager:
    return _session_manager


def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, Any]:
    """Get existing session or create new one."""
    if session_id:
        session = _session_manager.get_session(session_id)
        if session:
            return session_id, session
    # Create new session
    session = _session_manager.create_session()
    return session.session_id, session


def _cleanup_expired_sessions():
    """Clean up expired sessions on startup."""
    if not _session_manager:
        return
    try:
        for session_file in _session_manager.storage_dir.glob("*.json"):
            try:
                data = json.loads(session_file.read_text())
                updated_at = datetime.fromisoformat(data.get("updated_at", "").replace("Z", "+00:00"))
                if datetime.utcnow() - updated_at > _session_manager.ttl:
                    session_file.unlink()
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Session cleanup failed: {e}", extra={"module_name": "M6"})


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint.
    
    Accepts natural language query, processes through M1-M5 pipeline,
    returns structured results with execution trace.
    """
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    try:
        # Get or create session
        session_id, session = get_or_create_session(request.session_id)
        
        # Record query in session
        _session_manager.record_query(session, request.query_text)
        
        # Prepare M5 orchestrate call
        image_id = request.image_id
        aoi_override = request.aoi_override
        
        # Call M5 orchestration (async with timeout)
        try:
            response = await asyncio.wait_for(
                orchestrate_async(
                    query_text=request.query_text,
                    image_id=image_id,
                    aoi_override=aoi_override
                ),
                timeout=120.0  # Overall request timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Query timeout: {query_id}", extra={"module_name": "M6", "query_id": query_id})
            raise HTTPException(status_code=504, detail="Query processing timeout")
        
        # Update session with results
        result_ids = [r.image_id for r in response.results]
        _session_manager.record_artifacts(
            session,
            scene_ids=result_ids,
            analysis_result_ids=[query_id]
        )
        _session_manager.update_session(session)
        
        # Add query_id and session_id to response
        response.query_id = query_id
        # Store session_id in trace for debugging
        response.trace.tools_called.append({
            "tool_name": "session",
            "model_name": "SessionManager",
            "params": {"session_id": session_id},
            "latency_ms": 0,
            "success": True
        })
        
        # Cache for replay (with LRU)
        _cache_response(query_id, response)
        
        # Add timing header
        latency_ms = int((time.time() - start_time) * 1000)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}", extra={"module_name": "M6", "query_id": query_id})
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_endpoint():
    """Health check endpoint - wired to M0 catalog and M3 service."""
    try:
        # Check session manager
        session_count = 0
        if _session_manager:
            session_count = len(list(_session_manager.storage_dir.glob("*.json")))
        
        # Check M0 catalog
        catalog_count = 0
        try:
            catalog = get_catalog(read_only=True)
            catalog_count = catalog.count()
            catalog.close()
        except Exception as e:
            logger.warning(f"M0 catalog health check failed: {e}", extra={"module_name": "M6"})
        
        # Check M3 service
        models_loaded = False
        m3_status = "unavailable"
        try:
            m3_client = get_m3_client()
            health = await m3_client.health_check()
            models_loaded = health.get("model_loaded", False)
            m3_status = health.get("status", "unknown")
        except Exception as e:
            logger.warning(f"M3 health check failed: {e}", extra={"module_name": "M6"})
            m3_status = f"error: {str(e)[:50]}"
        
        # Overall status
        overall_status = "ok"
        if not models_loaded:
            overall_status = "degraded"
        
        return HealthResponse(
            status=overall_status,
            catalog_count=catalog_count,
            models_loaded=models_loaded,
            session_count=session_count,
            m3_service_status=m3_status,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}", extra={"module_name": "M6"})
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/replay", response_model=ReplayResponse)
async def replay_endpoint(request: ReplayRequest):
    """Offline replay mode - returns cached response."""
    query_id = request.query_id
    if request.query_text and not query_id:
        import hashlib
        import uuid
        h = hashlib.md5(request.query_text.lower().strip().encode()).hexdigest()
        query_id = str(uuid.UUID(h))
        
    if not query_id:
        return ReplayResponse(
            query_id="",
            cached=False,
            error="Must provide query_id or query_text"
        )
        
    cached_response = _get_cached_response(query_id)
    
    if cached_response:
        return ReplayResponse(
            query_id=query_id,
            cached=True,
            response=cached_response
        )
    
    return ReplayResponse(
        query_id=query_id,
        cached=False,
        error="No cached response found for query_id or query_text"
    )


def _cache_response(query_id: str, response: QueryResponse):
    """Cache response to disk and memory for replay with LRU eviction."""
    settings = get_settings()
    max_entries = settings.api.max_replay_entries
    
    try:
        # Update in-memory LRU
        _replay_cache[query_id] = response
        _replay_cache.move_to_end(query_id)
        
        # Evict if over limit
        while len(_replay_cache) > max_entries:
            oldest_id, _ = _replay_cache.popitem(last=False)
            # Also remove from disk
            cache_path = REPLAY_CACHE_DIR / f"{oldest_id}.json"
            if cache_path.exists():
                cache_path.unlink()
        
        # Write to disk
        cache_path = REPLAY_CACHE_DIR / f"{query_id}.json"
        cache_path.write_text(response.model_dump_json(indent=2))
        
    except Exception as e:
        logger.warning(f"Failed to cache response: {e}", extra={"module_name": "M6"})


def _get_cached_response(query_id: str) -> Optional[QueryResponse]:
    """Retrieve cached response from memory or disk."""
    # Check memory first
    if query_id in _replay_cache:
        _replay_cache.move_to_end(query_id)  # Mark as recently used
        return _replay_cache[query_id]
    
    # Check disk
    try:
        cache_path = REPLAY_CACHE_DIR / f"{query_id}.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            response = QueryResponse.model_validate(data)
            # Add to memory cache
            _replay_cache[query_id] = response
            _replay_cache.move_to_end(query_id)
            return response
    except Exception as e:
        logger.warning(f"Failed to load cached response: {e}", extra={"module_name": "M6"})
    
    return None


# Error handler for uniform error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", extra={"module_name": "M6"})
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "error_type": "internal_error",
            "message": str(exc),
            "query_id": str(uuid.uuid4())
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)