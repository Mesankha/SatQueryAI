"""
M6 FastAPI Server - Main entry point for SatQuery API.
Endpoints:
- POST /query - Main query endpoint
- POST /upload - Direct image upload for caption/VQA/fusion/change
- GET /report - Downloadable JSON/PDF report
- GET /gui - Simple HTML interface
"""
import os
import uuid
import shutil
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from shared.schemas import (
    QueryResponse, StructuredQuery, TaskType, Sensor, AOI,
    ResultItem, ExecutionTrace, UniformError, ErrorType
)
from shared.config import get_config, get_settings
from shared.logger import get_logger
from m5_controller.dispatch_table import orchestrate_async
from m5_controller.compatibility_checker import check_pair_compatibility

logger = get_logger(__name__)

# Upload directory
UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Reports directory
REPORTS_DIR = Path("./data/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("M6 API starting up")
    yield
    logger.info("M6 API shutting down")


app = FastAPI(
    title="SatQuery AI",
    description="Natural-Language Cross-Modal Earth Observation Retrieval and Analysis Engine",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    query_text: str
    image_id: Optional[str] = None
    aoi_override: Optional[Dict[str, Any]] = None


class UploadResponse(BaseModel):
    query_id: str
    results: List[ResultItem]
    trace: ExecutionTrace
    uploaded_files: List[str]


def _sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal."""
    name = Path(filename).name
    name = "".join(c for c in name if c.isalnum() or c in "._-")
    return name


def _validate_file(file: UploadFile) -> Optional[str]:
    """Validate uploaded file. Returns error message if invalid, None if valid."""
    if not file.filename:
        return "No filename provided"
    
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return f"File type {ext} not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check file size by reading content
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_FILE_SIZE:
        return f"File too large: {size} bytes (max {MAX_FILE_SIZE})"
    
    return None


def _infer_task_type(query_text: str, has_question: bool, has_file2: bool) -> TaskType:
    """Infer task type from upload parameters."""
    q = query_text.lower()
    if has_file2:
        if "combine" in q or "optical and sar" in q or "fusion" in q:
            return TaskType.fusion
        return TaskType.change
    if has_question:
        return TaskType.vqa
    return TaskType.caption


def _build_report_content(response: QueryResponse, format: str = "json") -> Dict[str, Any]:
    """Build structured report content from QueryResponse."""
    report = {
        "report_metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "query_id": response.query_id,
            "satquery_version": "0.1.0",
        },
        "query": {
            "text": "",  # Will be filled by caller if available
        },
        "results": [],
        "execution_trace": {},
    }
    
    # Add results
    for r in response.results:
        result_entry = {
            "rank": r.rank,
            "image_id": r.image_id,
            "score": r.score,
            "confidence_band": r.confidence_band,
            "confidence_reason": r.confidence_reason,
            "metadata": r.metadata,
            "explanation": r.explanation_text,
            "model_outputs": r.model_outputs.model_dump() if r.model_outputs else {},
            "score_breakdown": r.score_breakdown.model_dump() if r.score_breakdown else {},
        }
        report["results"].append(result_entry)
    
    # Add execution trace
    if response.trace:
        report["execution_trace"] = {
            "task_selected": response.trace.task_selected,
            "tools_called": [tc.model_dump() for tc in response.trace.tools_called],
            "candidates_considered": response.trace.candidates_considered,
            "candidates_after_filter": response.trace.candidates_after_filter,
            "fallback_used": response.trace.fallback_used,
        }
    
    return report


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint.
    Accepts natural language query, processes through M1-M5 pipeline,
    returns structured results with execution trace.
    """
    try:
        settings = get_settings()
        timeout = settings.api.request_timeout
        
        response = await asyncio.wait_for(
            orchestrate_async(
                query_text=request.query_text,
                image_id=request.image_id,
                aoi_override=request.aoi_override
            ),
            timeout=timeout
        )
        return response
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Query processing timeout")
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=UploadResponse)
async def upload_endpoint(
    file: UploadFile = File(...),
    file2: Optional[UploadFile] = File(None),
    question: Optional[str] = Form(None),
    intent: Optional[str] = Form(None)
):
    """
    Direct image upload endpoint.
    - Single image + optional question -> caption (no question) or VQA (with question)
    - Image pair -> fusion or change based on intent or query text
    Saves to upload_dir, runs compatibility checks, calls orchestrate.
    """
    # Validate files
    error = _validate_file(file)
    if error:
        raise HTTPException(status_code=422, detail=error)
    
    file_paths = []
    try:
        # Save first file
        filename1 = _sanitize_filename(file.filename)
        path1 = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename1}"
        with open(path1, "wb") as f:
            shutil.copyfileobj(file.file, f)
        file_paths.append(str(path1))
        
        # Save second file if provided
        if file2 and file2.filename:
            error = _validate_file(file2)
            if error:
                raise HTTPException(status_code=422, detail=error)
            filename2 = _sanitize_filename(file2.filename)
            path2 = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename2}"
            with open(path2, "wb") as f:
                shutil.copyfileobj(file2.file, f)
            file_paths.append(str(path2))
        
        # Determine task type
        has_question = bool(question and question.strip())
        has_file2 = len(file_paths) == 2
        
        if intent:
            task_type = TaskType(intent) if intent in ["fusion", "change"] else TaskType.caption
        else:
            task_type = _infer_task_type(question or "", has_question, has_file2)
        
        # For pair uploads, check compatibility
        if has_file2:
            ok, reason = check_pair_compatibility(file_paths[0], file_paths[1])
            if not ok:
                raise HTTPException(status_code=422, detail=f"Pair compatibility check failed: {reason}")
        
        # Call orchestrate with uploaded image path(s)
        # Build query text from question or default
        query_text = question or "Describe this image" if not has_file2 else \
                     "Analyze these images for changes" if task_type == TaskType.change else \
                     "Fuse these optical and SAR images"
        
        if has_file2:
            # For pair, we need to extend orchestrate signature
            # For now, use first image as primary and pass second via aoi_override
            response = await orchestrate_async(
                query_text=query_text,
                image_id=None,
                aoi_override={"uploaded_image_path_2": file_paths[1]}
            )
        else:
            response = await orchestrate_async(
                query_text=query_text,
                image_id=None,
                aoi_override={"uploaded_image_path": file_paths[0]}
            )
        
        return UploadResponse(
            query_id=response.query_id,
            results=response.results,
            trace=response.trace,
            uploaded_files=file_paths
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report")
async def report_endpoint(
    query_id: str = Query(..., description="Query ID from /query or /upload response"),
    format: str = Query("json", description="Output format: json or pdf"),
    query_text: Optional[str] = Query(None, description="Original query text for report context")
):
    """
    Downloadable report endpoint.
    Returns a structured report (JSON or PDF) for a given query_id.
    Note: In this prototype, query_id is used to reconstruct report from latest response.
    For production, implement persistent storage of query responses.
    """
    if format not in ("json", "pdf"):
        raise HTTPException(status_code=422, detail="Format must be 'json' or 'pdf'")
    
    # In a real implementation, we'd fetch the stored response by query_id
    # For this prototype, we return a template report structure
    # The frontend should call this after a query with the returned query_id
    
    report = {
        "report_metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "query_id": query_id,
            "satquery_version": "0.1.0",
            "format": format,
        },
        "query": {
            "text": query_text or "Not provided",
        },
        "results": [],
        "execution_trace": {},
        "note": "This is a prototype report. In production, query responses are persisted and retrieved by query_id."
    }
    
    if format == "json":
        return JSONResponse(content=report)
    
    # PDF generation placeholder
    # In production, use reportlab or weasyprint
    return JSONResponse(
        content={**report, "error": "PDF generation not implemented in prototype. Use format=json."},
        status_code=501
    )


# Catalog endpoints for frontend
@app.get("/catalog")
async def catalog_list(
    sensor: Optional[str] = Query(None),
    modality: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List catalog scenes with optional filters."""
    from m0_catalog.catalog import get_catalog
    catalog = get_catalog(read_only=True)
    
    scenes = catalog.get_all_scenes(limit=limit + offset)
    scenes = scenes[offset:offset + limit]
    
    if sensor:
        scenes = [s for s in scenes if s.sensor == sensor]
    if modality:
        scenes = [s for s in scenes if s.modality == modality]
    
    total = len(scenes)
    
    return {
        "scenes": [s.model_dump() for s in scenes],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/catalog/search")
async def catalog_search(
    query_text: str = Query(...),
    sensor: Optional[str] = Query(None),
    object: Optional[str] = Query(None),
    top_k: int = Query(10, ge=1, le=50),
):
    """Search catalog using M2 retrieval (uses mock or real engine based on config)."""
    from shared.schemas import StructuredQuery, TaskType, Sensor
    from m5_controller.dispatch_table import _m2_retrieve
    
    sensor_enum = None
    if sensor == "Sentinel-1":
        sensor_enum = Sensor.sentinel_1
    elif sensor == "Sentinel-2":
        sensor_enum = Sensor.sentinel_2
    elif sensor == "both":
        sensor_enum = Sensor.both
    
    query = StructuredQuery(
        query_text=query_text,
        task_type=TaskType.search,
        sensor=sensor_enum,
        object=object,
    )
    
    candidates = _m2_retrieve(query)
    candidates = candidates[:top_k]
    
    return [c.model_dump() for c in candidates]


@app.get("/catalog/{scene_id}")
async def catalog_get_scene(scene_id: str):
    """Get scene by ID."""
    from m0_catalog.catalog import get_catalog
    catalog = get_catalog(read_only=True)
    scene = catalog.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")
    return scene.model_dump()


@app.get("/catalog/{scene_id}/pair")
async def catalog_get_pair(scene_id: str):
    """Get paired scene for fusion/change."""
    from m0_catalog.catalog import get_catalog
    catalog = get_catalog(read_only=True)
    scene = catalog.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")
    
    if not scene.paired_scene_id:
        raise HTTPException(status_code=404, detail=f"No paired scene for {scene_id}")
    
    paired = catalog.get_scene(scene.paired_scene_id)
    if not paired:
        raise HTTPException(status_code=404, detail=f"Paired scene {scene.paired_scene_id} not found")
    
    return paired.model_dump()


@app.get("/gui", response_class=HTMLResponse)
async def gui_endpoint():
    """Serve the simple HTML GUI."""
    gui_path = Path(__file__).parent / "static" / "index.html"
    if gui_path.exists():
        return HTMLResponse(content=gui_path.read_text())
    return HTMLResponse(content="<h1>GUI not found</h1><p>Run build step to generate static files.</p>")


@app.get("/config-status")
async def config_status_endpoint():
    """Return current mock/real configuration status for UI."""
    from shared.config import get_config
    config = get_config()
    vlm = config.get("vlm", {})
    change = config.get("change", {})
    retrieval = config.get("retrieval", {})
    return {
        "vlm_mock": vlm.get("mock", True),
        "sar_mock": vlm.get("sar_mock", True),
        "change_mock": change.get("mock", True),
        "retrieval_mock": retrieval.get("mock", True),
    }


@app.get("/health")
async def health_endpoint():
    """Health check."""
    return {"status": "ok", "service": "satquery-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)