"""
M3 HTTP Service — Separate process for GPU isolation.
Run with: python -m m3_vlm.service

Fixes applied (vs original):
  F61: Replace deprecated @before_first_request with init_app pattern
  F62, F63: Single persistent event loop for all requests
  F64: thread=True for concurrent requests (with proper locking)
  F65: Health endpoint reports warm-up, OOM, last error
  F66: CORS enabled
  F67: Per-request config override (low priority)
  F68: Robust sys.path setup
  + NEW: /ready, /stats, /reload endpoints
  + NEW: Graceful shutdown handling
  + NEW: Request ID and timing
  + NEW: Multiprocess-safe (gunicorn compatible)
"""
import os
import sys
import asyncio
import logging
import signal
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# Path setup (F68 FIX: more robust)
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent  # satquery/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports (after path setup)
# ---------------------------------------------------------------------------
from flask import Flask, request, jsonify, g
from flask_cors import CORS

from m3_vlm.vlm_service import (
    initialize_service, get_service, shutdown_service,
)
from m3_vlm.api import run_vqa, run_caption, run_grounding
from shared.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Persistent event loop (F62, F63 FIX)
# ---------------------------------------------------------------------------
_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None
_loop_ready = threading.Event()


def _start_event_loop():
    """Start a dedicated event loop in a background thread for async ops."""
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop_ready.set()
    logger.info("M3 service event loop started")
    try:
        _loop.run_forever()
    finally:
        _loop.close()


def _ensure_loop():
    """Ensure the background event loop is running."""
    if _loop is None or not _loop.is_running():
        global _loop_thread
        _loop_ready.clear()
        _loop_thread = threading.Thread(target=_start_event_loop, daemon=True, name="m3-loop")
        _loop_thread.start()
        _loop_ready.wait(timeout=10)
    return _loop


def _run_async(coro, timeout: float = 120.0):
    """Run a coroutine on the persistent event loop from a sync context.

    F63 FIX: Use a single persistent loop instead of asyncio.run() per request.
    """
    loop = _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=timeout)
    except concurrent_timeout(timeout):
        future.cancel()
        return {"error": True, "error_type": "timeout", "message": f"Inference exceeded {timeout}s"}


import concurrent.futures

def concurrent_timeout(timeout):
    return concurrent.futures.TimeoutError


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # F66 FIX: enable CORS

# Request ID and timing
@app.before_request
def _before_request():
    g.request_id = str(uuid.uuid4())[:8]
    g.start_time = time.time()
    logger.info(f"[{g.request_id}] {request.method} {request.path}")


@app.after_request
def _after_request(response):
    elapsed = int((time.time() - g.get("start_time", time.time())) * 1000)
    response.headers["X-Request-ID"] = g.get("request_id", "unknown")
    response.headers["X-Response-Time-MS"] = str(elapsed)
    logger.info(
        f"[{g.get('request_id', '?')}] {request.method} {request.path} -> "
        f"{response.status_code} ({elapsed}ms)"
    )
    return response


# ---------------------------------------------------------------------------
# Config (F67: env-overridable)
# ---------------------------------------------------------------------------
def _get_config() -> Dict[str, Any]:
    return {
        "model_path": os.getenv("GEOCHAT_MODEL", "mbzuai-oryx/GeoChat"),
        "load_in_4bit": os.getenv("LOAD_IN_4BIT", "true").lower() == "true",
        "lora_adapter_path": os.getenv("LORA_ADAPTER_PATH") or None,
    }


# ---------------------------------------------------------------------------
# Service initialization (F61 FIX: no @before_first_request)
# ---------------------------------------------------------------------------
_initialized = False
_init_lock = threading.Lock()


def _init_service_once():
    """Initialize GeoChat service on first request (deprecated pattern replaced)."""
    global _initialized
    with _init_lock:
        if _initialized:
            return True
        logger.info("Initializing GeoChat service...")
        try:
            _ensure_loop()
            future = asyncio.run_coroutine_threadsafe(
                initialize_service(_get_config()), _loop
            )
            ok = future.result(timeout=600)  # up to 10 min for first model load
            _initialized = ok
            if ok:
                logger.info("GeoChat service initialized successfully")
            else:
                logger.error("GeoChat service initialization failed")
            return ok
        except Exception as e:
            logger.exception(f"Service init exception: {e}")
            _initialized = False
            return False


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": True, "error_type": "not_found", "message": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    logger.exception(f"Internal error: {e}")
    return jsonify({"error": True, "error_type": "internal_error", "message": str(e)}), 500


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.route("/health")
def health():
    """F65 FIX: comprehensive health check."""
    service = get_service()
    return jsonify({
        "status": "ok" if service.is_ready() else "loading",
        "model_loaded": service.is_ready(),
        "model_path": service._model_path,
        "adapter_path": service.get_adapter_path(),
        "mode": service.get_mode(),
        "gpu": service.get_gpu_memory(),
        "last_error": service.get_last_error(),
        "inference_count": service._inference_count,
    })


@app.route("/ready")
def ready():
    """Readiness probe — returns 200 only when model is fully loaded."""
    service = get_service()
    if service.is_ready():
        return jsonify({"ready": True})
    # Try init now
    if not _initialized:
        ok = _init_service_once()
        if ok:
            return jsonify({"ready": True})
    return jsonify({"ready": False, "error": get_service().get_last_error()}), 503


@app.route("/stats")
def stats():
    """Service statistics."""
    service = get_service()
    return jsonify({
        "inference_count": service._inference_count,
        "gpu": service.get_gpu_memory(),
        "mode": service.get_mode(),
    })


@app.route("/reload", methods=["POST"])
def reload():
    """Reload model (e.g. with new LoRA adapter)."""
    global _initialized
    data = request.json or {}
    new_adapter = data.get("lora_adapter_path")
    logger.info(f"Reloading service with adapter={new_adapter}")
    try:
        _ensure_loop()
        future = asyncio.run_coroutine_threadsafe(
            initialize_service(_get_config(), lora_adapter_path=new_adapter), _loop
        )
        ok = future.result(timeout=300)
        _initialized = ok
        return jsonify({"reloaded": ok, "adapter": new_adapter})
    except Exception as e:
        logger.exception(f"Reload failed: {e}")
        return jsonify({"reloaded": False, "error": str(e)}), 500


@app.route("/vqa", methods=["POST"])
def vqa_endpoint():
    """Visual Question Answering.

    Request:  {"image_path": "...", "question": "..."}
    Response: {"answer": "...", "error": false, "metadata": {...}}
    """
    if not _initialized and not _init_service_once():
        return jsonify({"error": True, "error_type": "model_unavailable",
                        "message": "Model not initialized"}), 503

    data = request.json or {}
    image_path = data.get("image_path")
    question = data.get("question")

    if not image_path or not question:
        return jsonify({
            "error": True, "error_type": "invalid_input",
            "message": "Missing image_path or question"
        }), 400

    try:
        result = _run_async(run_vqa(image_path, question), timeout=120.0)
        return jsonify(result)
    except Exception as e:
        logger.exception(f"/vqa error: {e}")
        return jsonify({"error": True, "error_type": "internal_error",
                        "message": str(e)}), 500


@app.route("/caption", methods=["POST"])
def caption_endpoint():
    """Image Captioning.

    Request:  {"image_path": "..."}
    Response: {"caption": "...", "error": false, "metadata": {...}}
    """
    if not _initialized and not _init_service_once():
        return jsonify({"error": True, "error_type": "model_unavailable",
                        "message": "Model not initialized"}), 503

    data = request.json or {}
    image_path = data.get("image_path")
    if not image_path:
        return jsonify({
            "error": True, "error_type": "invalid_input",
            "message": "Missing image_path"
        }), 400

    try:
        result = _run_async(run_caption(image_path), timeout=120.0)
        return jsonify(result)
    except Exception as e:
        logger.exception(f"/caption error: {e}")
        return jsonify({"error": True, "error_type": "internal_error",
                        "message": str(e)}), 500


@app.route("/grounding", methods=["POST"])
def grounding_endpoint():
    """Referring Expression Grounding.

    Request:  {"image_path": "...", "expression": "..."}
    Response: {"bbox": [x1,y1,x2,y2], "confidence": 0.85, "error": false}
    """
    if not _initialized and not _init_service_once():
        return jsonify({"error": True, "error_type": "model_unavailable",
                        "message": "Model not initialized"}), 503

    data = request.json or {}
    image_path = data.get("image_path")
    expression = data.get("expression")

    if not image_path or not expression:
        return jsonify({
            "error": True, "error_type": "invalid_input",
            "message": "Missing image_path or expression"
        }), 400

    try:
        result = _run_async(run_grounding(image_path, expression), timeout=120.0)
        return jsonify(result)
    except Exception as e:
        logger.exception(f"/grounding error: {e}")
        return jsonify({"error": True, "error_type": "internal_error",
                        "message": str(e)}), 500


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
def _shutdown_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    logger.info(f"Received signal {signum}, shutting down...")
    try:
        shutdown_service()
    except Exception as e:
        logger.warning(f"Shutdown error: {e}")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import atexit

    parser = argparse.ArgumentParser(description="M3 VLM Service")
    parser.add_argument("--host", default=os.getenv("M3_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("M3_PORT", "8001")))
    parser.add_argument("--workers", type=int, default=int(os.getenv("M3_WORKERS", "1")))
    parser.add_argument("--no-preload", action="store_true",
                        help="Don't preload model (load on first request)")
    args = parser.parse_args()

    # Register signal handlers
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    # Register cleanup
    atexit.register(shutdown_service)

    # Preload model unless --no-preload
    if not args.no_preload:
        logger.info(f"Preloading GeoChat model before serving on {args.host}:{args.port}...")
        _init_service_once()

    logger.info(f"Starting M3 VLM service on {args.host}:{args.port} (workers={args.workers})")
    # F64 FIX: threaded=True to allow concurrent requests
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=False,
        use_reloader=False,
    )