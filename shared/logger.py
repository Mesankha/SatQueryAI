"""Structured JSON logger."""
import json
import logging
import sys
from datetime import datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "module": getattr(record, "module_name", "unknown"),
            "trace_id": getattr(record, "trace_id", None),
            "message": record.getMessage(),
        }
        # Add any extra fields
        for key in ["latency_ms", "error_type", "fallback_used"]:
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)
        return json.dumps(log_obj, default=str)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
