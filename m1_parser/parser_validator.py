"""
M1 entrypoint: LLM primary with deterministic fallback.
Guarantees: always returns schema-valid StructuredQuery, never raises.
"""
from shared.schemas import StructuredQuery
from shared.logger import get_logger

# Graceful import for optional LLM parser
try:
    from .parser_llm import parse_llm
except ImportError:
    parse_llm = None

from .parser_fallback import parse_fallback

logger = get_logger(__name__)


def parse_query(query_text: str) -> StructuredQuery:
    """
    Primary: Claude LLM (5s timeout).
    On failure: deterministic fallback (<10ms).
    """
    # Attempt 1: LLM (only if available)
    if parse_llm is not None:
        result = parse_llm(query_text, timeout_seconds=5.0)
        if result is not None:
            logger.info("M1: LLM parse succeeded", extra={"module_name": "M1", "query": query_text})
            return result

    # Attempt 2 (only for validation errors): retry with hint — handled inside parse_llm via exception.
    # If we reach here, LLM failed entirely.

    logger.warning(
        "M1: LLM parse failed, using deterministic fallback",
        extra={"module_name": "M1", "query": query_text},
    )
    return parse_fallback(query_text)
