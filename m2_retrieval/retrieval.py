"""
M2 Retrieval - Legacy Academic Retrieval Engine (backward compatibility).
"""
from typing import Any, Dict, List, Optional


class RetrievalResult:
    """Legacy retrieval result."""
    def __init__(self, doc_id: str, score: float, rank: int = 0, metadata: Optional[Dict] = None):
        self.doc_id = doc_id
        self.score = score
        self.rank = rank
        self.metadata = metadata or {}


class SATRetrievalEngine:
    """Legacy academic retrieval engine - stub for backward compat."""

    def __init__(self, config: Any = None):
        self.config = config

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Legacy retrieve - returns empty list (use SatelliteMetadataFilter instead)."""
        return []