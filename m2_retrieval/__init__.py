"""
M2 Retrieval Module for SAT Query AI System.
"""

# ---------------------------------------------------------------------
# Legacy Academic Compatibility Imports (Preserved for Rule 6 / Rule 8 tests)
# ---------------------------------------------------------------------
from m2_retrieval.config import RetrievalConfig
from m2_retrieval.metadata_filter import QuestionMetadata, FilterCriteria, MetadataFilterEngine
from m2_retrieval.embedding_index import EmbeddingIndex, QuestionItem
from m2_retrieval.scoring import CompositeScorer, BM25Scorer
from m2_retrieval.retrieval import SATRetrievalEngine, RetrievalResult

# ---------------------------------------------------------------------
# Authoritative Satellite M2 Pipeline Imports
# ---------------------------------------------------------------------
from m2_retrieval.metadata_filter import SatelliteMetadataFilter
from m2_retrieval.config import SatelliteScoreWeights
from m2_retrieval.scoring import SatelliteScoreEngine
from m2_retrieval.satellite_index import (
    normalize_l2,
    StructuredQuery,
    SceneMetadata,
    BaseSatelliteEncoder,
    RemoteCLIPEncoder,
    GeoRSCLIPEncoder,
    FAISSIndexFlatIPAdapter,
    SemanticEncoderPipeline,
    SatelliteSemanticRanker
)

__version__ = "1.0.0"

__all__ = [
    # Authoritative Satellite M2 Exports
    "StructuredQuery",
    "SceneMetadata",
    "SatelliteMetadataFilter",
    "SatelliteSemanticRanker",
    "SemanticEncoderPipeline",
    "SatelliteScoreWeights",
    "SatelliteScoreEngine",
    "RemoteCLIPEncoder",
    "GeoRSCLIPEncoder",
    "FAISSIndexFlatIPAdapter",
    "normalize_l2",
    "BaseSatelliteEncoder",

    # Legacy Academic Compatibility Exports
    "RetrievalConfig",
    "QuestionMetadata",
    "FilterCriteria",
    "MetadataFilterEngine",
    "EmbeddingIndex",
    "QuestionItem",
    "CompositeScorer",
    "BM25Scorer",
    "SATRetrievalEngine",
    "RetrievalResult",
    "__version__",
]