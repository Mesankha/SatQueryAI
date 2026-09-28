"""
M2 Configuration - Hyperparameters and weights for the retrieval pipeline.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RetrievalConfig:
    """Legacy academic config - preserved for backward compatibility."""
    top_k: int = 10
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    embedding_dim: int = 512
    score_weights: dict = field(default_factory=lambda: {
        "semantic": 0.5,
        "bm25": 0.3,
        "metadata": 0.2
    })


@dataclass
class SatelliteScoreWeights:
    """Stage 3 multi-factor score fusion weights.

    S_composite = w_geo*S_geo + w_temp*S_temporal + w_sem*S_sem + w_mod*S_modality
    Default: w_geo=0.30, w_temp=0.25, w_sem=0.30, w_mod=0.15
    """
    w_geo: float = 0.30
    w_temporal: float = 0.25
    w_semantic: float = 0.30
    w_modality: float = 0.15

    def validate(self):
        total = self.w_geo + self.w_temporal + self.w_semantic + self.w_modality
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return True