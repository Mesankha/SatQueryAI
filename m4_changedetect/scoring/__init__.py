"""
Scoring package re-exports for M4 Change Detection.
"""

from m4_changedetect.scoring.change_score import normalize_change_map, calculate_change_score

__all__ = [
    "normalize_change_map",
    "calculate_change_score",
]
