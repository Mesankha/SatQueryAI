"""
M4 Schemas package re-export layer.
Re-exports canonical schemas imported from the shared schemas registry.
"""

from shared.schemas import ChangeResult

__all__ = [
    "ChangeResult",
]
