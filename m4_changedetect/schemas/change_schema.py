"""
M4 Module Schema Compatibility Layer.

IMPORTANT ARCHITECTURAL RULE:
This module does NOT define competing or duplicate schemas.
It imports and re-exports the canonical ChangeResult from the shared schemas registry.
"""

from shared.schemas import ChangeResult

__all__ = [
    "ChangeResult",
]
