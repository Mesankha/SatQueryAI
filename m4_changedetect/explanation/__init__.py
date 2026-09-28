"""
Explanation package re-exports for M4 Change Detection.
"""

from m4_changedetect.explanation.templates import generate_change_description
from m4_changedetect.explanation.llm_polish import polish_change_description, validate_llm_guardrails

__all__ = [
    "generate_change_description",
    "polish_change_description",
    "validate_llm_guardrails",
]
