"""
base.py — Abstract Base Class for Query Translators in Prefilter AI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from prefilter_ai.ir import IntermediateRepresentation


class BaseTranslator(ABC):
    """Abstract base class for all backend translators."""

    @abstractmethod
    def translate(self, ir: IntermediateRepresentation) -> Any:
        """Translate IntermediateRepresentation to database specific queries."""
        pass
