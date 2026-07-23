"""
base.py — Base interface for Prefilter AI Database Connectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseConnector(ABC):
    """Abstract Base Class for executing translated queries against target databases."""

    @abstractmethod
    def execute(self, query_payload: Any, **kwargs: Any) -> Any:
        """
        Execute a translated query payload against the connected database.

        Parameters
        ----------
        query_payload : Any
            The translated output (e.g. SQL string + params, MongoDB dict, ES DSL, Chroma dict).
        **kwargs : Any
            Additional database query options (e.g., limit, offset, collection_name).

        Returns
        -------
        Any
            Database execution results.
        """
