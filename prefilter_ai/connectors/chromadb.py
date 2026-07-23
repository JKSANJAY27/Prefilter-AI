"""
chromadb.py — ChromaDB Connector for Prefilter AI.

Executes generated ChromaDB metadata filter dictionaries against ChromaDB Collection objects.
"""

from __future__ import annotations

import logging
from typing import Any

from prefilter_ai.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class ChromaDBConnector(BaseConnector):
    """
    Executes generated ChromaDB metadata filter dictionaries against a Chroma Collection instance.
    """

    def __init__(self, collection: Any):
        self.collection = collection

    def execute(self, query_payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """
        Execute ChromaDB query or get with metadata filters.

        Parameters
        ----------
        query_payload : dict[str, Any]
            The ChromaDB where dict (from `result.chromadb`).
        kwargs:
            query_texts: list[str] | None = None
            n_results: int = 10
        """
        query_texts = kwargs.get("query_texts", None)
        n_results = kwargs.get("n_results", 10)

        logger.info("Executing ChromaDB query with where: %s", query_payload)

        if query_texts:
            return self.collection.query(
                query_texts=query_texts,
                where=query_payload if query_payload else None,
                n_results=n_results,
            )
        else:
            return self.collection.get(
                where=query_payload if query_payload else None,
                limit=n_results,
            )
