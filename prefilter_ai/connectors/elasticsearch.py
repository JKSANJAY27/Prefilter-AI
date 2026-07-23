"""
elasticsearch.py — Elasticsearch Connector for Prefilter AI.

Executes generated Elasticsearch DSL queries against official Elasticsearch python client instances.
"""

from __future__ import annotations

import logging
from typing import Any

from prefilter_ai.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class ElasticsearchConnector(BaseConnector):
    """
    Executes generated Elasticsearch DSL queries against an Elasticsearch client.
    """

    def __init__(self, client: Any, index: str = "search_index"):
        self.client = client
        self.index = index

    def execute(self, query_payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """
        Execute Elasticsearch query.

        Parameters
        ----------
        query_payload : dict[str, Any]
            The Elasticsearch DSL dict (from `result.elasticsearch`).
        kwargs:
            size: int = 100
        """
        size = kwargs.get("size", 100)
        index = kwargs.get("index", self.index)

        logger.info("Executing ES search on index '%s': %s", index, query_payload)

        return self.client.search(index=index, query=query_payload, size=size)
