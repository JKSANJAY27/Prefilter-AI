"""
mongodb.py — MongoDB Connector for Prefilter AI.

Executes generated MongoDB filter dicts against PyMongo Collection objects.
"""

from __future__ import annotations

import logging
from typing import Any
from prefilter_ai.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class MongoConnector(BaseConnector):
    """
    Executes generated MongoDB filter dicts against PyMongo Collection instances.
    """

    def __init__(self, collection: Any):
        self.collection = collection

    def execute(self, query_payload: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        """
        Execute MongoDB find query.

        Parameters
        ----------
        query_payload : dict[str, Any]
            The MongoDB filter dictionary (from `result.mongodb`).
        kwargs:
            limit: int = 100
            projection: dict | None = None
        """
        limit = kwargs.get("limit", 100)
        projection = kwargs.get("projection", None)

        logger.info("Executing Mongo find query: %s", query_payload)

        cursor = self.collection.find(query_payload, projection)
        if limit:
            cursor = cursor.limit(limit)

        return list(cursor)
