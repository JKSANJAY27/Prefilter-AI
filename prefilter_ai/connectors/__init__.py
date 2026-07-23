"""
connectors package for Prefilter AI.
Exposes database execution connectors for SQL, MongoDB, Elasticsearch, and ChromaDB.
"""

from prefilter_ai.connectors.base import BaseConnector
from prefilter_ai.connectors.sql import SQLConnector
from prefilter_ai.connectors.mongodb import MongoConnector
from prefilter_ai.connectors.elasticsearch import ElasticsearchConnector
from prefilter_ai.connectors.chromadb import ChromaDBConnector

__all__ = [
    "BaseConnector",
    "SQLConnector",
    "MongoConnector",
    "ElasticsearchConnector",
    "ChromaDBConnector",
]
