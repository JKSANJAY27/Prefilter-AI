"""
test_connectors.py — Unit tests for database connectors and execution wrappers.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest

from prefilter_ai import PrefilterPipeline
from prefilter_ai.connectors import (
    ChromaDBConnector,
    ElasticsearchConnector,
    MongoConnector,
    SQLConnector,
)


@pytest.fixture
def pipeline():
    return PrefilterPipeline(parser="spacy")


@pytest.fixture
def sqlite_conn():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE listings (id INT, product TEXT, price REAL, brand TEXT)")
    cursor.execute("INSERT INTO listings VALUES (1, 'laptop', 750.0, 'Lenovo')")
    cursor.execute("INSERT INTO listings VALUES (2, 'laptop', 1200.0, 'Apple')")
    cursor.execute("INSERT INTO listings VALUES (3, 'headphones', 150.0, 'Sony')")
    conn.commit()
    yield conn
    conn.close()


def test_sql_connector_execution(pipeline, sqlite_conn):
    res = pipeline.run("laptop under $800")
    connector = SQLConnector(connection=sqlite_conn, table_name="listings")

    # Direct execution via connector
    rows = connector.execute(res.sql)
    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0]["product"] == "laptop"
    assert rows[0]["price"] == 750.0

    # Execution via PipelineResult wrapper
    rows_wrapper = res.execute(connector)
    assert len(rows_wrapper) == 1
    assert rows_wrapper[0]["brand"] == "Lenovo"


def test_mongo_connector_execution(pipeline):
    res = pipeline.run("headphones under $200")
    mock_collection = MagicMock()
    mock_collection.find.return_value.limit.return_value = [
        {"_id": "1", "product": "headphones", "price": 150}
    ]

    connector = MongoConnector(collection=mock_collection)
    results = res.execute(connector, limit=10)

    assert len(results) == 1
    assert results[0]["product"] == "headphones"
    mock_collection.find.assert_called_once()


def test_elasticsearch_connector_execution(pipeline):
    res = pipeline.run("gaming laptop")
    mock_es = MagicMock()
    mock_es.search.return_value = {"hits": {"total": {"value": 1}, "hits": []}}

    connector = ElasticsearchConnector(client=mock_es, index="products")
    out = res.execute(connector)

    assert "hits" in out
    mock_es.search.assert_called_once()


def test_chromadb_connector_execution(pipeline):
    res = pipeline.run("laptop")
    mock_chroma = MagicMock()
    mock_chroma.get.return_value = {"ids": ["doc1"], "metadatas": [{"product": "laptop"}]}

    connector = ChromaDBConnector(collection=mock_chroma)
    out = res.execute(connector, n_results=5)

    assert "ids" in out
    mock_chroma.get.assert_called_once()
