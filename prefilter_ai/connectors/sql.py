"""
sql.py — SQL Database Connector for Prefilter AI.

Executes generated SQL queries against standard DBAPI connections or SQLAlchemy engines.
"""

from __future__ import annotations

import logging
from typing import Any
from prefilter_ai.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class SQLConnector(BaseConnector):
    """
    Executes generated SQL query tuples `(where_clause, params)` against a SQL connection or engine.

    Supports:
    - Standard DB-API 2.0 connections (sqlite3, psycopg2, pymysql)
    - SQLAlchemy Connection / Engine objects
    """

    def __init__(self, connection: Any, table_name: str = "listings"):
        self.connection = connection
        self.table_name = table_name

    def execute(self, query_payload: tuple[str, dict[str, Any]] | str, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Execute SQL query payload.

        Parameters
        ----------
        query_payload : tuple[str, dict[str, Any]] | str
            Either (where_clause, params) tuple or raw SQL WHERE clause string.
        kwargs:
            select_fields: str = "*"
            limit: int | None = 100
        """
        if isinstance(query_payload, tuple):
            where_clause, params = query_payload
        else:
            where_clause, params = query_payload, {}

        select_fields = kwargs.get("select_fields", "*")
        limit = kwargs.get("limit", 100)

        query_str = where_clause.strip()
        if query_str.upper().startswith("SELECT"):
            sql = query_str
        else:
            sql = f"SELECT {select_fields} FROM {self.table_name}"
            if query_str:
                sql += f" WHERE {query_str}"
            if limit is not None and "LIMIT" not in sql.upper():
                sql += f" LIMIT {limit}"

        logger.info("Executing SQL Query: %s | Params: %s", sql, params)

        # 1. Try SQLAlchemy connection / engine
        if hasattr(self.connection, "execute") and hasattr(self.connection, "connect"):
            with self.connection.connect() as conn:
                from sqlalchemy import text
                result = conn.execute(text(sql), params)
                keys = result.keys()
                return [dict(zip(keys, row)) for row in result.fetchall()]
        elif hasattr(self.connection, "execute"):
            # Direct connection object
            cursor = self.connection.cursor() if hasattr(self.connection, "cursor") else self.connection
            cursor.execute(sql, params)
            if hasattr(cursor, "description") and cursor.description:
                cols = [desc[0] for desc in cursor.description]
                return [dict(zip(cols, row)) for row in cursor.fetchall()]
            return cursor.fetchall()
        else:
            raise ValueError(f"Unsupported SQL connection object: {type(self.connection)}")
