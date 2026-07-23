"""
chromadb.py — ChromaDB Query Translator for Prefilter AI.
"""

from __future__ import annotations

from typing import Any

from prefilter_ai.ir import IntermediateRepresentation
from prefilter_ai.translators.base import BaseTranslator


class ChromaDBTranslator(BaseTranslator):
    """Translates IntermediateRepresentation filters to a ChromaDB metadata query dictionary."""

    def translate(self, ir: IntermediateRepresentation) -> dict[str, Any]:
        clauses = []

        for f in ir.filters:
            field = f.field
            op = f.operator
            val = f.value

            if op == "eq":
                clauses.append({field: {"$eq": val}})
            elif op == "ne":
                clauses.append({field: {"$ne": val}})
            elif op == "lt":
                clauses.append({field: {"$lt": val}})
            elif op == "lte":
                clauses.append({field: {"$lte": val}})
            elif op == "gt":
                clauses.append({field: {"$gt": val}})
            elif op == "gte":
                clauses.append({field: {"$gte": val}})
            elif op == "approx":
                if isinstance(val, (int, float)):
                    clauses.append({field: {"$gte": val * 0.85}})
                    clauses.append({field: {"$lte": val * 1.15}})
                else:
                    clauses.append({field: {"$eq": val}})
            elif op == "between":
                clauses.append({field: {"$gte": val}})
                clauses.append({field: {"$lte": f.value_hi}})

        if not clauses:
            return {}
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}
