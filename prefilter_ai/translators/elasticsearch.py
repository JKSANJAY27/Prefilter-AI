"""
elasticsearch.py — Elasticsearch DSL Query Translator for Prefilter AI.
"""

from __future__ import annotations

from typing import Any
from prefilter_ai.ir import IntermediateRepresentation
from prefilter_ai.translators.base import BaseTranslator


class ElasticsearchTranslator(BaseTranslator):
    """Translates IntermediateRepresentation filters & preferences to Elasticsearch DSL."""

    def translate(self, ir: IntermediateRepresentation) -> dict[str, Any]:
        """
        Translates IR to Elasticsearch bool query DSL.
        
        Maps:
          - explicit/implicit filters -> `must` (requirements) or `must_not` (exclusions)
          - soft preferences -> `should` (ranking signals)
        """
        must_clauses = []
        must_not_clauses = []
        should_clauses = []

        # 1. Translate filters
        for f in ir.filters:
            field = f.field
            op = f.operator
            val = f.value

            if op == "eq":
                must_clauses.append({"term": {field: val}})
            elif op == "ne":
                must_not_clauses.append({"term": {field: val}})
            elif op == "lt":
                must_clauses.append({"range": {field: {"lt": val}}})
            elif op == "lte":
                must_clauses.append({"range": {field: {"lte": val}}})
            elif op == "gt":
                must_clauses.append({"range": {field: {"gt": val}}})
            elif op == "gte":
                must_clauses.append({"range": {field: {"gte": val}}})
            elif op == "approx":
                if isinstance(val, (int, float)):
                    must_clauses.append({"range": {field: {"gte": val * 0.85, "lte": val * 1.15}}})
                else:
                    must_clauses.append({"term": {field: val}})
            elif op == "between":
                must_clauses.append({"range": {field: {"gte": val, "lte": f.value_hi}}})

        # 2. Translate soft preferences
        for p in ir.preferences:
            should_clauses.append({
                "term": {
                    p.field: {
                        "value": p.value,
                        "boost": p.weight * 2.0  # boost based on preference weight
                    }
                }
            })

        # Assemble bool query
        bool_query: dict[str, Any] = {}
        if must_clauses:
            bool_query["must"] = must_clauses
        if must_not_clauses:
            bool_query["must_not"] = must_not_clauses
        if should_clauses:
            bool_query["should"] = should_clauses

        return {"query": {"bool": bool_query}}
