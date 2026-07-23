"""
mongodb.py — MongoDB Query Translator for Prefilter AI.
"""

from __future__ import annotations

from typing import Any

from prefilter_ai.ir import IntermediateRepresentation
from prefilter_ai.translators.base import BaseTranslator


class MongoDBTranslator(BaseTranslator):
    """Translates IntermediateRepresentation filters to a MongoDB query document."""

    def translate(self, ir: IntermediateRepresentation) -> dict[str, Any]:
        mongo_filter: dict[str, Any] = {}

        # Track multiple ne / range constraints on the same field
        field_ops: dict[str, dict[str, Any]] = {}

        for f in ir.filters:
            field = f.field
            op = f.operator
            val = f.value

            if op == "eq":
                mongo_filter[field] = val
            elif op == "ne":
                if field not in field_ops:
                    field_ops[field] = {}
                # Handle $nin if there are multiple exclusions
                if "$ne" in field_ops[field]:
                    existing = field_ops[field]["$ne"]
                    field_ops[field].pop("$ne")
                    field_ops[field]["$nin"] = [existing, val]
                elif "$nin" in field_ops[field]:
                    field_ops[field]["$nin"].append(val)
                else:
                    field_ops[field]["$ne"] = val
            elif op == "lt":
                if field not in field_ops:
                    field_ops[field] = {}
                field_ops[field]["$lt"] = val
            elif op == "lte":
                if field not in field_ops:
                    field_ops[field] = {}
                field_ops[field]["$lte"] = val
            elif op == "gt":
                if field not in field_ops:
                    field_ops[field] = {}
                field_ops[field]["$gt"] = val
            elif op == "gte":
                if field not in field_ops:
                    field_ops[field] = {}
                field_ops[field]["$gte"] = val
            elif op == "approx":
                if isinstance(val, (int, float)):
                    if field not in field_ops:
                        field_ops[field] = {}
                    field_ops[field]["$gte"] = val * 0.85
                    field_ops[field]["$lte"] = val * 1.15
                else:
                    mongo_filter[field] = val
            elif op == "between":
                if field not in field_ops:
                    field_ops[field] = {}
                field_ops[field]["$gte"] = val
                field_ops[field]["$lte"] = f.value_hi

        # Merge field operations back to mongo_filter
        for field, ops in field_ops.items():
            if field in mongo_filter:
                # If already an eq constraint, keep it, but log warning or merge
                pass
            else:
                mongo_filter[field] = ops

        return mongo_filter
