"""
sql.py — SQL Query Translator for Prefilter AI.
"""

from __future__ import annotations

from typing import Any
from prefilter_ai.ir import IntermediateRepresentation
from prefilter_ai.translators.base import BaseTranslator


class SQLTranslator(BaseTranslator):
    """Translates IntermediateRepresentation filters to parameterised SQL clauses."""

    def __init__(self, table_name: str = "listings"):
        self.table_name = table_name

    def translate(self, ir: IntermediateRepresentation) -> tuple[str, dict[str, Any]]:
        """
        Translates IR to parameterised SQL.
        
        Returns:
            (sql_string, param_dict)
        """
        clauses = []
        params = {}
        param_counter = 0

        for f in ir.filters:
            field = f.field
            op = f.operator
            val = f.value

            if op == "eq":
                param_key = f"{field}_{param_counter}"
                param_counter += 1
                clauses.append(f"{field} = :{param_key}")
                params[param_key] = val
            elif op == "ne":
                param_key = f"{field}_ne_{param_counter}"
                param_counter += 1
                clauses.append(f"{field} != :{param_key}")
                params[param_key] = val
            elif op == "lt":
                param_key = f"{field}_{param_counter}"
                param_counter += 1
                clauses.append(f"{field} < :{param_key}")
                params[param_key] = val
            elif op == "lte":
                param_key = f"{field}_{param_counter}"
                param_counter += 1
                clauses.append(f"{field} <= :{param_key}")
                params[param_key] = val
            elif op == "gt":
                param_key = f"{field}_{param_counter}"
                param_counter += 1
                clauses.append(f"{field} > :{param_key}")
                params[param_key] = val
            elif op == "gte":
                param_key = f"{field}_{param_counter}"
                param_counter += 1
                clauses.append(f"{field} >= :{param_key}")
                params[param_key] = val
            elif op == "approx":
                param_low = f"{field}_low_{param_counter}"
                param_high = f"{field}_high_{param_counter}"
                param_counter += 1
                clauses.append(f"{field} BETWEEN :{param_low} AND :{param_high}")
                if isinstance(val, (int, float)):
                    params[param_low] = val * 0.85
                    params[param_high] = val * 1.15
                else:
                    params[param_low] = val
                    params[param_high] = val
            elif op == "between":
                param_low = f"{field}_low_{param_counter}"
                param_high = f"{field}_high_{param_counter}"
                param_counter += 1
                clauses.append(f"{field} BETWEEN :{param_low} AND :{param_high}")
                params[param_low] = val
                params[param_high] = f.value_hi

        where_clause = " AND ".join(clauses)
        sql = f"SELECT * FROM {self.table_name}"
        if where_clause:
            sql += f" WHERE {where_clause}"

        return sql, params
