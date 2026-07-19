"""
ParseResult: the structured output returned by PrefilterAI.parse().
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class ParseResult:
    """
    The result of parsing a natural language search query.

    Attributes
    ----------
    query : str
        The original user query, unchanged.
    fields : dict[str, Any]
        Extracted structured fields.  Numeric constraints are expressed
        with operator prefixes, e.g. ``"price": "lt:2000"``.
    raw_output : str
        The raw string produced by the model before parsing.
        Useful for debugging when fields look unexpected.
    model_format : str
        Which adapter was used ("json" or "yaml").

    Examples
    --------
    >>> result.fields
    {'domain': 'ecommerce', 'product': 'MacBook', 'price': 'lt:2000'}

    >>> result.to_json()
    '{"domain": "ecommerce", "product": "MacBook", "price": "lt:2000"}'

    >>> result.to_yaml()
    'domain: ecommerce\nproduct: MacBook\nprice: lt:2000\n'

    Operator reference
    ------------------
    ``lt:N``            < N   (under, less than, below)
    ``lte:N``           ≤ N   (up to, at most, or less)
    ``gt:N``            > N   (over, more than, above)
    ``gte:N``           ≥ N   (at least, N+, starting from)
    ``approx:N``        ≈ N   (around, roughly, ~N)
    ``between:Lo:Hi``   Lo ≤ x ≤ Hi
    """

    query: str
    fields: dict[str, Any]
    raw_output: str
    model_format: str
    _extra: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_json(self, indent: int | None = None) -> str:
        """Return fields as a JSON string."""
        return json.dumps(self.fields, ensure_ascii=False, indent=indent)

    def to_yaml(self) -> str:
        """Return fields as a YAML string."""
        return yaml.dump(self.fields, default_flow_style=False, allow_unicode=True)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for JSON serialisation of the full result."""
        return {
            "query": self.query,
            "fields": self.fields,
            "model_format": self.model_format,
        }

    # ------------------------------------------------------------------
    # Query translators
    # ------------------------------------------------------------------

    def to_sql(self, table_name: str = "listings") -> tuple[str, dict[str, Any]]:
        """
        Convert the fields into a SQL WHERE clause and parameter dictionary.

        Example
        -------
        >>> query, params = result.to_sql()
        >>> query
        "SELECT * FROM listings WHERE brand = :brand AND price < :price"
        """
        clauses = []
        params = {}
        param_counter = 0

        for field_name, value in self.fields.items():
            if field_name == "domain":
                continue

            if isinstance(value, list):
                for val in value:
                    op_data = self._parse_operator_val(val)
                    op = op_data["op"]
                    if op == "ne":
                        param_key = f"{field_name}_ne_{param_counter}"
                        param_counter += 1
                        clauses.append(f"{field_name} != :{param_key}")
                        params[param_key] = op_data["val"]
                continue

            op_data = self._parse_operator_val(value)
            op = op_data["op"]

            if op == "eq":
                param_key = f"{field_name}_{param_counter}"
                param_counter += 1
                clauses.append(f"{field_name} = :{param_key}")
                params[param_key] = op_data["val"]
            elif op == "ne":
                param_key = f"{field_name}_{param_counter}"
                param_counter += 1
                clauses.append(f"{field_name} != :{param_key}")
                params[param_key] = op_data["val"]
            elif op == "lt":
                param_key = f"{field_name}_{param_counter}"
                param_counter += 1
                clauses.append(f"{field_name} < :{param_key}")
                params[param_key] = op_data["val"]
            elif op == "lte":
                param_key = f"{field_name}_{param_counter}"
                param_counter += 1
                clauses.append(f"{field_name} <= :{param_key}")
                params[param_key] = op_data["val"]
            elif op == "gt":
                param_key = f"{field_name}_{param_counter}"
                param_counter += 1
                clauses.append(f"{field_name} > :{param_key}")
                params[param_key] = op_data["val"]
            elif op == "gte":
                param_key = f"{field_name}_{param_counter}"
                param_counter += 1
                clauses.append(f"{field_name} >= :{param_key}")
                params[param_key] = op_data["val"]
            elif op == "approx":
                param_low = f"{field_name}_low_{param_counter}"
                param_high = f"{field_name}_high_{param_counter}"
                param_counter += 1
                clauses.append(f"{field_name} BETWEEN :{param_low} AND :{param_high}")
                val = op_data["val"]
                if isinstance(val, (int, float)):
                    params[param_low] = val * 0.85
                    params[param_high] = val * 1.15
                else:
                    params[param_low] = val
                    params[param_high] = val
            elif op == "between":
                param_low = f"{field_name}_low_{param_counter}"
                param_high = f"{field_name}_high_{param_counter}"
                param_counter += 1
                clauses.append(f"{field_name} BETWEEN :{param_low} AND :{param_high}")
                params[param_low] = op_data["val"]
                params[param_high] = op_data["val_hi"]

        where_clause = " AND ".join(clauses)
        sql = f"SELECT * FROM {table_name}"
        if where_clause:
            sql += f" WHERE {where_clause}"
        return sql, params

    def to_mongodb(self) -> dict[str, Any]:
        """Convert the fields into a MongoDB filter dictionary."""
        mongo_filter = {}
        for field_name, value in self.fields.items():
            if field_name == "domain":
                continue

            if isinstance(value, list):
                ne_vals = []
                for val in value:
                    op_data = self._parse_operator_val(val)
                    if op_data["op"] == "ne":
                        ne_vals.append(op_data["val"])
                if ne_vals:
                    if len(ne_vals) == 1:
                        mongo_filter[field_name] = {"$ne": ne_vals[0]}
                    else:
                        mongo_filter[field_name] = {"$nin": ne_vals}
                continue

            op_data = self._parse_operator_val(value)
            op = op_data["op"]
            val = op_data["val"]

            if op == "eq":
                mongo_filter[field_name] = val
            elif op == "ne":
                mongo_filter[field_name] = {"$ne": val}
            elif op == "lt":
                mongo_filter[field_name] = {"$lt": val}
            elif op == "lte":
                mongo_filter[field_name] = {"$lte": val}
            elif op == "gt":
                mongo_filter[field_name] = {"$gt": val}
            elif op == "gte":
                mongo_filter[field_name] = {"$gte": val}
            elif op == "approx":
                if isinstance(val, (int, float)):
                    mongo_filter[field_name] = {"$gte": val * 0.85, "$lte": val * 1.15}
                else:
                    mongo_filter[field_name] = val
            elif op == "between":
                mongo_filter[field_name] = {"$gte": val, "$lte": op_data["val_hi"]}

        return mongo_filter

    def to_chromadb(self) -> dict[str, Any]:
        """Convert the fields into a ChromaDB where query dictionary."""
        clauses = []
        for field_name, value in self.fields.items():
            if field_name == "domain":
                continue

            if isinstance(value, list):
                for val in value:
                    op_data = self._parse_operator_val(val)
                    op = op_data["op"]
                    if op == "ne":
                        clauses.append({field_name: {"$ne": op_data["val"]}})
                continue

            op_data = self._parse_operator_val(value)
            op = op_data["op"]
            val = op_data["val"]

            if op == "eq":
                clauses.append({field_name: {"$eq": val}})
            elif op == "ne":
                clauses.append({field_name: {"$ne": val}})
            elif op == "lt":
                clauses.append({field_name: {"$lt": val}})
            elif op == "lte":
                clauses.append({field_name: {"$lte": val}})
            elif op == "gt":
                clauses.append({field_name: {"$gt": val}})
            elif op == "gte":
                clauses.append({field_name: {"$gte": val}})
            elif op == "approx":
                if isinstance(val, (int, float)):
                    clauses.append({field_name: {"$gte": val * 0.85, "$lte": val * 1.15}})
                else:
                    clauses.append({field_name: {"$eq": val}})
            elif op == "between":
                clauses.append({field_name: {"$gte": val}})
                clauses.append({field_name: {"$lte": op_data["val_hi"]}})

        if not clauses:
            return {}
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _parse_operator_val(self, value: Any) -> dict[str, Any]:
        import re
        if not isinstance(value, str):
            return {"op": "eq", "val": value}

        m = re.match(r"^(lt|lte|gt|gte|eq|ne|approx|between):(.+)$", value)
        if not m:
            return {"op": "eq", "val": value}

        op, rest = m.group(1), m.group(2)

        def _num(s):
            try:
                return float(s.replace(",", ""))
            except ValueError:
                return s

        if op == "between":
            parts = rest.split(":")
            if len(parts) == 2:
                return {"op": op, "val": _num(parts[0]), "val_hi": _num(parts[1])}
            return {"op": "eq", "val": value}

        return {"op": op, "val": _num(rest)}

    # ------------------------------------------------------------------
    # Operator helpers: let callers decode operator-prefixed numerics
    # ------------------------------------------------------------------

    def get_numeric_constraint(self, field_name: str) -> dict[str, Any] | None:
        """
        Decode an operator-prefixed numeric field into a structured dict.

        Returns None if the field is missing or not operator-prefixed.

        Example
        -------
        >>> result.fields["price"]
        'lt:2000'
        >>> result.get_numeric_constraint("price")
        {'operator': 'lt', 'value': 2000.0, 'value_hi': None}
        """
        raw = self.fields.get(field_name)
        if not isinstance(raw, str) or ":" not in raw:
            return None

        parts = raw.split(":")
        op = parts[0]

        if op not in {"lt", "lte", "gt", "gte", "approx", "between"}:
            return None

        try:
            if op == "between" and len(parts) == 3:
                return {
                    "operator": op,
                    "value": float(parts[1]),
                    "value_hi": float(parts[2]),
                }
            return {
                "operator": op,
                "value": float(parts[1]),
                "value_hi": None,
            }
        except (ValueError, IndexError):
            return None

    def numeric_fields(self) -> dict[str, dict[str, Any]]:
        """
        Return all fields that carry operator-prefixed numeric constraints.

        Example
        -------
        >>> result.numeric_fields()
        {'price': {'operator': 'lt', 'value': 2000.0, 'value_hi': None},
         'rating': {'operator': 'gte', 'value': 4.5, 'value_hi': None}}
        """
        out = {}
        for key in self.fields:
            decoded = self.get_numeric_constraint(key)
            if decoded is not None:
                out[key] = decoded
        return out

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ParseResult(query={self.query!r}, "
            f"fields={self.fields!r}, "
            f"model_format={self.model_format!r})"
        )

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access: result['domain']"""
        return self.fields[key]

    def __contains__(self, key: str) -> bool:
        return key in self.fields
