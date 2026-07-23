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
    _ir: Any = field(default=None, repr=False)

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

    def _get_or_create_ir(self) -> Any:
        from prefilter_ai.ir import IntermediateRepresentation
        if self._ir is not None:
            return self._ir

        # Reconstruct IR from self.fields for legacy compatibility
        domain = self.fields.get("domain", "general")
        ir = IntermediateRepresentation(domain=domain, intent="search")
        for k, v in self.fields.items():
            if k == "domain":
                continue
            if isinstance(v, list):
                for item in v:
                    op, val, val_hi = self._split_legacy_operator(item)
                    ir.add_filter(k, op, val, val_hi)
            else:
                op, val, val_hi = self._split_legacy_operator(v)
                ir.add_filter(k, op, val, val_hi)
        self._ir = ir
        return ir

    def _split_legacy_operator(self, value: Any) -> tuple[str, Any, Any | None]:
        import re
        if not isinstance(value, str):
            return "eq", value, None

        m = re.match(r"^(lt|lte|gt|gte|eq|ne|approx|between):(.+)$", value)
        if not m:
            return "eq", value, None

        op, rest = m.group(1), m.group(2)

        def _num(s):
            try:
                return float(s.replace(",", ""))
            except ValueError:
                return s

        if op == "between":
            parts = rest.split(":")
            if len(parts) == 2:
                return op, _num(parts[0]), _num(parts[1])
            return "eq", value, None

        return op, _num(rest), None

    def to_sql(self, table_name: str = "listings") -> tuple[str, dict[str, Any]]:
        """
        Convert the fields into a SQL WHERE clause and parameter dictionary.
        """
        from prefilter_ai.translators.sql import SQLTranslator
        ir = self._get_or_create_ir()
        return SQLTranslator(table_name=table_name).translate(ir)

    def to_mongodb(self) -> dict[str, Any]:
        """Convert the fields into a MongoDB filter dictionary."""
        from prefilter_ai.translators.mongodb import MongoDBTranslator
        ir = self._get_or_create_ir()
        return MongoDBTranslator().translate(ir)

    def to_chromadb(self) -> dict[str, Any]:
        """Convert the fields into a ChromaDB where query dictionary."""
        from prefilter_ai.translators.chromadb import ChromaDBTranslator
        ir = self._get_or_create_ir()
        return ChromaDBTranslator().translate(ir)

    def to_elasticsearch(self) -> dict[str, Any]:
        """Convert the fields into an Elasticsearch query DSL."""
        from prefilter_ai.translators.elasticsearch import ElasticsearchTranslator
        ir = self._get_or_create_ir()
        return ElasticsearchTranslator().translate(ir)

    def execute(self, connector: Any, **kwargs: Any) -> Any:
        """
        Execute the query against a database connector.

        Parameters
        ----------
        connector : BaseConnector
            Instance of SQLConnector, MongoConnector, ElasticsearchConnector, or ChromaDBConnector.
        """
        from prefilter_ai.connectors import (
            SQLConnector,
            MongoConnector,
            ElasticsearchConnector,
            ChromaDBConnector,
        )

        if isinstance(connector, SQLConnector):
            return connector.execute(self.to_sql(), **kwargs)
        elif isinstance(connector, MongoConnector):
            return connector.execute(self.to_mongodb(), **kwargs)
        elif isinstance(connector, ElasticsearchConnector):
            return connector.execute(self.to_elasticsearch(), **kwargs)
        elif isinstance(connector, ChromaDBConnector):
            return connector.execute(self.to_chromadb(), **kwargs)
        elif hasattr(connector, "execute"):
            return connector.execute(self, **kwargs)
        else:
            raise ValueError(f"Unsupported connector type: {type(connector)}")

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
