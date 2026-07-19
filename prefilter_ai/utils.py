"""
utils.py — Shared utility functions for Prefilter AI.

Centralises logic that was previously duplicated across SpacyParser,
SLMParser, and GeminiParser, and adds real confidence scoring.
"""

from __future__ import annotations

from typing import Any


# ── Operator splitting ─────────────────────────────────────────────────


def split_operator_value(value: Any) -> tuple[str, Any, Any | None]:
    """
    Parse an operator-prefixed constraint string into (operator, value, value_hi).

    Handles:
      "lt:200"          → ("lt", 200.0, None)
      "gte:4.5"         → ("gte", 4.5, None)
      "between:100:300" → ("between", 100.0, 300.0)
      "ne:red"          → ("ne", "red", None)
      "eq:Sony"         → ("eq", "Sony", None)
      True / 5 / etc.   → ("eq", value, None)   (passthrough for non-strings)

    Returns
    -------
    (operator, value, value_hi)
    """
    if not isinstance(value, str):
        return "eq", value, None

    if ":" not in value:
        return "eq", value, None

    parts = value.split(":")
    op = parts[0].lower()

    _NUMERIC_OPS = {"lt", "lte", "gt", "gte", "approx", "ne"}

    def _try_num(s: str) -> Any:
        try:
            return float(s.replace(",", ""))
        except ValueError:
            return s

    if op == "between" and len(parts) == 3:
        lo = _try_num(parts[1])
        hi = _try_num(parts[2])
        return "between", lo, hi

    if op in _NUMERIC_OPS and len(parts) >= 2:
        val = _try_num(":".join(parts[1:]))  # re-join in case value itself has ':'
        return op, val, None

    if op == "eq" and len(parts) >= 2:
        return "eq", ":".join(parts[1:]), None

    # Unknown prefix — treat whole string as an eq value
    return "eq", value, None


# ── Confidence scoring ─────────────────────────────────────────────────


def confidence_from_extraction(
    *,
    entity_type: str = "",
    pattern_matched: bool = False,
    is_numeric: bool = False,
    query_position: float = 0.5,  # 0.0 = start, 1.0 = end
    backend: str = "spacy",
) -> float:
    """
    Compute a real confidence score instead of a hardcoded constant.

    Parameters
    ----------
    entity_type : str
        spaCy entity label (e.g. "ORG", "GPE", "MONEY", "CARDINAL").
    pattern_matched : bool
        Whether the value was extracted via an explicit regex pattern.
    is_numeric : bool
        Whether the extracted constraint is numeric (operators like lt/gte).
    query_position : float
        Relative position of the match in the query (0.0 start, 1.0 end).
    backend : str
        Which backend produced the extraction.

    Returns
    -------
    float in [0.0, 1.0]
    """
    base = 0.70  # start from a reasonable floor

    # Backend bonus
    if backend == "gemini":
        base = 0.92
    elif backend == "slm":
        base = 0.88
    elif backend == "spacy":
        base = 0.75

    # Numeric constraints are generally more reliable than string extractions
    if is_numeric:
        base = min(base + 0.08, 0.99)

    # Explicit pattern match (regex) is more reliable than NER alone
    if pattern_matched:
        base = min(base + 0.06, 0.99)

    # High-confidence entity types
    if entity_type in {"MONEY", "CARDINAL", "ORDINAL"}:
        base = min(base + 0.07, 0.99)
    elif entity_type in {"ORG", "GPE", "PRODUCT"}:
        base = min(base + 0.04, 0.99)
    elif entity_type in {"DATE", "TIME"}:
        base = min(base + 0.03, 0.99)

    # Slight boost for earlier-position mentions (users often state key constraints first)
    if query_position < 0.3:
        base = min(base + 0.02, 0.99)

    return round(base, 3)


# ── Explanation builder ────────────────────────────────────────────────


def build_explanation(ir: Any) -> dict[str, str]:
    """
    Build a human-readable explanation dict for every IR filter.
    Maps field_name → explanation string.
    """
    explanations: dict[str, str] = {}
    for f in ir.filters:
        prov = f.provenance
        op_str = _op_to_english(f.operator, f.value, f.value_hi)
        explanations[f.field] = (
            f"'{f.field}' {op_str} — extracted via {prov} "
            f"(confidence: {f.confidence:.0%})"
        )
    for p in ir.preferences:
        explanations[f"pref:{p.field}"] = (
            f"Soft preference: '{p.field}' preferred as '{p.value}' "
            f"(weight: {p.weight:.0%}, from {p.provenance})"
        )
    return explanations


def _op_to_english(op: str, value: Any, value_hi: Any | None) -> str:
    mapping = {
        "lt":  f"must be less than {value}",
        "lte": f"must be at most {value}",
        "gt":  f"must be greater than {value}",
        "gte": f"must be at least {value}",
        "approx": f"approximately {value}",
        "ne":  f"must NOT be '{value}'",
        "eq":  f"= '{value}'",
        "between": f"between {value} and {value_hi}",
    }
    return mapping.get(op, f"{op}: {value}")
