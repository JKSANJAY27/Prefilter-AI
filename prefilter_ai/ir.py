"""
ir.py — Intermediate Representation (IR) contract for Prefilter AI.

Acts as the standard unified contract between parsers, ontology, validators,
relaxers, and backend translators. Tracks confidence and provenance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence
import yaml


@dataclass
class IRFilterConstraint:
    """Represents a single query constraint with confidence and provenance."""

    field: str
    operator: str  # 'lt', 'lte', 'gt', 'gte', 'eq', 'ne', 'approx', 'between'
    value: Any
    value_hi: Any | None = None
    confidence: float = 1.0
    provenance: str = "explicit"

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
            "value_hi": self.value_hi,
            "confidence": self.confidence,
            "provenance": self.provenance,
        }


@dataclass
class IRPreference:
    """Represents a soft preference or recommendation for ranking."""

    field: str
    value: Any
    weight: float = 0.5
    confidence: float = 1.0
    provenance: str = "implicit"

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value,
            "weight": self.weight,
            "confidence": self.confidence,
            "provenance": self.provenance,
        }


@dataclass
class IntermediateRepresentation:
    """The complete query understanding intermediate representation state."""

    domain: str = "general"
    intent: str = "search"
    filters: list[IRFilterConstraint] = field(default_factory=list)
    preferences: list[IRPreference] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_filter(
        self,
        field_name: str,
        operator: str,
        value: Any,
        value_hi: Any | None = None,
        confidence: float = 1.0,
        provenance: str = "explicit",
    ) -> IntermediateRepresentation:
        # Prevent duplicates
        for f in self.filters:
            if f.field == field_name and f.operator == operator and f.value == value:
                return self
        self.filters.append(
            IRFilterConstraint(
                field=field_name,
                operator=operator,
                value=value,
                value_hi=value_hi,
                confidence=confidence,
                provenance=provenance,
            )
        )
        return self

    def add_preference(
        self,
        field_name: str,
        value: Any,
        weight: float = 0.5,
        confidence: float = 1.0,
        provenance: str = "implicit",
    ) -> IntermediateRepresentation:
        for p in self.preferences:
            if p.field == field_name and p.value == value:
                return self
        self.preferences.append(
            IRPreference(
                field=field_name,
                value=value,
                weight=weight,
                confidence=confidence,
                provenance=provenance,
            )
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "intent": self.intent,
            "filters": [f.to_dict() for f in self.filters],
            "preferences": [p.to_dict() for p in self.preferences],
            "conflicts": self.conflicts,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)

    # ------------------------------------------------------------------
    # Legacy output representation wrapper to keep backward compatibility
    # ------------------------------------------------------------------

    @property
    def legacy_fields(self) -> dict[str, Any]:
        """Convert IR filters to the legacy format returned by ParseResult.fields."""
        out = {"domain": self.domain}
        # Explicit/implicit filters
        for f in self.filters:
            if f.operator == "eq":
                out[f.field] = f.value
            elif f.operator == "ne":
                # handle list of ne: operators
                if f.field in out:
                    if isinstance(out[f.field], list):
                        out[f.field].append(f"ne:{f.value}")
                    else:
                        out[f.field] = [out[f.field], f"ne:{f.value}"]
                else:
                    out[f.field] = [f"ne:{f.value}"]
            elif f.operator == "between":
                out[f.field] = f"{f.operator}:{f.value}:{f.value_hi}"
            else:
                out[f.field] = f"{f.operator}:{f.value}"
        return out
