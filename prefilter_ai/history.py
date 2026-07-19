"""
history.py — Stateful Session Management & Query Diff Engine for Prefilter AI.

Maintains context across sequential search queries, allowing conversational
refinements (e.g., "Only Dell" -> "cheaper" -> "actually Lenovo").

Fix applied: process_query() now re-runs OntologyEngine, ConflictDetector,
and QueryRelaxer after every merge so conversational refinements trigger
the full intelligence layer (Gap #6 fix).
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from prefilter_ai.ir import IntermediateRepresentation, IRFilterConstraint
from prefilter_ai.parser_interface import BaseParser
from prefilter_ai.ontology import OntologyEngine
from prefilter_ai.validator import ConflictDetector
from prefilter_ai.relaxer import QueryRelaxer
from prefilter_ai.registry import SchemaRegistry

logger = logging.getLogger(__name__)


class QueryDiffEngine:
    """Merges sequential refinement queries into the active query context."""

    def diff_and_merge(
        self, base_ir: IntermediateRepresentation, refinement_ir: IntermediateRepresentation
    ) -> IntermediateRepresentation:
        """
        Merge refinement constraints into the base IR state.
        
        Handles:
          - Overwriting matching fields (e.g., "Actually Lenovo" overrides "Dell")
          - Appending array filters (e.g., "Exclude black" + "Exclude pink")
          - Modifying numerical bounds (e.g., "cheaper" reduces price constraint)
        """
        merged = copy.deepcopy(base_ir)

        # Merge domain if refinement detected a specific one
        if refinement_ir.domain != "general" and refinement_ir.domain != merged.domain:
            merged.domain = refinement_ir.domain

        # If refinement says "cheaper", let's adjust price ceil
        ref_text = refinement_ir.metadata.get("query_text", "").lower()
        
        # Track which explicit filters are overwritten
        overwritten_fields = set()

        for ref_filter in refinement_ir.filters:
            field = ref_filter.field
            op = ref_filter.operator
            val = ref_filter.value

            # If brand/make changes, overwrite the old one
            if field in {"brand", "make", "model", "product", "job_title", "city"}:
                # Remove any existing eq filters for this field
                merged.filters = [f for f in merged.filters if not (f.field == field and f.operator == "eq")]
                overwritten_fields.add(field)

            # Handle price/budget relaxation/tightening
            if field == "price" and op in {"lt", "lte"}:
                # If we already have a price filter, let's keep the lowest ceiling
                existing_prices = [f for f in merged.filters if f.field == "price" and f.operator in {"lt", "lte"}]
                if existing_prices:
                    min_ceiling = min(float(existing_prices[0].value), float(val))
                    # Remove old price filters
                    merged.filters = [f for f in merged.filters if not (f.field == "price")]
                    merged.add_filter("price", "lt", min_ceiling, provenance=f"Refined (min ceiling)")
                    continue

            # Standard append or merge
            merged.add_filter(
                field_name=field,
                operator=op,
                value=val,
                value_hi=ref_filter.value_hi,
                confidence=ref_filter.confidence,
                provenance=f"Refinement: {ref_filter.provenance}",
            )

        # Apply specific keyword modifications if no explicit filters extracted
        if "cheaper" in ref_text or "less expensive" in ref_text:
            self._apply_cheaper_heuristic(merged)

        # Merge soft preferences
        for ref_pref in refinement_ir.preferences:
            # Overwrite preferences if same field
            merged.preferences = [p for p in merged.preferences if p.field != ref_pref.field]
            merged.add_preference(
                field_name=ref_pref.field,
                value=ref_pref.value,
                weight=ref_pref.weight,
                confidence=ref_pref.confidence,
                provenance=f"Refinement preference: {ref_pref.provenance}",
            )

        return merged

    def _apply_cheaper_heuristic(self, ir: IntermediateRepresentation) -> None:
        """Heuristically reduce price ceiling by 20% if 'cheaper' is requested."""
        price_filters = [f for f in ir.filters if f.field == "price" and f.operator in {"lt", "lte"}]
        if price_filters:
            f = price_filters[0]
            try:
                old_val = float(f.value)
                new_val = old_val * 0.80
                f.value = new_val
                f.provenance = f"{f.provenance} (reduced by 20% for 'cheaper')"
                ir.warnings.append(f"Applied refinement: reduced price ceiling to ${new_val:.2f}")
            except ValueError:
                pass
        else:
            # No existing price limit, set a soft preference for low price
            ir.add_preference(
                field_name="price",
                value="low",
                weight=0.85,
                confidence=0.8,
                provenance="Refinement: 'cheaper'",
            )


class PrefilterSession:
    """Manages sequential conversational search state and refinement histories."""

    def __init__(
        self,
        parser: BaseParser,
        run_pipeline: bool = True,
        auto_relax: bool = True,
    ):
        self.parser = parser
        self.engine = QueryDiffEngine()
        self.current_ir: IntermediateRepresentation | None = None
        self.history: list[IntermediateRepresentation] = []
        self.run_pipeline = run_pipeline
        self.auto_relax = auto_relax

        # Shared singleton pipeline components
        self._ontology = OntologyEngine()
        self._validator = ConflictDetector()
        self._relaxer = QueryRelaxer(registry=SchemaRegistry())

    def process_query(self, query: str) -> IntermediateRepresentation:
        """Process a query in the context of the current conversation history."""
        # 1. Parse query
        new_ir = self.parser.parse(query)
        new_ir.metadata["query_text"] = query

        # 2. Merge with session state
        if self.current_ir is None:
            merged_ir = new_ir
        else:
            merged_ir = self.engine.diff_and_merge(self.current_ir, new_ir)

        # 3. Re-run full pipeline on merged state (Gap #6 fix)
        if self.run_pipeline:
            # Clear stale conflict/warning data before re-running
            merged_ir.conflicts = []
            merged_ir.warnings = []
            merged_ir.metadata.pop("conflict_tags", None)

            merged_ir = self._ontology.infer(merged_ir, query)
            merged_ir = self._validator.validate(merged_ir)

            if self.auto_relax and merged_ir.conflicts:
                relaxed = self._relaxer.relax_from_conflicts(merged_ir)
                merged_ir.metadata["relaxed_ir"] = relaxed

        self.current_ir = merged_ir
        self.history.append(copy.deepcopy(self.current_ir))
        return self.current_ir

    def reset(self) -> None:
        """Clear session state and history."""
        self.current_ir = None
        self.history.clear()
