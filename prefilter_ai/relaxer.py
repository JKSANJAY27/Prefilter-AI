"""
relaxer.py — Query Filter Relaxation Engine for Prefilter AI.

Allows gradual, importance-based relaxation of search constraints
when a search yields zero or too few results.
"""

from __future__ import annotations

import copy
from prefilter_ai.ir import IntermediateRepresentation, IRFilterConstraint
from prefilter_ai.registry import SchemaRegistry, Importance


class QueryRelaxer:
    """Relaxes search query constraints based on schema importance weights."""

    def __init__(self, registry: SchemaRegistry | None = None):
        self.registry = registry or SchemaRegistry()

    def relax(
        self, ir: IntermediateRepresentation, relaxation_level: int = 1
    ) -> IntermediateRepresentation:
        """
        Generate a relaxed copy of the IR state.
        
        Parameters
        ----------
        ir : IntermediateRepresentation
            The original query understanding state.
        relaxation_level : int
            1 = remove LOW importance filters (e.g. color, ratings).
            2 = remove LOW + expand numeric constraints (e.g. increase price range by 25%).
            3 = remove LOW + MEDIUM importance filters (e.g. brand, features).
        """
        relaxed_ir = copy.deepcopy(ir)
        schema = self.registry.get(ir.domain)

        # If no schema registered, we fallback to general heuristic
        if not schema:
            return self._relax_heuristically(relaxed_ir, relaxation_level)

        # Get field definitions
        fields_def = schema.fields

        new_filters: list[IRFilterConstraint] = []
        log_entries: list[str] = []

        for f in ir.filters:
            f_def = fields_def.get(f.field)
            importance = f_def.importance if f_def else Importance.MEDIUM

            if relaxation_level == 1:
                # Level 1: Drop LOW importance filters
                if importance == Importance.LOW:
                    log_entries.append(f"Dropped low importance constraint: {f.field} = {f.value}")
                    continue
                new_filters.append(f)

            elif relaxation_level == 2:
                # Level 2: Drop LOW + relax numeric bounds
                if importance == Importance.LOW:
                    log_entries.append(f"Dropped low importance constraint: {f.field} = {f.value}")
                    continue
                
                # Relax price or numeric limit by 25%
                if f_def and f_def.name == "price" and f.operator in {"lt", "lte"}:
                    try:
                        old_val = float(f.value)
                        new_val = old_val * 1.25
                        relaxed_f = copy.deepcopy(f)
                        relaxed_f.value = new_val
                        relaxed_f.provenance = f"{f.provenance} (relaxed from {old_val} by +25%)"
                        new_filters.append(relaxed_f)
                        log_entries.append(f"Relaxed price ceiling from {old_val} to {new_val}")
                        continue
                    except ValueError:
                        pass
                
                new_filters.append(f)

            elif relaxation_level >= 3:
                # Level 3: Drop LOW and MEDIUM constraints
                if importance in {Importance.LOW, Importance.MEDIUM}:
                    log_entries.append(f"Dropped low/medium constraint: {f.field}")
                    continue
                new_filters.append(f)

        relaxed_ir.filters = new_filters
        relaxed_ir.metadata["relaxed_level"] = relaxation_level
        relaxed_ir.metadata["relaxation_logs"] = log_entries
        for entry in log_entries:
            relaxed_ir.warnings.append(f"Auto-relaxation: {entry}")

        return relaxed_ir

    def _relax_heuristically(
        self, ir: IntermediateRepresentation, level: int
    ) -> IntermediateRepresentation:
        """Fallback relaxation when domain schema definitions are not present."""
        relaxed_ir = copy.deepcopy(ir)
        new_filters = []
        for f in ir.filters:
            # Heuristic: keep price and domain/product filters, drop others
            if f.field in {"price", "product", "origin", "destination", "job_title"}:
                if level >= 2 and f.field == "price" and f.operator in {"lt", "lte"}:
                    try:
                        val = float(f.value) * 1.25
                        f.value = val
                        f.provenance = f"{f.provenance} (relaxed)"
                        relaxed_ir.warnings.append(f"Relaxed pricing limit to {val}")
                    except ValueError:
                        pass
                new_filters.append(f)
            else:
                relaxed_ir.warnings.append(f"Dropped filter: {f.field}")
        
        relaxed_ir.filters = new_filters
        return relaxed_ir
