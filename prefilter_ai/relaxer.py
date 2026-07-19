"""
relaxer.py — Query Filter Relaxation Engine for Prefilter AI.

Allows gradual, importance-based relaxation of search constraints when a
search yields zero or too few results. Also supports conflict-aware relaxation:
when the validator tags specific fields as conflicting, those are targeted
directly rather than removing arbitrary low-priority constraints.
"""

from __future__ import annotations

import copy
from prefilter_ai.ir import IntermediateRepresentation, IRFilterConstraint
from prefilter_ai.registry import SchemaRegistry, Importance


class QueryRelaxer:
    """Relaxes search query constraints based on schema importance weights."""

    def __init__(self, registry: SchemaRegistry | None = None):
        self.registry = registry or SchemaRegistry()

    # ------------------------------------------------------------------
    # Conflict-aware relaxation (primary method — fixes Gap #5)
    # ------------------------------------------------------------------

    def relax_from_conflicts(
        self, ir: IntermediateRepresentation
    ) -> IntermediateRepresentation:
        """
        Targeted relaxation driven by conflict detector tags.

        Instead of removing low-priority constraints arbitrarily, this method
        reads ``ir.metadata["conflict_tags"]`` to find *which fields actually
        caused a conflict* and relaxes those first.

        For price conflicts: expands price ceiling by 40%.
        For categorical conflicts (cabin_class, stars): removes the constraint.
        For experience/salary conflicts: drops the lesser-priority of the two.
        """
        conflict_tags = ir.metadata.get("conflict_tags", [])
        if not conflict_tags:
            return self.relax(ir, relaxation_level=1)

        relaxed_ir = copy.deepcopy(ir)
        schema = self.registry.get(ir.domain)

        def _field_importance(field: str) -> Importance:
            if schema:
                fd = schema.fields.get(field)
                if fd:
                    return fd.importance
            return Importance.MEDIUM

        for tag in conflict_tags:
            conflicting = tag.get("conflicting_fields", [])
            # Sort by importance ascending — relax least important first
            conflicting_sorted = sorted(
                conflicting, key=lambda f: _field_importance(f).value
            )

            for field in conflicting_sorted:
                filters_for_field = [f for f in relaxed_ir.filters if f.field == field]
                if not filters_for_field:
                    continue

                # Price: expand ceiling by 40%
                if field == "price":
                    for f in filters_for_field:
                        if f.operator in {"lt", "lte"}:
                            try:
                                old = float(f.value)
                                new_val = round(old * 1.40, 2)
                                f.value = new_val
                                f.provenance = f"{f.provenance} (conflict-relaxed from {old} +40%)"
                                relaxed_ir.warnings.append(
                                    f"Conflict relaxation: expanded price ceiling from ${old:.0f} to ${new_val:.0f}"
                                )
                            except (TypeError, ValueError):
                                pass

                # Stars / cabin_class / categorical: remove the constraint
                elif field in {"stars", "cabin_class", "experience_years"}:
                    imp = _field_importance(field)
                    if imp in {Importance.LOW, Importance.MEDIUM}:
                        relaxed_ir.filters = [
                            f for f in relaxed_ir.filters if f.field != field
                        ]
                        relaxed_ir.warnings.append(
                            f"Conflict relaxation: removed '{field}' constraint to resolve feasibility issue"
                        )

                # Break after relaxing one field per tag to avoid over-relaxing
                break

        relaxed_ir.metadata["relaxed_by"] = "conflict_aware"
        return relaxed_ir

    # ------------------------------------------------------------------
    # Standard importance-level relaxation
    # ------------------------------------------------------------------

    def relax(
        self, ir: IntermediateRepresentation, relaxation_level: int = 1
    ) -> IntermediateRepresentation:
        """
        Generate a relaxed copy of the IR state.

        Parameters
        ----------
        relaxation_level : int
            1 = remove LOW importance filters (e.g. color, amenities).
            2 = remove LOW + expand numeric constraints (price +25%).
            3 = remove LOW + MEDIUM importance filters (brand, features).
        """
        relaxed_ir = copy.deepcopy(ir)
        schema = self.registry.get(ir.domain)

        if not schema:
            return self._relax_heuristically(relaxed_ir, relaxation_level)

        fields_def = schema.fields
        new_filters: list[IRFilterConstraint] = []
        log_entries: list[str] = []

        for f in ir.filters:
            f_def = fields_def.get(f.field)
            importance = f_def.importance if f_def else Importance.MEDIUM

            if relaxation_level == 1:
                if importance == Importance.LOW:
                    log_entries.append(f"Dropped low-importance constraint: {f.field}")
                    continue
                new_filters.append(f)

            elif relaxation_level == 2:
                if importance == Importance.LOW:
                    log_entries.append(f"Dropped low-importance constraint: {f.field}")
                    continue

                # Expand price or other numeric limit by 25%
                if f_def and f_def.name == "price" and f.operator in {"lt", "lte"}:
                    try:
                        old_val = float(f.value)
                        new_val = round(old_val * 1.25, 2)
                        relaxed_f = copy.deepcopy(f)
                        relaxed_f.value = new_val
                        relaxed_f.provenance = f"{f.provenance} (relaxed from {old_val} +25%)"
                        new_filters.append(relaxed_f)
                        log_entries.append(f"Expanded price ceiling from {old_val} to {new_val}")
                        continue
                    except (TypeError, ValueError):
                        pass

                new_filters.append(f)

            elif relaxation_level >= 3:
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
        """Fallback relaxation when domain schema is not registered."""
        relaxed_ir = copy.deepcopy(ir)
        new_filters = []
        _CORE_FIELDS = {"price", "product", "origin", "destination", "job_title", "city"}
        for f in ir.filters:
            if f.field in _CORE_FIELDS:
                if level >= 2 and f.field == "price" and f.operator in {"lt", "lte"}:
                    try:
                        new_val = round(float(f.value) * 1.25, 2)
                        f.value = new_val
                        f.provenance = f"{f.provenance} (relaxed)"
                        relaxed_ir.warnings.append(f"Relaxed price limit to {new_val}")
                    except (TypeError, ValueError):
                        pass
                new_filters.append(f)
            else:
                relaxed_ir.warnings.append(f"Dropped constraint: {f.field}")

        relaxed_ir.filters = new_filters
        return relaxed_ir
