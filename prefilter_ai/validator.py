"""
validator.py — Conflict Detector & Feasibility Validator for Prefilter AI.

Detects logical, mathematical, or pricing contradictions within constraints
and provides helpful user feedback/recommendations.
"""

from __future__ import annotations

import re
from prefilter_ai.ir import IntermediateRepresentation


class ConflictDetector:
    """Detects logical or pricing conflicts in query constraints."""

    def validate(self, ir: IntermediateRepresentation) -> IntermediateRepresentation:
        """Analyze IR filters and flags any contradictions or impossible combinations."""
        self._check_numerical_conflicts(ir)
        self._check_categorical_conflicts(ir)
        self._check_pricing_feasibility(ir)
        return ir

    def _check_numerical_conflicts(self, ir: IntermediateRepresentation) -> None:
        """Detect disjoint numerical bounds, e.g. price < 100 and price > 200."""
        # Group numeric constraints by field
        by_field: dict[str, list[dict]] = {}
        for f in ir.filters:
            # We parse operators
            op = f.operator
            if op in {"lt", "lte", "gt", "gte", "between"}:
                if f.field not in by_field:
                    by_field[f.field] = []
                by_field[f.field].append({
                    "op": op,
                    "val": f.value,
                    "val_hi": f.value_hi
                })

        for field_name, constraints in by_field.items():
            min_val = float("-inf")
            max_val = float("inf")

            for c in constraints:
                op, val = c["op"], c["val"]
                if op == "lt":
                    max_val = min(max_val, val - 0.01)
                elif op == "lte":
                    max_val = min(max_val, val)
                elif op == "gt":
                    min_val = max(min_val, val + 0.01)
                elif op == "gte":
                    min_val = max(min_val, val)
                elif op == "between":
                    min_val = max(min_val, val)
                    max_val = min(max_val, c["val_hi"])

            if min_val > max_val:
                ir.conflicts.append(
                    f"Contradictory numerical constraints on '{field_name}': "
                    f"requires value to be simultaneously >={min_val} and <={max_val}."
                )
                ir.warnings.append(
                    f"Warning: No listings will match because the '{field_name}' range is impossible."
                )

    def _check_categorical_conflicts(self, ir: IntermediateRepresentation) -> None:
        """Detect mutual exclusions, e.g., color = 'red' AND color != 'red' (ne:red)."""
        eq_fields: dict[str, set] = {}
        ne_fields: dict[str, set] = {}

        for f in ir.filters:
            if f.operator == "eq":
                if f.field not in eq_fields:
                    eq_fields[f.field] = set()
                eq_fields[f.field].add(str(f.value).lower())
            elif f.operator == "ne":
                if f.field not in ne_fields:
                    ne_fields[f.field] = set()
                ne_fields[f.field].add(str(f.value).lower())

        for field_name, eq_vals in eq_fields.items():
            ne_vals = ne_fields.get(field_name, set())
            overlap = eq_vals.intersection(ne_vals)
            if overlap:
                items = ", ".join(overlap)
                ir.conflicts.append(
                    f"Contradictory categorical constraints on '{field_name}': "
                    f"requires '{items}' to be both included and excluded."
                )
                ir.warnings.append(
                    f"Conflict: Exclusions clash with requirements for '{field_name}'."
                )

    def _check_pricing_feasibility(self, ir: IntermediateRepresentation) -> None:
        """Detect business-logic conflicts, e.g. gaming laptops or RTX 4080 under $400."""
        # Find price bounds
        max_price = float("inf")
        has_rtx = False
        is_gaming = False
        is_luxury_hotel = False

        for f in ir.filters:
            if f.field == "price":
                if f.operator in {"lt", "lte"}:
                    max_price = min(max_price, float(f.value))
                elif f.operator == "between":
                    max_price = min(max_price, float(f.value_hi))
            
            # Identify products/features
            val_str = str(f.value).lower()
            if "rtx" in val_str or "4080" in val_str or "4090" in val_str:
                has_rtx = True
            if "gaming" in val_str or "laptop" in val_str:
                is_gaming = True
            if f.field == "stars" and f.operator in {"eq", "gt", "gte"} and float(f.value) >= 5:
                is_luxury_hotel = True

        # Check laptop pricing feasibility
        if ir.domain == "ecommerce" and (has_rtx or is_gaming) and max_price < 600:
            ir.conflicts.append(
                f"Feasibility conflict: High-performance gaming/RTX systems cannot be found under ${max_price}."
            )
            ir.warnings.append(
                f"Recommendation: Increase your budget to $1,000+ for gaming systems, or relax the RTX/GPU constraint."
            )

        # Check hotel pricing feasibility
        if ir.domain == "hotels" and is_luxury_hotel and max_price < 150:
            ir.conflicts.append(
                f"Feasibility conflict: 5-star luxury hotels are not available under ${max_price}/night."
            )
            ir.warnings.append(
                f"Recommendation: Increase budget per night to $250+, or lower the hotel star rating requirement."
            )
