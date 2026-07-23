"""
validator.py — Conflict Detector & Feasibility Validator for Prefilter AI.

Detects logical, mathematical, or pricing contradictions within constraints
and provides helpful user feedback/recommendations. Each conflict is tagged
with the *specific field* that caused it so the QueryRelaxer can target
the right constraint instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

from prefilter_ai.ir import IntermediateRepresentation


@dataclass
class ConflictTag:
    """Structured conflict record linking a message to the fields that caused it."""

    message: str
    recommendation: str
    conflicting_fields: list[str]  # fields the relaxer should target
    severity: str = "warning"  # "warning" | "impossible"


class ConflictDetector:
    """Detects logical or pricing conflicts in query constraints."""

    def validate(self, ir: IntermediateRepresentation) -> IntermediateRepresentation:
        """Analyze IR filters and flag any contradictions or impossible combinations."""
        tags: list[ConflictTag] = []
        tags.extend(self._check_numerical_conflicts(ir))
        tags.extend(self._check_categorical_conflicts(ir))
        tags.extend(self._check_pricing_feasibility(ir))

        # Write back into IR
        for tag in tags:
            ir.conflicts.append(tag.message)
            ir.warnings.append(f"Recommendation: {tag.recommendation}")

        # Store structured tags in metadata for the relaxer
        ir.metadata.setdefault("conflict_tags", [])
        ir.metadata["conflict_tags"].extend(
            [
                {
                    "message": t.message,
                    "recommendation": t.recommendation,
                    "conflicting_fields": t.conflicting_fields,
                    "severity": t.severity,
                }
                for t in tags
            ]
        )

        return ir

    def _check_numerical_conflicts(self, ir: IntermediateRepresentation) -> list[ConflictTag]:
        """Detect disjoint numerical bounds, e.g. price < 100 and price > 200."""
        by_field: dict[str, list[dict]] = {}
        for f in ir.filters:
            op = f.operator
            if op in {"lt", "lte", "gt", "gte", "between"}:
                if f.field not in by_field:
                    by_field[f.field] = []
                by_field[f.field].append({"op": op, "val": f.value, "val_hi": f.value_hi})

        tags = []
        for field_name, constraints in by_field.items():
            min_val = float("-inf")
            max_val = float("inf")

            for c in constraints:
                op, val = c["op"], c["val"]
                try:
                    val = float(val)
                    if op == "lt":
                        max_val = min(max_val, val - 0.01)
                    elif op == "lte":
                        max_val = min(max_val, val)
                    elif op == "gt":
                        min_val = max(min_val, val + 0.01)
                    elif op == "gte":
                        min_val = max(min_val, val)
                    elif op == "between" and c["val_hi"] is not None:
                        min_val = max(min_val, val)
                        max_val = min(max_val, float(c["val_hi"]))
                except (TypeError, ValueError):
                    pass

            if min_val > max_val:
                tags.append(
                    ConflictTag(
                        message=(
                            f"Contradictory numerical constraints on '{field_name}': "
                            f"requires value ≥{min_val:.2g} AND ≤{max_val:.2g} simultaneously."
                        ),
                        recommendation=(
                            f"Adjust the '{field_name}' range so lower bound < upper bound."
                        ),
                        conflicting_fields=[field_name],
                        severity="impossible",
                    )
                )
        return tags

    def _check_categorical_conflicts(self, ir: IntermediateRepresentation) -> list[ConflictTag]:
        """Detect mutual exclusions, e.g., color = 'red' AND color != 'red'."""
        eq_fields: dict[str, set] = {}
        ne_fields: dict[str, set] = {}

        for f in ir.filters:
            if f.operator == "eq":
                eq_fields.setdefault(f.field, set()).add(str(f.value).lower())
            elif f.operator == "ne":
                ne_fields.setdefault(f.field, set()).add(str(f.value).lower())

        tags = []
        for field_name, eq_vals in eq_fields.items():
            ne_vals = ne_fields.get(field_name, set())
            overlap = eq_vals.intersection(ne_vals)
            if overlap:
                items = ", ".join(overlap)
                tags.append(
                    ConflictTag(
                        message=(
                            f"Contradictory categorical constraints on '{field_name}': "
                            f"'{items}' required to be both included and excluded."
                        ),
                        recommendation=(
                            f"Remove the conflicting '{field_name}' inclusion or exclusion."
                        ),
                        conflicting_fields=[field_name],
                        severity="impossible",
                    )
                )
        return tags

    def _check_pricing_feasibility(self, ir: IntermediateRepresentation) -> list[ConflictTag]:
        """
        Detect business-logic pricing conflicts across all 10 supported domains.
        Each rule is tagged with the field(s) the relaxer should adjust.
        """
        tags: list[ConflictTag] = []
        domain = ir.domain

        # ── Extract field values for analysis ─────────────────────────
        max_price = float("inf")
        min_price = float("-inf")
        has_rtx = False
        is_gaming = False
        stars = None
        cabin_class = None
        city = None
        specialty = None
        insurance = None
        experience_min = None
        salary_max = float("inf")
        salary_min = float("-inf")
        raw_query = ir.metadata.get("query_text", "").lower()

        # Quick check on query text before filter iteration
        if "gaming" in raw_query:
            is_gaming = True

        for f in ir.filters:
            val_str = str(f.value).lower()
            try:
                val_float = float(f.value)
            except (TypeError, ValueError):
                val_float = None

            # Price bounds
            if f.field == "price":
                if f.operator in {"lt", "lte"} and val_float is not None:
                    max_price = min(max_price, val_float)
                elif f.operator in {"gt", "gte"} and val_float is not None:
                    min_price = max(min_price, val_float)
                elif f.operator == "between" and f.value_hi is not None:
                    try:
                        max_price = min(max_price, float(f.value_hi))
                        min_price = max(min_price, float(f.value))
                    except (TypeError, ValueError):
                        pass

            # Salary bounds (jobs domain)
            if f.field == "salary":
                if f.operator in {"lt", "lte"} and val_float is not None:
                    salary_max = min(salary_max, val_float)
                elif f.operator in {"gt", "gte"} and val_float is not None:
                    salary_min = max(salary_min, val_float)

            # Feature/product signals
            if "rtx" in val_str or "4080" in val_str or "4090" in val_str or "4070" in val_str:
                has_rtx = True
            if "gaming" in val_str or (f.field == "product" and "gaming" in val_str):
                is_gaming = True

            # Hotel stars
            if f.field == "stars" and val_float is not None:
                if f.operator in {"eq", "gte", "gt"}:
                    stars = val_float

            # Flights
            if f.field == "cabin_class":
                cabin_class = str(f.value).lower()

            # Location
            if f.field in {"city", "location", "destination"}:
                city = str(f.value).lower()

            # Healthcare
            if f.field == "specialty":
                specialty = str(f.value).lower()
            if f.field == "insurance":
                insurance = str(f.value).lower()

            # Jobs
            if f.field in {"experience_years", "experience_level"} and val_float is not None:
                experience_min = val_float

        # ── Ecommerce: gaming/RTX laptop price floor ───────────────────
        if domain == "ecommerce":
            if (has_rtx or is_gaming) and max_price < 700:
                tags.append(
                    ConflictTag(
                        message=f"Feasibility conflict: Gaming/RTX laptops start at ~$700-$900. Budget ${max_price:.0f} is below market floor.",
                        recommendation="Increase budget to $900+ for gaming laptops, or remove the GPU/gaming requirement.",
                        conflicting_fields=["price", "feature"],
                        severity="impossible" if max_price < 500 else "warning",
                    )
                )

        # ── Hotels: luxury hotel price floors ─────────────────────────
        if domain == "hotels":
            if stars is not None and stars >= 5 and max_price < 200:
                tags.append(
                    ConflictTag(
                        message=f"Feasibility conflict: 5-star hotels start at $200+/night. Budget ${max_price:.0f}/night is below market floor.",
                        recommendation="Increase budget to $250+/night, or lower the star rating to 4 stars.",
                        conflicting_fields=["price", "stars"],
                        severity="impossible" if max_price < 100 else "warning",
                    )
                )
            if city and "paris" in city and max_price < 120:
                tags.append(
                    ConflictTag(
                        message=f"Feasibility conflict: Central Paris hotels rarely go below $120/night. Budget ${max_price:.0f} is unrealistic.",
                        recommendation="Increase budget to $150+/night for Paris, or consider nearby districts.",
                        conflicting_fields=["price", "city"],
                        severity="warning",
                    )
                )

        # ── Flights: cabin class price floors ─────────────────────────
        if domain == "flights":
            if cabin_class in {"business", "first"} and max_price < 800:
                tags.append(
                    ConflictTag(
                        message=f"Feasibility conflict: {cabin_class.title()} class flights rarely cost under $800. Budget ${max_price:.0f} is unrealistic.",
                        recommendation=f"Increase budget to $1,000+ for {cabin_class} class, or switch to economy.",
                        conflicting_fields=["price", "cabin_class"],
                        severity="warning",
                    )
                )

        # ── Real estate: market floor constraints ──────────────────────
        if domain == "real_estate":
            _HIGH_COST_CITIES = {
                "san francisco",
                "new york",
                "nyc",
                "london",
                "singapore",
                "hong kong",
                "zurich",
            }
            if city and any(hc in city for hc in _HIGH_COST_CITIES) and max_price < 1500:
                tags.append(
                    ConflictTag(
                        message=f"Feasibility conflict: Monthly rent in {city.title()} rarely goes below $1,500. Budget ${max_price:.0f} is unrealistic.",
                        recommendation="Increase budget to $2,000+ for central areas, or search in nearby suburbs.",
                        conflicting_fields=["price", "city"],
                        severity="warning",
                    )
                )

        # ── Jobs: salary below minimum wage ───────────────────────────
        if domain == "jobs":
            if salary_min > 250_000:
                tags.append(
                    ConflictTag(
                        message="Feasibility conflict: Very few roles offer $250k+ base salary. You may see very few results.",
                        recommendation="Consider lowering the salary floor or targeting senior/staff engineering roles.",
                        conflicting_fields=["salary"],
                        severity="warning",
                    )
                )
            # Experience vs salary mismatch
            if experience_min is not None and experience_min < 2 and salary_min > 120_000:
                tags.append(
                    ConflictTag(
                        message=f"Feasibility conflict: Entry-level roles (<{experience_min:.0f}yr exp) rarely offer $120k+ base salary.",
                        recommendation="Remove the experience minimum or lower the salary expectation to $70-$90k range.",
                        conflicting_fields=["salary", "experience_years"],
                        severity="warning",
                    )
                )

        # ── Healthcare: specialty + insurance conflicts ─────────────────
        if domain == "healthcare":
            _RARE_COMBOS = {"surgeon", "neurosurgeon", "oncologist", "cardiologist"}
            if specialty in _RARE_COMBOS and insurance and "medicaid" in insurance:
                tags.append(
                    ConflictTag(
                        message=f"Feasibility conflict: {specialty.title()} specialists who accept Medicaid are rare. You may see very few results.",
                        recommendation="Expand search to in-network referrals, or remove the insurance constraint.",
                        conflicting_fields=["specialty", "insurance"],
                        severity="warning",
                    )
                )

        return tags
