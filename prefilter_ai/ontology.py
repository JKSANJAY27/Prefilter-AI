"""
ontology.py — Ontology & Soft Preference Inference Engine for Prefilter AI.

Maps semantic queries, synonyms, and high-level developer concepts to
explicit and implicit constraints with provenance tracking.
"""

from __future__ import annotations

from typing import Any
from prefilter_ai.ir import IntermediateRepresentation, IRPreference


class OntologyEngine:
    _instance: OntologyEngine | None = None
    _rules: dict[str, list[dict[str, Any]]]

    def __new__(cls) -> OntologyEngine:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._rules = {}
            cls._instance._load_default_ontology()
        return cls._instance

    def register_rule(
        self,
        domain: str,
        keyword: str,
        inferred_filters: list[dict[str, Any]] | None = None,
        inferred_preferences: list[dict[str, Any]] | None = None,
    ) -> None:
        """Register a new ontology rule dynamically for custom domain mappings."""
        if domain not in self._rules:
            self._rules[domain] = []
        self._rules[domain].append({
            "keyword": keyword.lower(),
            "filters": inferred_filters or [],
            "preferences": inferred_preferences or [],
        })

    def infer(self, ir: IntermediateRepresentation, query: str) -> IntermediateRepresentation:
        """
        Inspect query text and current IR state, mapping high-level intents
        to explicit filters or soft ranking preferences.
        """
        q_lower = query.lower()
        domain = ir.domain

        # Apply domain-specific rules
        rules = self._rules.get(domain, [])
        for rule in rules:
            if rule["keyword"] in q_lower:
                # Add implicit filters
                for f in rule["filters"]:
                    ir.add_filter(
                        field_name=f["field"],
                        operator=f["operator"],
                        value=f["value"],
                        value_hi=f.get("value_hi"),
                        confidence=f.get("confidence", 0.8),
                        provenance=f.get("provenance", f"Ontology match: '{rule['keyword']}'"),
                    )
                # Add soft preferences
                for p in rule["preferences"]:
                    ir.add_preference(
                        field_name=p["field"],
                        value=p["value"],
                        weight=p.get("weight", 0.5),
                        confidence=p.get("confidence", 0.75),
                        provenance=p.get("provenance", f"Ontology mapping: '{rule['keyword']}'"),
                    )

        # Apply cross-domain heuristics
        self._apply_heuristics(ir, q_lower)

        return ir

    def _apply_heuristics(self, ir: IntermediateRepresentation, query_lower: str) -> None:
        # Synonyms and simple mappings
        if "cheap" in query_lower or "budget" in query_lower:
            ir.add_preference(
                field_name="price",
                value="low",
                weight=0.8,
                confidence=0.85,
                provenance="Ontology heuristic: 'budget' -> price preference",
            )
        if "luxury" in query_lower or "premium" in query_lower:
            ir.add_preference(
                field_name="price",
                value="high",
                weight=0.7,
                confidence=0.8,
                provenance="Ontology heuristic: 'luxury' -> premium pricing preference",
            )

    def _load_default_ontology(self) -> None:
        """Register the default out-of-the-box knowledge maps."""
        # Ecommerce - Laptop Specs
        self.register_rule(
            domain="ecommerce",
            keyword="ai",
            inferred_filters=[
                {"field": "feature", "operator": "eq", "value": "RTX GPU", "confidence": 0.85, "provenance": "Ontology: 'AI' -> CUDA/RTX GPU Required"}
            ],
            inferred_preferences=[
                {"field": "ram", "value": "16GB+", "weight": 0.9, "confidence": 0.9, "provenance": "Ontology: 'AI' -> 16GB+ RAM Preferred"},
                {"field": "vram", "value": "8GB+", "weight": 0.8, "confidence": 0.8, "provenance": "Ontology: 'AI' -> 8GB+ VRAM Preferred"},
            ]
        )
        self.register_rule(
            domain="ecommerce",
            keyword="gaming",
            inferred_filters=[
                {"field": "feature", "operator": "eq", "value": "Dedicated GPU", "confidence": 0.9, "provenance": "Ontology: 'gaming' -> Dedicated GPU Required"}
            ],
            inferred_preferences=[
                {"field": "refresh_rate", "value": "144Hz", "weight": 0.75, "confidence": 0.8, "provenance": "Ontology: 'gaming' -> 144Hz refresh rate"},
                {"field": "ram", "value": "16GB", "weight": 0.8, "confidence": 0.85, "provenance": "Ontology: 'gaming' -> 16GB RAM recommendation"},
            ]
        )
        self.register_rule(
            domain="ecommerce",
            keyword="coding",
            inferred_preferences=[
                {"field": "ram", "value": "16GB", "weight": 0.85, "confidence": 0.85, "provenance": "Ontology: 'coding' -> 16GB RAM Preferred"},
                {"field": "cpu", "value": "i7/Ryzen 7", "weight": 0.7, "confidence": 0.75, "provenance": "Ontology: 'coding' -> Core i7/Ryzen 7 Processor"},
            ]
        )

        # Hotels - Tourist preferences
        self.register_rule(
            domain="hotels",
            keyword="eiffel tower",
            inferred_filters=[
                {"field": "city", "operator": "eq", "value": "Paris", "confidence": 0.99, "provenance": "Ontology: Eiffel Tower location is Paris"}
            ],
            inferred_preferences=[
                {"field": "walking_distance", "value": True, "weight": 0.85, "confidence": 0.8, "provenance": "Ontology: Eiffel Tower -> walking distance"},
                {"field": "public_transport", "value": True, "weight": 0.7, "confidence": 0.75, "provenance": "Ontology: tourist spot -> transport access"},
            ]
        )
        self.register_rule(
            domain="hotels",
            keyword="beachfront",
            inferred_preferences=[
                {"field": "view", "value": "ocean", "weight": 0.9, "confidence": 0.9, "provenance": "Ontology: beachfront -> ocean view"},
                {"field": "pool", "value": True, "weight": 0.6, "confidence": 0.7, "provenance": "Ontology: resort -> pool recommendation"},
            ]
        )
