"""
ontology.py — Ontology & Soft Preference Inference Engine for Prefilter AI.

Maps semantic queries, synonyms, and high-level intent concepts to explicit
and implicit constraints with provenance tracking. Covers all 10 domains.
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
                for f in rule["filters"]:
                    ir.add_filter(
                        field_name=f["field"],
                        operator=f["operator"],
                        value=f["value"],
                        value_hi=f.get("value_hi"),
                        confidence=f.get("confidence", 0.8),
                        provenance=f.get("provenance", f"Ontology: '{rule['keyword']}'"),
                    )
                for p in rule["preferences"]:
                    ir.add_preference(
                        field_name=p["field"],
                        value=p["value"],
                        weight=p.get("weight", 0.5),
                        confidence=p.get("confidence", 0.75),
                        provenance=p.get("provenance", f"Ontology: '{rule['keyword']}'"),
                    )

        # Cross-domain heuristics
        self._apply_heuristics(ir, q_lower)
        return ir

    def _apply_heuristics(self, ir: IntermediateRepresentation, query_lower: str) -> None:
        if "cheap" in query_lower or "budget" in query_lower or "affordable" in query_lower:
            ir.add_preference("price", "low", weight=0.8, confidence=0.85,
                              provenance="Ontology heuristic: 'budget' → low price")
        if "luxury" in query_lower or "premium" in query_lower or "high-end" in query_lower:
            ir.add_preference("price", "high", weight=0.7, confidence=0.8,
                              provenance="Ontology heuristic: 'luxury' → premium pricing")
        if "new" in query_lower and ir.domain in {"cars", "ecommerce"}:
            ir.add_preference("condition", "new", weight=0.85, confidence=0.8,
                              provenance="Ontology heuristic: 'new' → new condition preferred")
        if "beginner" in query_lower or "starter" in query_lower or "entry level" in query_lower:
            ir.add_preference("difficulty", "beginner", weight=0.85, confidence=0.85,
                              provenance="Ontology heuristic: 'beginner' → entry-level content")

    def _load_default_ontology(self) -> None:
        """Register the full out-of-the-box knowledge map for all 10 domains."""

        # ── ECOMMERCE ──────────────────────────────────────────────────
        self.register_rule(
            domain="ecommerce", keyword="ai",
            inferred_filters=[
                {"field": "feature", "operator": "eq", "value": "CUDA/RTX GPU",
                 "confidence": 0.85, "provenance": "Ontology: 'AI' → CUDA/RTX GPU required"},
            ],
            inferred_preferences=[
                {"field": "ram", "value": "16GB+", "weight": 0.9, "confidence": 0.9,
                 "provenance": "Ontology: 'AI' → 16GB+ RAM"},
                {"field": "vram", "value": "8GB+", "weight": 0.8, "confidence": 0.8,
                 "provenance": "Ontology: 'AI' → 8GB+ VRAM"},
                {"field": "storage", "value": "512GB+ SSD", "weight": 0.65, "confidence": 0.75,
                 "provenance": "Ontology: 'AI' → fast SSD storage"},
            ]
        )
        self.register_rule(
            domain="ecommerce", keyword="machine learning",
            inferred_filters=[
                {"field": "feature", "operator": "eq", "value": "Dedicated GPU",
                 "confidence": 0.9, "provenance": "Ontology: 'ML' → GPU required"},
            ],
            inferred_preferences=[
                {"field": "ram", "value": "32GB", "weight": 0.9, "confidence": 0.88,
                 "provenance": "Ontology: 'ML' → 32GB RAM preferred"},
                {"field": "vram", "value": "12GB+", "weight": 0.85, "confidence": 0.85,
                 "provenance": "Ontology: 'ML' → 12GB+ VRAM for model training"},
            ]
        )
        self.register_rule(
            domain="ecommerce", keyword="gaming",
            inferred_filters=[
                {"field": "feature", "operator": "eq", "value": "Dedicated GPU",
                 "confidence": 0.9, "provenance": "Ontology: 'gaming' → Dedicated GPU"},
            ],
            inferred_preferences=[
                {"field": "refresh_rate", "value": "144Hz+", "weight": 0.75, "confidence": 0.8,
                 "provenance": "Ontology: 'gaming' → 144Hz+ display"},
                {"field": "ram", "value": "16GB", "weight": 0.8, "confidence": 0.85,
                 "provenance": "Ontology: 'gaming' → 16GB RAM"},
            ]
        )
        self.register_rule(
            domain="ecommerce", keyword="coding",
            inferred_preferences=[
                {"field": "ram", "value": "16GB", "weight": 0.85, "confidence": 0.85,
                 "provenance": "Ontology: 'coding' → 16GB RAM"},
                {"field": "cpu", "value": "i7/Ryzen 7", "weight": 0.7, "confidence": 0.75,
                 "provenance": "Ontology: 'coding' → i7/Ryzen 7 CPU"},
                {"field": "battery", "value": "10hr+", "weight": 0.6, "confidence": 0.7,
                 "provenance": "Ontology: 'coding' → good battery for portability"},
            ]
        )
        self.register_rule(
            domain="ecommerce", keyword="photo",
            inferred_preferences=[
                {"field": "display", "value": "color accurate", "weight": 0.85, "confidence": 0.8,
                 "provenance": "Ontology: 'photo editing' → color accurate display"},
                {"field": "ram", "value": "16GB", "weight": 0.75, "confidence": 0.78,
                 "provenance": "Ontology: 'photo editing' → 16GB RAM for large files"},
            ]
        )

        # ── HOTELS ────────────────────────────────────────────────────
        self.register_rule(
            domain="hotels", keyword="honeymoon",
            inferred_preferences=[
                {"field": "view", "value": "ocean/scenic", "weight": 0.9, "confidence": 0.85,
                 "provenance": "Ontology: 'honeymoon' → scenic view"},
                {"field": "amenities", "value": "spa", "weight": 0.8, "confidence": 0.8,
                 "provenance": "Ontology: 'honeymoon' → spa amenity"},
                {"field": "room_type", "value": "suite", "weight": 0.85, "confidence": 0.82,
                 "provenance": "Ontology: 'honeymoon' → suite preferred"},
            ]
        )
        self.register_rule(
            domain="hotels", keyword="beachfront",
            inferred_preferences=[
                {"field": "view", "value": "ocean", "weight": 0.9, "confidence": 0.9,
                 "provenance": "Ontology: 'beachfront' → ocean view"},
                {"field": "pool", "value": True, "weight": 0.6, "confidence": 0.7,
                 "provenance": "Ontology: 'beachfront resort' → pool"},
            ]
        )
        self.register_rule(
            domain="hotels", keyword="business trip",
            inferred_preferences=[
                {"field": "wifi", "value": "high-speed", "weight": 0.95, "confidence": 0.9,
                 "provenance": "Ontology: 'business trip' → high-speed wifi"},
                {"field": "amenities", "value": "gym", "weight": 0.6, "confidence": 0.7,
                 "provenance": "Ontology: 'business trip' → gym"},
                {"field": "location", "value": "city center/CBD", "weight": 0.85, "confidence": 0.82,
                 "provenance": "Ontology: 'business trip' → central location"},
            ]
        )
        self.register_rule(
            domain="hotels", keyword="eiffel tower",
            inferred_filters=[
                {"field": "city", "operator": "eq", "value": "Paris", "confidence": 0.99,
                 "provenance": "Ontology: Eiffel Tower → Paris"},
            ],
            inferred_preferences=[
                {"field": "walking_distance", "value": True, "weight": 0.85, "confidence": 0.8,
                 "provenance": "Ontology: Eiffel Tower → walking distance"},
            ]
        )
        self.register_rule(
            domain="hotels", keyword="family",
            inferred_preferences=[
                {"field": "amenities", "value": "pool", "weight": 0.8, "confidence": 0.8,
                 "provenance": "Ontology: 'family' → pool"},
                {"field": "amenities", "value": "kids club", "weight": 0.7, "confidence": 0.72,
                 "provenance": "Ontology: 'family' → kids facilities"},
                {"field": "room_type", "value": "family room/suite", "weight": 0.85, "confidence": 0.82,
                 "provenance": "Ontology: 'family' → larger room"},
            ]
        )

        # ── FLIGHTS ───────────────────────────────────────────────────
        self.register_rule(
            domain="flights", keyword="business trip",
            inferred_preferences=[
                {"field": "cabin_class", "value": "business", "weight": 0.8, "confidence": 0.75,
                 "provenance": "Ontology: 'business trip' → business class preferred"},
                {"field": "flexibility", "value": "refundable", "weight": 0.75, "confidence": 0.72,
                 "provenance": "Ontology: 'business trip' → flexible/refundable ticket"},
            ]
        )
        self.register_rule(
            domain="flights", keyword="backpacking",
            inferred_preferences=[
                {"field": "cabin_class", "value": "economy", "weight": 0.9, "confidence": 0.88,
                 "provenance": "Ontology: 'backpacking' → economy class"},
                {"field": "price", "value": "low", "weight": 0.9, "confidence": 0.88,
                 "provenance": "Ontology: 'backpacking' → budget fares"},
            ]
        )
        self.register_rule(
            domain="flights", keyword="long haul",
            inferred_preferences=[
                {"field": "stops", "value": "lte:1", "weight": 0.7, "confidence": 0.72,
                 "provenance": "Ontology: 'long haul' → prefer fewer stops"},
                {"field": "cabin_class", "value": "premium economy", "weight": 0.6, "confidence": 0.65,
                 "provenance": "Ontology: 'long haul' → comfort preference"},
            ]
        )

        # ── JOBS ──────────────────────────────────────────────────────
        self.register_rule(
            domain="jobs", keyword="senior",
            inferred_filters=[
                {"field": "experience_level", "operator": "eq", "value": "senior",
                 "confidence": 0.92, "provenance": "Ontology: 'senior' → experience level"},
            ],
            inferred_preferences=[
                {"field": "salary", "value": "120000+", "weight": 0.8, "confidence": 0.78,
                 "provenance": "Ontology: 'senior' → higher salary band"},
            ]
        )
        self.register_rule(
            domain="jobs", keyword="startup",
            inferred_preferences=[
                {"field": "company_size", "value": "small (<200)", "weight": 0.8, "confidence": 0.78,
                 "provenance": "Ontology: 'startup' → small company"},
                {"field": "equity", "value": "offered", "weight": 0.75, "confidence": 0.72,
                 "provenance": "Ontology: 'startup' → equity compensation"},
            ]
        )
        self.register_rule(
            domain="jobs", keyword="remote",
            inferred_filters=[
                {"field": "remote", "operator": "eq", "value": True,
                 "confidence": 0.95, "provenance": "Ontology: 'remote' → remote position"},
            ]
        )
        self.register_rule(
            domain="jobs", keyword="data scientist",
            inferred_preferences=[
                {"field": "skills", "value": "Python/R/SQL", "weight": 0.85, "confidence": 0.85,
                 "provenance": "Ontology: 'data scientist' → Python/R/SQL"},
                {"field": "skills", "value": "machine learning", "weight": 0.8, "confidence": 0.8,
                 "provenance": "Ontology: 'data scientist' → ML skills"},
            ]
        )

        # ── REAL ESTATE ───────────────────────────────────────────────
        self.register_rule(
            domain="real_estate", keyword="family",
            inferred_preferences=[
                {"field": "bedrooms", "value": "3+", "weight": 0.85, "confidence": 0.8,
                 "provenance": "Ontology: 'family' → 3+ bedrooms"},
                {"field": "schools", "value": "good school district", "weight": 0.8, "confidence": 0.75,
                 "provenance": "Ontology: 'family' → school district quality"},
                {"field": "yard", "value": True, "weight": 0.7, "confidence": 0.7,
                 "provenance": "Ontology: 'family' → yard/garden"},
            ]
        )
        self.register_rule(
            domain="real_estate", keyword="commute",
            inferred_preferences=[
                {"field": "transport", "value": "transit accessible", "weight": 0.85, "confidence": 0.82,
                 "provenance": "Ontology: 'commute' → transit access"},
                {"field": "location", "value": "near city center", "weight": 0.8, "confidence": 0.78,
                 "provenance": "Ontology: 'commute' → proximity to work"},
            ]
        )
        self.register_rule(
            domain="real_estate", keyword="investment",
            inferred_preferences=[
                {"field": "rental_yield", "value": "high", "weight": 0.85, "confidence": 0.8,
                 "provenance": "Ontology: 'investment' → rental yield"},
                {"field": "appreciation", "value": "growth area", "weight": 0.8, "confidence": 0.75,
                 "provenance": "Ontology: 'investment' → growth area"},
            ]
        )

        # ── CARS ──────────────────────────────────────────────────────
        self.register_rule(
            domain="cars", keyword="commute",
            inferred_preferences=[
                {"field": "fuel_type", "value": "hybrid/electric", "weight": 0.8, "confidence": 0.78,
                 "provenance": "Ontology: 'commute' → fuel efficiency preferred"},
                {"field": "mpg", "value": "35+", "weight": 0.75, "confidence": 0.72,
                 "provenance": "Ontology: 'commute' → good mileage"},
            ]
        )
        self.register_rule(
            domain="cars", keyword="off road",
            inferred_preferences=[
                {"field": "drivetrain", "value": "4WD/AWD", "weight": 0.9, "confidence": 0.88,
                 "provenance": "Ontology: 'off-road' → 4WD/AWD drivetrain"},
                {"field": "ground_clearance", "value": "high", "weight": 0.8, "confidence": 0.78,
                 "provenance": "Ontology: 'off-road' → high ground clearance"},
                {"field": "body_type", "value": "SUV/truck", "weight": 0.85, "confidence": 0.82,
                 "provenance": "Ontology: 'off-road' → SUV or truck"},
            ]
        )
        self.register_rule(
            domain="cars", keyword="family",
            inferred_preferences=[
                {"field": "seating", "value": "7+", "weight": 0.85, "confidence": 0.82,
                 "provenance": "Ontology: 'family car' → 7+ seating"},
                {"field": "safety_rating", "value": "5-star", "weight": 0.9, "confidence": 0.88,
                 "provenance": "Ontology: 'family car' → 5-star safety"},
                {"field": "body_type", "value": "minivan/SUV", "weight": 0.8, "confidence": 0.78,
                 "provenance": "Ontology: 'family car' → minivan or SUV"},
            ]
        )

        # ── RESTAURANTS ───────────────────────────────────────────────
        self.register_rule(
            domain="restaurants", keyword="date night",
            inferred_preferences=[
                {"field": "ambiance", "value": "romantic/dim lighting", "weight": 0.9, "confidence": 0.85,
                 "provenance": "Ontology: 'date night' → romantic ambiance"},
                {"field": "noise_level", "value": "quiet", "weight": 0.8, "confidence": 0.78,
                 "provenance": "Ontology: 'date night' → quiet atmosphere"},
                {"field": "reservation", "value": "required/recommended", "weight": 0.7, "confidence": 0.7,
                 "provenance": "Ontology: 'date night' → reservation likely needed"},
            ]
        )
        self.register_rule(
            domain="restaurants", keyword="vegan",
            inferred_filters=[
                {"field": "diet", "operator": "eq", "value": "vegan",
                 "confidence": 0.95, "provenance": "Ontology: 'vegan' → vegan menu filter"},
            ]
        )
        self.register_rule(
            domain="restaurants", keyword="family",
            inferred_preferences=[
                {"field": "kids_menu", "value": True, "weight": 0.85, "confidence": 0.82,
                 "provenance": "Ontology: 'family' → kids menu"},
                {"field": "noise_level", "value": "lively/casual", "weight": 0.75, "confidence": 0.72,
                 "provenance": "Ontology: 'family' → casual atmosphere"},
            ]
        )

        # ── MOVIES ────────────────────────────────────────────────────
        self.register_rule(
            domain="movies", keyword="date night",
            inferred_preferences=[
                {"field": "genre", "value": "romance/comedy", "weight": 0.8, "confidence": 0.75,
                 "provenance": "Ontology: 'date night' → romance or comedy"},
            ]
        )
        self.register_rule(
            domain="movies", keyword="kids",
            inferred_preferences=[
                {"field": "rating", "value": "G/PG", "weight": 0.95, "confidence": 0.92,
                 "provenance": "Ontology: 'kids' → G/PG rating"},
                {"field": "genre", "value": "animation/family", "weight": 0.85, "confidence": 0.82,
                 "provenance": "Ontology: 'kids' → animation or family genre"},
            ]
        )
        self.register_rule(
            domain="movies", keyword="binge",
            inferred_preferences=[
                {"field": "episodes", "value": "10+", "weight": 0.8, "confidence": 0.75,
                 "provenance": "Ontology: 'binge' → multi-episode series"},
                {"field": "type", "value": "series", "weight": 0.85, "confidence": 0.82,
                 "provenance": "Ontology: 'binge' → series format"},
            ]
        )

        # ── HEALTHCARE ────────────────────────────────────────────────
        self.register_rule(
            domain="healthcare", keyword="anxiety",
            inferred_preferences=[
                {"field": "specialty", "value": "therapist/psychiatrist", "weight": 0.9, "confidence": 0.88,
                 "provenance": "Ontology: 'anxiety' → mental health specialist"},
                {"field": "modality", "value": "CBT/online therapy", "weight": 0.75, "confidence": 0.72,
                 "provenance": "Ontology: 'anxiety' → CBT recommended"},
            ]
        )
        self.register_rule(
            domain="healthcare", keyword="teeth",
            inferred_filters=[
                {"field": "specialty", "operator": "eq", "value": "dentist",
                 "confidence": 0.95, "provenance": "Ontology: 'teeth' → dentist"},
            ]
        )
        self.register_rule(
            domain="healthcare", keyword="heart",
            inferred_preferences=[
                {"field": "specialty", "value": "cardiologist", "weight": 0.9, "confidence": 0.88,
                 "provenance": "Ontology: 'heart' → cardiologist"},
            ]
        )

        # ── COURSES ───────────────────────────────────────────────────
        self.register_rule(
            domain="courses", keyword="machine learning",
            inferred_preferences=[
                {"field": "skills", "value": "Python/NumPy/Pandas", "weight": 0.85, "confidence": 0.85,
                 "provenance": "Ontology: 'ML course' → Python/Pandas prerequisites"},
                {"field": "difficulty", "value": "intermediate/advanced", "weight": 0.75, "confidence": 0.72,
                 "provenance": "Ontology: 'ML' → intermediate or advanced level"},
            ]
        )
        self.register_rule(
            domain="courses", keyword="beginner",
            inferred_filters=[
                {"field": "difficulty", "operator": "eq", "value": "beginner",
                 "confidence": 0.92, "provenance": "Ontology: 'beginner' → beginner difficulty"},
            ]
        )
        self.register_rule(
            domain="courses", keyword="certification",
            inferred_preferences=[
                {"field": "certificate", "value": "included", "weight": 0.9, "confidence": 0.88,
                 "provenance": "Ontology: 'certification' → certificate of completion"},
            ]
        )
        self.register_rule(
            domain="courses", keyword="web development",
            inferred_preferences=[
                {"field": "skills", "value": "HTML/CSS/JavaScript", "weight": 0.85, "confidence": 0.85,
                 "provenance": "Ontology: 'web dev' → HTML/CSS/JS skills"},
                {"field": "skills", "value": "React or Vue", "weight": 0.7, "confidence": 0.7,
                 "provenance": "Ontology: 'web dev' → modern framework"},
            ]
        )
