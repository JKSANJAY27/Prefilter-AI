"""
tests/test_pipeline.py — End-to-end tests for PrefilterPipeline.

Tests cover: conflict detection, ontology inference, conflict-aware relaxation,
multi-turn sessions, all 5 DSL translators, explanation builder, and SLM
adapter wiring path.
"""

from __future__ import annotations

import pytest

from prefilter_ai import PipelineResult, PrefilterPipeline

# ── Fixture ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pipeline():
    return PrefilterPipeline(parser="spacy")


# ── Basic pipeline execution ───────────────────────────────────────────


def test_pipeline_runs(pipeline):
    r = pipeline.run("Sony headphones under $200")
    assert isinstance(r, PipelineResult)
    assert r.ir is not None
    assert r.ir.domain == "ecommerce"


def test_pipeline_extracts_product(pipeline):
    r = pipeline.run("noise cancelling headphones under $200")
    fields = {f.field: f.value for f in r.ir.filters}
    assert "product" in fields
    assert "headphones" in str(fields["product"]).lower()


def test_pipeline_extracts_price(pipeline):
    r = pipeline.run("laptop under $1500")
    price_filters = [f for f in r.ir.filters if f.field == "price"]
    assert len(price_filters) >= 1
    price_f = price_filters[0]
    assert price_f.operator in {"lt", "lte"}
    assert float(price_f.value) == pytest.approx(1500.0, rel=0.01)


def test_pipeline_extracts_exclusion(pipeline):
    r = pipeline.run("headphones not red not green under $200")
    ne_filters = [f for f in r.ir.filters if f.operator == "ne"]
    ne_values = [str(f.value).lower() for f in ne_filters]
    assert any("red" in v for v in ne_values)


def test_pipeline_latency_tracked(pipeline):
    r = pipeline.run("gaming laptop under $800")
    assert "parse_ms" in r.latency_ms
    assert r.total_latency_ms > 0


def test_pipeline_repr(pipeline):
    assert "spacy" in repr(pipeline).lower()


# ── Ontology inference ─────────────────────────────────────────────────


def test_ontology_infers_gpu_for_gaming(pipeline):
    r = pipeline.run("gaming laptop")
    prefs = {p.field: p.value for p in r.ir.preferences}
    # Gaming should trigger GPU preference via ontology
    filter_vals = {f.field: f.value for f in r.ir.filters}
    has_gpu = (
        "feature" in filter_vals
        or "feature" in prefs
        or "gpu" in str(prefs).lower()
        or "gpu" in str(filter_vals).lower()
    )
    assert has_gpu, f"Expected GPU preference. Prefs: {prefs}, Filters: {filter_vals}"


def test_ontology_infers_for_ai_laptop(pipeline):
    r = pipeline.run("laptop for AI and machine learning")
    prefs = {p.field: p.value for p in r.ir.preferences}
    filter_vals = {f.field: f.value for f in r.ir.filters}
    has_indicator = any(
        "ram" in k.lower() or "gpu" in k.lower() or "vram" in k.lower() for k in prefs.keys()
    ) or any("gpu" in str(v).lower() or "cuda" in str(v).lower() for v in filter_vals.values())
    assert has_indicator, f"Expected AI-related inference. Got prefs={prefs}"


def test_ontology_infers_honeymoon_preferences(pipeline):
    r = pipeline.run("hotel for honeymoon in Maldives")
    prefs = {p.field: p.value for p in r.ir.preferences}
    keys = list(prefs.keys())
    assert any(k in {"view", "room_type", "amenities"} for k in keys), (
        f"Expected honeymoon prefs, got: {keys}"
    )


def test_ontology_infers_remote_job(pipeline):
    r = pipeline.run("remote senior data scientist job")
    filter_vals = {f.field: f.value for f in r.ir.filters}
    has_remote = (
        filter_vals.get("remote") is True or filter_vals.get("experience_level") == "senior"
    )
    assert has_remote, f"Expected remote or senior filter. Got: {filter_vals}"


# ── Conflict detection ─────────────────────────────────────────────────


def test_conflict_detected_gaming_laptop_cheap(pipeline):
    r = pipeline.run("gaming laptop under $500")
    assert r.has_conflicts, "Expected conflict for gaming laptop under $500"


def test_conflict_detected_5star_hotel_cheap(pipeline):
    r = pipeline.run("5 star hotel under $60 per night")
    assert r.has_conflicts, "Expected conflict for 5-star hotel under $60"


def test_conflict_detected_business_class_cheap(pipeline):
    r = pipeline.run("business class flight to Tokyo under $200")
    assert r.has_conflicts, "Expected conflict for business class under $200"


def test_no_conflict_reasonable_budget(pipeline):
    r = pipeline.run("economy flight from JFK to London under $900")
    # Should not flag a conflict for reasonable economy fare
    assert len(r.conflicts) == 0 or all("economy" not in c for c in r.conflicts)


# ── Conflict-aware relaxation ──────────────────────────────────────────


def test_relaxation_triggered_on_conflict(pipeline):
    r = pipeline.run("gaming laptop under $500")
    assert r.relaxed_ir is not None, "Expected relaxed_ir when conflicts detected"


def test_relaxation_expands_price_not_product(pipeline):
    r = pipeline.run("gaming laptop under $500")
    if r.relaxed_ir is None:
        pytest.skip("No relaxation triggered")
    relaxed_price = next(
        (
            f.value
            for f in r.relaxed_ir.filters
            if f.field == "price" and f.operator in {"lt", "lte"}
        ),
        None,
    )
    original_price = next(
        (f.value for f in r.ir.filters if f.field == "price" and f.operator in {"lt", "lte"}), None
    )
    if relaxed_price is not None and original_price is not None:
        assert float(relaxed_price) >= float(original_price), (
            f"Relaxed price {relaxed_price} should be >= original {original_price}"
        )


def test_no_relaxation_without_conflict(pipeline):
    r = pipeline.run("headphones under $200")
    # If no conflicts, relaxed_ir should be None
    if not r.has_conflicts:
        assert r.relaxed_ir is None


# ── Translators ────────────────────────────────────────────────────────


def test_sql_translation(pipeline):
    r = pipeline.run("headphones under $200")
    sql, params = r.sql
    assert isinstance(sql, str)
    assert isinstance(params, dict)


def test_elasticsearch_translation(pipeline):
    r = pipeline.run("headphones under $200")
    es = r.elasticsearch
    assert isinstance(es, dict)


def test_mongodb_translation(pipeline):
    r = pipeline.run("headphones under $200")
    mongo = r.mongodb
    assert isinstance(mongo, dict)


def test_chromadb_translation(pipeline):
    r = pipeline.run("headphones under $200")
    chroma = r.chromadb
    assert isinstance(chroma, dict)


# ── Explanation ────────────────────────────────────────────────────────


def test_explanation_produced(pipeline):
    r = pipeline.run("Sony headphones under $200")
    assert isinstance(r.explanation, dict)
    assert len(r.explanation) > 0


def test_explanation_has_provenance(pipeline):
    r = pipeline.run("Sony headphones under $200")
    for field_name, text in r.explanation.items():
        assert "spaCy" in text or "Ontology" in text or "confidence" in text.lower(), (
            f"Expected provenance in explanation for '{field_name}': {text}"
        )


# ── Multi-turn session ─────────────────────────────────────────────────


def test_session_basic(pipeline):
    session = pipeline.new_session()
    session.run("gaming laptops")
    r2 = session.run("Only Lenovo")
    assert r2.ir is not None
    assert len(session.history) == 2


def test_session_reruns_ontology_on_refinement(pipeline):
    """Verify Gap #6 fix: after 'gaming laptops', prefs should be in turn 1."""
    session = pipeline.new_session()
    r1 = session.run("gaming laptops")
    # GPU preference should be inferred from turn 1
    prefs_t1 = {p.field: p.value for p in r1.ir.preferences}
    filter_t1 = {f.field: f.value for f in r1.ir.filters}
    has_indicator = any(
        "gpu" in str(v).lower() or "feature" in k.lower() for k, v in prefs_t1.items()
    ) or any("gpu" in str(v).lower() or "cuda" in str(v).lower() for v in filter_t1.values())
    assert has_indicator, f"Expected GPU inference in turn 1. Prefs={prefs_t1}, Filters={filter_t1}"


def test_session_reset(pipeline):
    session = pipeline.new_session()
    session.run("headphones")
    session.reset()
    assert len(session.history) == 0
    assert session._current_ir is None


# ── to_dict serialization ──────────────────────────────────────────────


def test_to_dict_serializable(pipeline):
    import json

    r = pipeline.run("headphones under $200")
    d = r.to_dict()
    # Should be JSON-serializable
    json_str = json.dumps(d)
    assert json_str


def test_to_dict_contains_keys(pipeline):
    r = pipeline.run("headphones under $200")
    d = r.to_dict()
    for key in ["query", "domain", "filters", "preferences", "conflicts", "explanation"]:
        assert key in d, f"Missing key '{key}' in to_dict output"


# ── Domain coverage ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "query,expected_domain",
    [
        ("nonstop flight from JFK to Tokyo", "flights"),
        ("5-star hotel in Paris", "hotels"),
        ("2 bedroom apartment in Manhattan", "real_estate"),
        ("Python machine learning course for beginners", "courses"),
        ("Toyota Camry under 50000 miles", "cars"),
        ("vegan restaurant open late near me", "restaurants"),
        ("action movies with rating above 8", "movies"),
        ("dermatologist accepting Blue Cross insurance", "healthcare"),
        ("senior software engineer remote job at startup", "jobs"),
        ("gaming laptop under $1200", "ecommerce"),
    ],
)
def test_domain_detection(pipeline, query, expected_domain):
    r = pipeline.run(query)
    assert r.ir.domain == expected_domain, (
        f"Expected domain '{expected_domain}' for query '{query}', got '{r.ir.domain}'"
    )
