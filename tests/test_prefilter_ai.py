"""
Tests for prefilter_ai.

Run with:
    pytest tests/ -v

The model-loading tests are skipped in CI (no GPU) unless
PREFILTER_AI_RUN_MODEL_TESTS=1 is set in the environment.
"""

from __future__ import annotations

import json
import os

import pytest
import yaml

from prefilter_ai import ModelFormat, PrefilterAI
from prefilter_ai.config import build_inference_prompt, get_system_prompt
from prefilter_ai.exceptions import ParseError
from prefilter_ai.parser import parse_model_output
from prefilter_ai.result import ParseResult

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def sample_json_fields():
    return {
        "domain": "ecommerce",
        "product": "headphones",
        "price": "lt:200",
        "feature": "noise cancelling",
    }


@pytest.fixture
def sample_json_result(sample_json_fields):
    return ParseResult(
        query="noise cancelling headphones under $200",
        fields=sample_json_fields,
        raw_output=json.dumps(sample_json_fields),
        model_format="json",
    )


# ── Config tests ───────────────────────────────────────────────


def test_system_prompt_contains_format_name():
    assert "JSON" in get_system_prompt(ModelFormat.JSON)
    assert "YAML" in get_system_prompt(ModelFormat.YAML)


def test_inference_prompt_structure():
    prompt = build_inference_prompt("find a laptop", ModelFormat.JSON)
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "<|im_start|>assistant" in prompt
    assert "find a laptop" in prompt


def test_model_format_from_string():
    assert ModelFormat("json") == ModelFormat.JSON
    assert ModelFormat("yaml") == ModelFormat.YAML


# ── Parser tests ───────────────────────────────────────────────


class TestJsonParser:
    def test_clean_json(self):
        raw = '{"domain": "ecommerce", "product": "laptop"}'
        result = parse_model_output(raw, ModelFormat.JSON)
        assert result == {"domain": "ecommerce", "product": "laptop"}

    def test_json_with_whitespace(self):
        raw = '  \n{"domain": "jobs", "job_title": "ML Engineer"}\n  '
        result = parse_model_output(raw, ModelFormat.JSON)
        assert result["domain"] == "jobs"

    def test_json_embedded_in_prose(self):
        raw = 'Sure! Here is the JSON: {"domain": "flights", "origin": "JFK"}'
        result = parse_model_output(raw, ModelFormat.JSON)
        assert result["origin"] == "JFK"

    def test_invalid_json_raises(self):
        with pytest.raises(ParseError) as exc_info:
            parse_model_output("this is not json at all", ModelFormat.JSON)
        assert exc_info.value.raw_output == "this is not json at all"

    def test_operator_value_preserved(self):
        raw = '{"price": "lt:2000", "rating": "gte:4.5"}'
        result = parse_model_output(raw, ModelFormat.JSON)
        assert result["price"] == "lt:2000"
        assert result["rating"] == "gte:4.5"

    def test_json_truncated_repair(self):
        raw = '{"domain": "flights", "origin": "JFK"'
        result = parse_model_output(raw, ModelFormat.JSON)
        assert result == {"domain": "flights", "origin": "JFK"}


class TestYamlParser:
    def test_clean_yaml(self):
        raw = "domain: ecommerce\nproduct: laptop\nprice: lt:2000"
        result = parse_model_output(raw, ModelFormat.YAML)
        assert result == {
            "domain": "ecommerce",
            "product": "laptop",
            "price": "lt:2000",
        }

    def test_yaml_with_code_fence(self):
        raw = "```yaml\ndomain: hotels\ncity: Paris\n```"
        result = parse_model_output(raw, ModelFormat.YAML)
        assert result["city"] == "Paris"

    def test_invalid_yaml_raises(self):
        with pytest.raises(ParseError):
            parse_model_output(": : : broken yaml :", ModelFormat.YAML)


# ── ParseResult tests ──────────────────────────────────────────


class TestParseResult:
    def test_to_json(self, sample_json_result):
        out = sample_json_result.to_json()
        parsed = json.loads(out)
        assert parsed["domain"] == "ecommerce"

    def test_to_json_indent(self, sample_json_result):
        out = sample_json_result.to_json(indent=2)
        assert "\n" in out  # indented means newlines

    def test_to_yaml(self, sample_json_result):
        out = sample_json_result.to_yaml()
        parsed = yaml.safe_load(out)
        assert parsed["product"] == "headphones"

    def test_to_dict(self, sample_json_result):
        d = sample_json_result.to_dict()
        assert "query" in d
        assert "fields" in d
        assert "model_format" in d

    def test_getitem(self, sample_json_result):
        assert sample_json_result["domain"] == "ecommerce"

    def test_contains(self, sample_json_result):
        assert "product" in sample_json_result
        assert "nonexistent_field" not in sample_json_result

    def test_get_numeric_constraint_lt(self, sample_json_result):
        c = sample_json_result.get_numeric_constraint("price")
        assert c == {"operator": "lt", "value": 200.0, "value_hi": None}

    def test_get_numeric_constraint_between(self):
        result = ParseResult(
            query="jobs paying $80k-$120k",
            fields={"salary": "between:80000:120000"},
            raw_output="",
            model_format="json",
        )
        c = result.get_numeric_constraint("salary")
        assert c == {"operator": "between", "value": 80000.0, "value_hi": 120000.0}

    def test_get_numeric_constraint_missing(self, sample_json_result):
        assert sample_json_result.get_numeric_constraint("nonexistent") is None

    def test_get_numeric_constraint_plain_string(self, sample_json_result):
        assert sample_json_result.get_numeric_constraint("domain") is None

    def test_numeric_fields(self, sample_json_result):
        nf = sample_json_result.numeric_fields()
        assert "price" in nf
        assert "domain" not in nf
        assert "product" not in nf

    def test_repr(self, sample_json_result):
        r = repr(sample_json_result)
        assert "ParseResult" in r
        assert "ecommerce" in r

    def test_translators(self):
        result = ParseResult(
            query="test",
            fields={
                "domain": "ecommerce",
                "brand": "Sony",
                "price": "lt:200",
                "color": ["ne:red", "ne:green"],
                "rating": "between:4.0:5.0"
            },
            raw_output="",
            model_format="json"
        )
        # Test SQL
        sql, params = result.to_sql("products")
        assert "SELECT * FROM products WHERE brand = :brand_0" in sql
        assert "price < :price_1" in sql
        assert "color != :color_ne_2" in sql
        assert params["brand_0"] == "Sony"
        assert params["price_1"] == 200.0

        # Test MongoDB
        mongo = result.to_mongodb()
        assert mongo["brand"] == "Sony"
        assert mongo["price"] == {"$lt": 200.0}
        assert mongo["color"] == {"$nin": ["red", "green"]}

        # Test ChromaDB
        chroma = result.to_chromadb()
        assert {"brand": {"$eq": "Sony"}} in chroma["$and"]

        # Test Elasticsearch
        es = result.to_elasticsearch()
        assert es["query"]["bool"]["must"][0]["term"]["brand"] == "Sony"


class TestMiddlewarePipeline:
    def test_registry(self):
        from prefilter_ai.registry import SchemaRegistry, DataType, Importance
        reg = SchemaRegistry()
        assert "ecommerce" in reg.list_domains()
        
        schema = reg.get("ecommerce")
        assert schema is not None
        assert schema.fields["price"].data_type == DataType.NUMBER
        assert schema.fields["price"].importance == Importance.HIGH

    def test_ontology(self):
        from prefilter_ai.ontology import OntologyEngine
        from prefilter_ai.ir import IntermediateRepresentation
        
        ir = IntermediateRepresentation(domain="ecommerce")
        ir = OntologyEngine().infer(ir, "laptop for AI and coding")
        
        # Check inferred filters
        feature_filters = [f for f in ir.filters if f.field == "feature"]
        assert len(feature_filters) > 0
        assert feature_filters[0].value == "CUDA/RTX GPU"
        
        # Check inferred preferences
        ram_prefs = [p for p in ir.preferences if p.field == "ram"]
        assert len(ram_prefs) > 0
        assert ram_prefs[0].value == "16GB+"
        assert "AI" in ram_prefs[0].provenance  # provenance mentions 'AI' keyword

    def test_conflict_detection(self):
        from prefilter_ai.validator import ConflictDetector
        from prefilter_ai.ir import IntermediateRepresentation
        
        detector = ConflictDetector()
        
        # Case A: Mathematical pricing contradiction
        ir = IntermediateRepresentation()
        ir.add_filter("price", "lt", 100)
        ir.add_filter("price", "gt", 200)
        detector.validate(ir)
        assert len(ir.conflicts) > 0
        assert "Contradictory numerical constraints" in ir.conflicts[0]

        # Case B: Feasibility contradiction
        ir_laptop = IntermediateRepresentation(domain="ecommerce")
        ir_laptop.add_filter("feature", "eq", "RTX 4080")
        ir_laptop.add_filter("price", "lt", 400)
        detector.validate(ir_laptop)
        assert len(ir_laptop.conflicts) > 0
        assert "Feasibility conflict" in ir_laptop.conflicts[0]

    def test_query_relaxation(self):
        from prefilter_ai.relaxer import QueryRelaxer
        from prefilter_ai.ir import IntermediateRepresentation
        
        relaxer = QueryRelaxer()
        ir = IntermediateRepresentation(domain="ecommerce")
        ir.add_filter("brand", "eq", "Sony")
        ir.add_filter("price", "lt", 200)
        ir.add_filter("color", "ne", "black") # LOW importance
        
        # Level 1 should drop color
        relaxed_1 = relaxer.relax(ir, relaxation_level=1)
        assert len(relaxed_1.filters) == 2
        assert not any(f.field == "color" for f in relaxed_1.filters)
        
        # Level 2 should drop color + expand price ceiling
        relaxed_2 = relaxer.relax(ir, relaxation_level=2)
        price_f = [f for f in relaxed_2.filters if f.field == "price"][0]
        assert price_f.value == 250.0 # 200 * 1.25

    def test_stateful_session_and_diff(self):
        from prefilter_ai.history import PrefilterSession
        from prefilter_ai.expert import PrefilterAI
        
        expert = PrefilterAI(parse_backend="spacy")
        session = PrefilterSession(parser=expert.parser)
        
        # Turn 1
        ir = session.process_query("Sony headphones under $200")
        assert any(f.field == "brand" and f.value == "Sony" for f in ir.filters)
        assert any(f.field == "price" and f.value == 200.0 for f in ir.filters)
        
        # Turn 2: refinement query
        ir_ref = session.process_query("Actually Apple headphones")
        # Should override brand Sony to Apple
        assert any(f.field == "brand" and f.value == "Apple" for f in ir_ref.filters)
        assert not any(f.field == "brand" and f.value == "Sony" for f in ir_ref.filters)
        # Should preserve price
        assert any(f.field == "price" and f.value == 200.0 for f in ir_ref.filters)

        # Turn 3: "cheaper" refinement
        ir_cheap = session.process_query("make it cheaper")
        price_val = [f.value for f in ir_cheap.filters if f.field == "price"][0]
        assert price_val == 160.0 # 200 * 0.80

    def test_evaluation_harness(self):
        from prefilter_ai.evaluation import EvaluationHarness
        harness = EvaluationHarness()
        
        dataset = [
            {
                "query": "Sony headphones under $200",
                "ground_truth": {
                    "domain": "ecommerce",
                    "brand": "Sony",
                    "price": "lt:200"
                }
            }
        ]
        summary = harness.evaluate_dataset(dataset)
        assert summary["metrics"]["avg_f1"] > 0.0
        assert summary["metrics"]["avg_latency_ms"] > 0.0



# ── Integration test (skipped in CI without GPU) ───────────────

SKIP_MODEL = not os.environ.get("PREFILTER_AI_RUN_MODEL_TESTS")


@pytest.mark.skipif(
    SKIP_MODEL, reason="Set PREFILTER_AI_RUN_MODEL_TESTS=1 to run model tests"
)
class TestPrefilterAIIntegration:
    @pytest.fixture(scope="class")
    def expert_json(self):
        return PrefilterAI(fmt=ModelFormat.JSON, eager=True)

    @pytest.fixture(scope="class")
    def expert_yaml(self):
        return PrefilterAI(fmt=ModelFormat.YAML, eager=True)

    def test_json_parse_returns_result(self, expert_json):
        result = expert_json.parse("MacBook under $2000")
        assert isinstance(result, ParseResult)
        assert "domain" in result.fields

    def test_yaml_parse_returns_result(self, expert_yaml):
        result = expert_yaml.parse("3 star hotel in Paris under $150 per night")
        assert isinstance(result, ParseResult)

    def test_numeric_operator_in_output(self, expert_json):
        result = expert_json.parse("headphones under $200")
        price_constraint = result.get_numeric_constraint("price")
        assert price_constraint is not None
        assert price_constraint["operator"] in {"lt", "lte", "approx", "between"}

    def test_empty_query_raises(self, expert_json):
        with pytest.raises(ValueError):
            expert_json.parse("")

    def test_batch_parse(self, expert_json):
        queries = [
            "Python course for beginners under $50",
            "Remote ML engineer job over $150k",
        ]
        results = expert_json.parse_batch(queries)
        assert len(results) == 2
        assert all(isinstance(r, ParseResult) for r in results)
