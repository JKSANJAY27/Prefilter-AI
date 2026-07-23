"""
pipeline.py — PrefilterPipeline: the unified public API for Prefilter AI.

Wires all 7 middleware stages into a single call:
  1. Parser       — NL → raw constraints (spaCy / SLM / Gemini)
  2. Ontology     — implicit preference inference
  3. Validator    — conflict & feasibility detection
  4. Relaxer      — conflict-aware constraint relaxation
  5. Translators  — IR → SQL / ES / MongoDB / ChromaDB DSL
  6. Explainer    — per-field provenance explanation
  7. Evaluation   — latency profiling per stage

Usage
-----
    from prefilter_ai import PrefilterPipeline

    pipeline = PrefilterPipeline(parser="spacy")
    result = pipeline.run("gaming laptop under $800")

    result.ir               # IntermediateRepresentation
    result.relaxed_ir       # Relaxed IR if conflicts detected
    result.conflicts        # List of conflict messages
    result.sql              # SQL WHERE clause string
    result.elasticsearch    # Elasticsearch DSL dict
    result.mongodb          # MongoDB filter dict
    result.chromadb         # ChromaDB metadata dict
    result.explanation      # Per-field provenance dict
    result.latency_ms       # Stage-by-stage latency dict
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any

from prefilter_ai.ir import IntermediateRepresentation
from prefilter_ai.ontology import OntologyEngine
from prefilter_ai.validator import ConflictDetector
from prefilter_ai.relaxer import QueryRelaxer
from prefilter_ai.registry import SchemaRegistry
from prefilter_ai.utils import build_explanation

logger = logging.getLogger(__name__)


# ── Pipeline Result ────────────────────────────────────────────────────


@dataclass
class PipelineResult:
    """
    The complete output of a PrefilterPipeline.run() call.

    Contains every intermediate and final artifact produced by the
    middleware pipeline, including the IR, conflict list, DSL translations,
    per-field explanations, and stage latency breakdown.
    """
    query: str
    ir: IntermediateRepresentation
    relaxed_ir: IntermediateRepresentation | None = None
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    explanation: dict[str, str] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    parser_backend: str = "spacy"

    # ── Backend DSL translations (lazy-computed) ───────────────────────

    _sql: tuple[str, dict] | None = field(default=None, repr=False)
    _elasticsearch: dict | None = field(default=None, repr=False)
    _mongodb: dict | None = field(default=None, repr=False)
    _chromadb: dict | None = field(default=None, repr=False)

    @property
    def sql(self) -> tuple[str, dict[str, Any]]:
        """Return (WHERE clause, params) for SQL backends."""
        if self._sql is None:
            from prefilter_ai.translators.sql import SQLTranslator
            self._sql = SQLTranslator().translate(self.ir)
        return self._sql

    @property
    def elasticsearch(self) -> dict[str, Any]:
        """Return Elasticsearch query DSL."""
        if self._elasticsearch is None:
            from prefilter_ai.translators.elasticsearch import ElasticsearchTranslator
            self._elasticsearch = ElasticsearchTranslator().translate(self.ir)
        return self._elasticsearch

    @property
    def mongodb(self) -> dict[str, Any]:
        """Return MongoDB filter dict."""
        if self._mongodb is None:
            from prefilter_ai.translators.mongodb import MongoDBTranslator
            self._mongodb = MongoDBTranslator().translate(self.ir)
        return self._mongodb

    @property
    def chromadb(self) -> dict[str, Any]:
        """Return ChromaDB where-metadata dict."""
        if self._chromadb is None:
            from prefilter_ai.translators.chromadb import ChromaDBTranslator
            self._chromadb = ChromaDBTranslator().translate(self.ir)
        return self._chromadb

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0

    @property
    def total_latency_ms(self) -> float:
        return sum(self.latency_ms.values())

    def execute(self, connector: Any, **kwargs: Any) -> Any:
        """
        Execute the translated query directly against a target database connector.

        Parameters
        ----------
        connector : BaseConnector
            Instance of SQLConnector, MongoConnector, ElasticsearchConnector, or ChromaDBConnector.
        **kwargs:
            Connector-specific execution arguments (limit, select_fields, etc.).
        """
        from prefilter_ai.connectors import (
            SQLConnector,
            MongoConnector,
            ElasticsearchConnector,
            ChromaDBConnector,
        )

        if isinstance(connector, SQLConnector):
            return connector.execute(self.sql, **kwargs)
        elif isinstance(connector, MongoConnector):
            return connector.execute(self.mongodb, **kwargs)
        elif isinstance(connector, ElasticsearchConnector):
            return connector.execute(self.elasticsearch, **kwargs)
        elif isinstance(connector, ChromaDBConnector):
            return connector.execute(self.chromadb, **kwargs)
        elif hasattr(connector, "execute"):
            return connector.execute(self, **kwargs)
        else:
            raise ValueError(f"Unsupported connector type: {type(connector)}")


    def to_dict(self) -> dict[str, Any]:
        """Serialize the full result as a plain dict (JSON-safe)."""
        return {
            "query": self.query,
            "parser_backend": self.parser_backend,
            "domain": self.ir.domain,
            "intent": self.ir.intent,
            "filters": [f.to_dict() for f in self.ir.filters],
            "preferences": [p.to_dict() for p in self.ir.preferences],
            "conflicts": self.conflicts,
            "warnings": self.warnings,
            "explanation": self.explanation,
            "latency_ms": self.latency_ms,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "relaxed": {
                "filters": [f.to_dict() for f in self.relaxed_ir.filters],
                "warnings": self.relaxed_ir.warnings,
            } if self.relaxed_ir else None,
            "sql": self.sql[0] if self.sql else None,
            "elasticsearch": self.elasticsearch,
            "mongodb": self.mongodb,
            "chromadb": self.chromadb,
        }

    def __repr__(self) -> str:
        return (
            f"PipelineResult(domain={self.ir.domain!r}, "
            f"filters={len(self.ir.filters)}, "
            f"conflicts={len(self.conflicts)}, "
            f"latency={self.total_latency_ms:.1f}ms)"
        )


# ── Pipeline ───────────────────────────────────────────────────────────


class PrefilterPipeline:
    """
    Complete AI Query Understanding Middleware pipeline.

    Stages
    ------
    1. Parser       — NL → structured IR constraints
    2. Ontology     — implicit preference inference for the detected domain
    3. Validator    — conflict detection + feasibility analysis
    4. Relaxer      — conflict-targeted constraint relaxation
    5. Translators  — IR → multi-backend DSL (lazy, on property access)
    6. Explainer    — per-field provenance explanation
    7. Latency      — per-stage timing metrics

    Parameters
    ----------
    parser : str
        Which parser backend to use: ``"spacy"`` (default, CPU, fast),
        ``"slm"`` (local fine-tuned Qwen 0.8B, requires torch+peft),
        or ``"gemini"`` (API-based, requires GEMINI_API_KEY).
    spacy_model : str
        spaCy model name. Only used when parser="spacy".
    auto_relax : bool
        If True (default), automatically produce a relaxed IR whenever
        conflicts are detected. Access via result.relaxed_ir.
    gemini_api_key : str | None
        API key for Gemini parser. Falls back to GEMINI_API_KEY env var.
    slm_fmt : str
        Format for the SLM adapter: "json" (default) or "yaml".
    """

    def __init__(
        self,
        parser: str = "spacy",
        spacy_model: str = "en_core_web_sm",
        auto_relax: bool = True,
        gemini_api_key: str | None = None,
        slm_fmt: str = "json",
        slm_model_id: str | None = None,
        slm_load_in_4bit: bool = True,
    ) -> None:
        self.parser_name = parser.lower()
        self.auto_relax = auto_relax

        # Lazy-initialised components (shared singletons)
        self._parser_instance = None
        self._spacy_model = spacy_model
        self._gemini_api_key = gemini_api_key
        self._slm_fmt = slm_fmt
        self._slm_model_id = slm_model_id
        self._slm_load_in_4bit = slm_load_in_4bit

        self._ontology = OntologyEngine()
        self._validator = ConflictDetector()
        self._relaxer = QueryRelaxer(registry=SchemaRegistry())

    @property
    def _parser(self):
        if self._parser_instance is None:
            self._parser_instance = self._build_parser()
        return self._parser_instance

    def _build_parser(self):
        from prefilter_ai.parser_interface import SpacyParser, SLMParser, GeminiParser
        if self.parser_name == "spacy":
            return SpacyParser(spacy_model=self._spacy_model)
        elif self.parser_name == "slm":
            from prefilter_ai.config import ModelFormat
            return SLMParser(
                fmt=ModelFormat(self._slm_fmt),
                model_id=self._slm_model_id,
                load_in_4bit=self._slm_load_in_4bit,
            )
        elif self.parser_name == "gemini":
            return GeminiParser(api_key=self._gemini_api_key)
        else:
            raise ValueError(
                f"Unknown parser backend '{self.parser_name}'. "
                "Choose from: 'spacy', 'slm', 'gemini'."
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, query: str) -> PipelineResult:
        """
        Run the full query understanding pipeline on a natural language query.

        Parameters
        ----------
        query : str
            The user's natural language search query.

        Returns
        -------
        PipelineResult
            Rich result object with IR, conflicts, DSL translations,
            explanations, and per-stage latency.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        latency: dict[str, float] = {}

        # ── Stage 1: Parse ─────────────────────────────────────────────
        t0 = time.perf_counter()
        ir = self._parser.parse(query)
        ir.metadata["query_text"] = query
        latency["parse_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        logger.debug("Stage 1 (parse): %s ms, domain=%s", latency["parse_ms"], ir.domain)

        # ── Stage 2: Ontology inference ────────────────────────────────
        t0 = time.perf_counter()
        ir = self._ontology.infer(ir, query)
        latency["ontology_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        logger.debug("Stage 2 (ontology): %s ms, preferences=%d", latency["ontology_ms"], len(ir.preferences))

        # ── Stage 3: Conflict & feasibility validation ─────────────────
        t0 = time.perf_counter()
        ir = self._validator.validate(ir)
        latency["validate_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        logger.debug("Stage 3 (validate): %s ms, conflicts=%d", latency["validate_ms"], len(ir.conflicts))

        # ── Stage 4: Conflict-aware relaxation ─────────────────────────
        relaxed_ir = None
        if self.auto_relax and ir.conflicts:
            t0 = time.perf_counter()
            relaxed_ir = self._relaxer.relax_from_conflicts(ir)
            latency["relax_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            logger.debug("Stage 4 (relax): %s ms", latency["relax_ms"])

        # ── Stage 5: Explanation ───────────────────────────────────────
        t0 = time.perf_counter()
        explanation = build_explanation(ir)
        latency["explain_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        return PipelineResult(
            query=query,
            ir=ir,
            relaxed_ir=relaxed_ir,
            conflicts=list(ir.conflicts),
            warnings=list(ir.warnings),
            explanation=explanation,
            latency_ms=latency,
            parser_backend=self.parser_name,
        )

    # ------------------------------------------------------------------
    # Convenience: run a session (multi-turn)
    # ------------------------------------------------------------------

    def new_session(self) -> "PipelineSession":
        """Create a new stateful conversational search session."""
        return PipelineSession(pipeline=self)

    def __repr__(self) -> str:
        return f"PrefilterPipeline(parser={self.parser_name!r}, auto_relax={self.auto_relax})"


# ── Stateful Session ───────────────────────────────────────────────────


class PipelineSession:
    """
    Multi-turn conversational search session.

    Wraps PrefilterSession to run the full pipeline (ontology + validator +
    relaxer) after every conversational refinement, fixing Gap #6.

    Usage
    -----
        session = pipeline.new_session()
        r1 = session.run("gaming laptops")
        r2 = session.run("Only Lenovo")       # merges + re-runs pipeline
        r3 = session.run("Actually under $1200")
    """

    def __init__(self, pipeline: PrefilterPipeline) -> None:
        self.pipeline = pipeline
        self._base_parser = pipeline._parser
        self._current_ir: IntermediateRepresentation | None = None
        self._history: list[PipelineResult] = []

    def run(self, query: str) -> PipelineResult:
        """Process a query in the context of current session history."""
        from prefilter_ai.history import QueryDiffEngine

        # 1. Parse the new query
        new_ir = self._base_parser.parse(query)
        new_ir.metadata["query_text"] = query

        # 2. Merge with existing session state
        if self._current_ir is None:
            merged_ir = new_ir
        else:
            engine = QueryDiffEngine()
            merged_ir = engine.diff_and_merge(self._current_ir, new_ir)

        # 3. Re-run full pipeline stages on merged state (fixes Gap #6)
        merged_ir = self.pipeline._ontology.infer(merged_ir, query)
        merged_ir = self.pipeline._validator.validate(merged_ir)

        relaxed_ir = None
        if self.pipeline.auto_relax and merged_ir.conflicts:
            relaxed_ir = self.pipeline._relaxer.relax_from_conflicts(merged_ir)

        explanation = build_explanation(merged_ir)
        result = PipelineResult(
            query=query,
            ir=merged_ir,
            relaxed_ir=relaxed_ir,
            conflicts=list(merged_ir.conflicts),
            warnings=list(merged_ir.warnings),
            explanation=explanation,
            latency_ms={},
            parser_backend=self.pipeline.parser_name,
        )

        self._current_ir = merged_ir
        self._history.append(result)
        return result

    def reset(self) -> None:
        """Clear session state."""
        self._current_ir = None
        self._history.clear()

    @property
    def history(self) -> list[PipelineResult]:
        return list(self._history)
