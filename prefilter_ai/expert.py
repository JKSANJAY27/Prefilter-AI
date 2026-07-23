"""
expert.py — Main public interface for Prefilter AI.

PrefilterAI.parse() now runs the complete middleware pipeline
(parse → ontology → validate → relax), not just the parser stage.
"""

from __future__ import annotations

import logging
from typing import Any

from prefilter_ai.config import ModelFormat, ParseBackend
from prefilter_ai.ontology import OntologyEngine
from prefilter_ai.parser_interface import GeminiParser, SLMParser, SpacyParser
from prefilter_ai.registry import SchemaRegistry
from prefilter_ai.relaxer import QueryRelaxer
from prefilter_ai.result import ParseResult
from prefilter_ai.validator import ConflictDetector

logger = logging.getLogger(__name__)


class PrefilterAI:
    """
    Main entry point for parsing natural language search queries.

    Runs the full middleware pipeline: parse → ontology → validate → relax.
    Use ``PrefilterPipeline`` for richer output including DSL translations,
    per-field explanations, and latency profiling.

    Three parser backends:
    - SPACY  — Rule-based spaCy NER + regex (~1ms, CPU, default).
    - SLM    — Local fine-tuned Qwen 0.8B LoRA adapter (requires torch+peft).
    - GEMINI — API-driven Gemini extraction (requires GEMINI_API_KEY).
    """

    def __init__(
        self,
        fmt: ModelFormat | str = ModelFormat.JSON,
        model_id: str | None = None,
        load_in_4bit: bool = True,
        backend: str = "auto",
        generation_config: dict[str, Any] | None = None,
        parse_backend: ParseBackend | str = ParseBackend.SPACY,
        spacy_model: str = "en_core_web_sm",
        eager: bool = False,
        run_pipeline: bool = True,
        auto_relax: bool = True,
    ) -> None:
        self.fmt = ModelFormat(fmt)
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.backend = backend
        self.generation_cfg = generation_config
        self.parse_backend = ParseBackend(parse_backend)
        self.spacy_model = spacy_model
        self.run_pipeline = run_pipeline
        self.auto_relax = auto_relax

        self._parser = None
        self._ontology = OntologyEngine()
        self._validator = ConflictDetector()
        self._relaxer = QueryRelaxer(registry=SchemaRegistry())

        if eager:
            self._init_parser()

    def _init_parser(self):
        if self.parse_backend == ParseBackend.SPACY:
            self._parser = SpacyParser(spacy_model=self.spacy_model)
        elif self.parse_backend == ParseBackend.GEMINI:
            self._parser = GeminiParser()
        else:
            self._parser = SLMParser(
                fmt=self.fmt,
                model_id=self.model_id,
                load_in_4bit=self.load_in_4bit,
                backend=self.backend,
                generation_config=self.generation_cfg,
            )

    @property
    def parser(self):
        if self._parser is None:
            self._init_parser()
        return self._parser

    # Legacy attribute compatibility
    @property
    def model(self):
        if isinstance(self.parser, SLMParser):
            return self.parser.model
        return None

    @property
    def tokenizer(self):
        if isinstance(self.parser, SLMParser):
            return self.parser.tokenizer
        return None

    @property
    def spacy_extractor(self):
        if isinstance(self.parser, SpacyParser):
            return self.parser.extractor
        return None

    def parse(self, query: str) -> ParseResult:
        """
        Parse a natural language query through the full middleware pipeline.

        Runs: parser → ontology inference → conflict detection → relaxation.
        Returns a ParseResult with the enriched IR attached.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        ir = self.parser.parse(query)
        ir.metadata["query_text"] = query

        if self.run_pipeline:
            ir = self._ontology.infer(ir, query)
            ir = self._validator.validate(ir)

            if self.auto_relax and ir.conflicts:
                relaxed_ir = self._relaxer.relax_from_conflicts(ir)
                ir.metadata["relaxed_ir"] = relaxed_ir

        return ParseResult(
            query=query,
            fields=ir.legacy_fields,
            raw_output="",
            model_format=self.parse_backend.value,
            _ir=ir,
        )

    def parse_batch(self, queries: list[str]) -> list[ParseResult]:
        return [self.parse(q) for q in queries]

    def __repr__(self) -> str:
        return (
            f"PrefilterAI("
            f"parse_backend={self.parse_backend.value!r}, "
            f"format={self.fmt.value!r}, "
            f"run_pipeline={self.run_pipeline})"
        )
