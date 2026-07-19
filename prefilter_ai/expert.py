"""
expert.py — Main public interface for Prefilter AI.
"""

from __future__ import annotations

import logging
from typing import Any

from prefilter_ai.config import ModelFormat, ParseBackend
from prefilter_ai.parser_interface import SpacyParser, SLMParser, GeminiParser
from prefilter_ai.result import ParseResult

logger = logging.getLogger(__name__)


class PrefilterAI:
    """
    Main entry point for parsing natural language search queries.

    Three backends are supported:
    - SLM (default) — Local fine-tuned LoRA adapter model.
    - SPACY — Fast CPU rule-based extraction (~1ms).
    - GEMINI — API-driven Gemini extraction.
    """

    def __init__(
        self,
        fmt: ModelFormat | str = ModelFormat.JSON,
        model_id: str | None = None,
        load_in_4bit: bool = True,
        backend: str = "auto",
        generation_config: dict[str, Any] | None = None,
        parse_backend: ParseBackend | str = ParseBackend.SLM,
        spacy_model: str = "en_core_web_sm",
        eager: bool = False,
    ) -> None:
        self.fmt = ModelFormat(fmt)
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.backend = backend
        self.generation_cfg = generation_config
        self.parse_backend = ParseBackend(parse_backend)
        self.spacy_model = spacy_model

        self._parser = None

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

    # Legacy attributes support (backward compatibility)
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
        """Parse natural language query into ParseResult."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        ir = self.parser.parse(query)
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
            f"format={self.fmt.value!r})"
        )
