"""
parser_interface.py — Unified Parser Interface for Prefilter AI.

Supports local fine-tuned SLM, rule-based spaCy, and API-based Gemini models.
All parsers share the split_operator_value utility from prefilter_ai.utils.
"""

from __future__ import annotations

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from prefilter_ai.config import ModelFormat, ParseBackend, build_inference_prompt, get_system_prompt
from prefilter_ai.exceptions import ParseError
from prefilter_ai.ir import IntermediateRepresentation
from prefilter_ai.parser import parse_model_output
from prefilter_ai.utils import split_operator_value, confidence_from_extraction

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """Abstract base class for all query extraction parsers."""

    @abstractmethod
    def parse(self, query: str) -> IntermediateRepresentation:
        """Parse natural language query into IntermediateRepresentation."""
        pass


class SpacyParser(BaseParser):
    """Rule-based extractor using spaCy NER + regex."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        from prefilter_ai.spacy_extractor import SpacyExtractor
        self.extractor = SpacyExtractor(model=spacy_model)

    def parse(self, query: str) -> IntermediateRepresentation:
        meta = self.extractor.extract_with_meta(query.strip())
        domain = meta.pop("domain", "general")

        ir = IntermediateRepresentation(domain=domain, intent="search")

        for k, entry in meta.items():
            # entry is either a raw value or a dict with {value, entity_type, pattern_matched}
            if isinstance(entry, list):
                for item in entry:
                    raw, etype, pmatched = self._unpack_entry(item)
                    op, val, val_hi = split_operator_value(raw)
                    conf = confidence_from_extraction(
                        entity_type=etype,
                        pattern_matched=pmatched,
                        is_numeric=op not in {"eq", "ne"},
                        backend="spacy",
                    )
                    ir.add_filter(
                        field_name=k, operator=op, value=val, value_hi=val_hi,
                        confidence=conf, provenance="spaCy extractor",
                    )
            else:
                raw, etype, pmatched = self._unpack_entry(entry)
                op, val, val_hi = split_operator_value(raw)
                conf = confidence_from_extraction(
                    entity_type=etype,
                    pattern_matched=pmatched,
                    is_numeric=op not in {"eq", "ne"},
                    backend="spacy",
                )
                ir.add_filter(
                    field_name=k, operator=op, value=val, value_hi=val_hi,
                    confidence=conf, provenance="spaCy extractor",
                )
        return ir

    def _unpack_entry(self, entry: Any) -> tuple[Any, str, bool]:
        """Unpack an extraction entry into (raw_value, entity_type, pattern_matched)."""
        if isinstance(entry, dict) and "__raw__" in entry:
            return entry["__raw__"], entry.get("entity_type", ""), entry.get("pattern_matched", False)
        return entry, "", True  # fallback: treat as raw value, pattern matched


class SLMParser(BaseParser):
    """Local inference using fine-tuned Qwen 0.8B adapter model."""

    def __init__(
        self,
        fmt: ModelFormat | str = ModelFormat.JSON,
        model_id: str | None = None,
        load_in_4bit: bool = True,
        backend: str = "auto",
        generation_config: dict[str, Any] | None = None,
    ):
        from prefilter_ai.loader import load_model
        self.fmt = ModelFormat(fmt)
        self.generation_cfg = generation_config or {
            "max_new_tokens": 256,
            "temperature": 0.1,
            "do_sample": True,
            "use_cache": True,
        }
        self.model, self.tokenizer = load_model(
            fmt=self.fmt,
            model_id=model_id,
            load_in_4bit=load_in_4bit,
            backend=backend,
        )

    def parse(self, query: str) -> IntermediateRepresentation:
        import torch

        prompt = build_inference_prompt(query.strip(), self.fmt)
        inputs = self.tokenizer(text=prompt, return_tensors="pt", add_special_tokens=False)

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.generation_cfg,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][prompt_len:]
        raw_output = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        fields = parse_model_output(raw_output, self.fmt)
        domain = fields.pop("domain", "general")

        ir = IntermediateRepresentation(domain=domain, intent="search")
        for k, v in fields.items():
            if isinstance(v, list):
                for item in v:
                    op, val, val_hi = split_operator_value(item)
                    conf = confidence_from_extraction(
                        pattern_matched=True,
                        is_numeric=op not in {"eq", "ne"},
                        backend="slm",
                    )
                    ir.add_filter(
                        field_name=k, operator=op, value=val, value_hi=val_hi,
                        confidence=conf, provenance="SLM fine-tuned model",
                    )
            else:
                op, val, val_hi = split_operator_value(v)
                conf = confidence_from_extraction(
                    pattern_matched=True,
                    is_numeric=op not in {"eq", "ne"},
                    backend="slm",
                )
                ir.add_filter(
                    field_name=k, operator=op, value=val, value_hi=val_hi,
                    confidence=conf, provenance="SLM fine-tuned model",
                )
        return ir


class GeminiParser(BaseParser):
    """API-based parser using Gemini to parse queries without local model dependencies."""

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        if not self.api_key:
            logger.warning("GEMINI_API_KEY environment variable is not set. GeminiParser calls will fail.")

    def parse(self, query: str) -> IntermediateRepresentation:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be set to use GeminiParser.")

        import urllib.request

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

        system_instruction = (
            "You are a structured search query parser. Given a natural language query, extract the fields "
            "as a JSON object. The JSON must contain a 'domain' field, and key-value fields. "
            "Categorize operators using prefixes like 'lt:N', 'lte:N', 'gt:N', 'gte:N', 'approx:N', 'between:Lo:Hi', 'ne:Val'. "
            "Return ONLY the raw JSON block without formatting, comments or markdown fences."
        )

        data = {
            "contents": [{
                "parts": [
                    {"text": f"System prompt: {system_instruction}\nUser Query: {query}"}
                ]
            }],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                text_out = res_body["candidates"][0]["content"]["parts"][0]["text"].strip()
                fields = json.loads(text_out)
        except Exception as e:
            raise ParseError(f"Gemini API request failed: {e}", raw_output=str(e))

        domain = fields.pop("domain", "general")
        ir = IntermediateRepresentation(domain=domain, intent="search")
        for k, v in fields.items():
            if isinstance(v, list):
                for item in v:
                    op, val, val_hi = split_operator_value(item)
                    conf = confidence_from_extraction(
                        pattern_matched=True,
                        is_numeric=op not in {"eq", "ne"},
                        backend="gemini",
                    )
                    ir.add_filter(
                        field_name=k, operator=op, value=val, value_hi=val_hi,
                        confidence=conf, provenance="Gemini API",
                    )
            else:
                op, val, val_hi = split_operator_value(v)
                conf = confidence_from_extraction(
                    pattern_matched=True,
                    is_numeric=op not in {"eq", "ne"},
                    backend="gemini",
                )
                ir.add_filter(
                    field_name=k, operator=op, value=val, value_hi=val_hi,
                    confidence=conf, provenance="Gemini API",
                )
        return ir
