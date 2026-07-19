"""
prefilter-ai
============
A lightweight library for parsing natural language search queries into
structured output using fine-tuned Qwen3.5-0.8B LoRA adapters.

Quick start
-----------
>>> from prefilter_ai import PrefilterAI
>>> expert = PrefilterAI()                  # defaults to JSON model
>>> result = expert.parse("MacBook under $2000 with 16GB RAM")
>>> print(result.fields)
{'domain': 'ecommerce', 'product': 'MacBook', 'price': 'lt:2000', 'feature': '16GB RAM'}
"""

from prefilter_ai.config import DEFAULT_MODEL_IDS, ModelFormat
from prefilter_ai.exceptions import ModelLoadError, ParseError, PrefilterAIError
from prefilter_ai.expert import PrefilterAI
from prefilter_ai.result import ParseResult

__version__ = "0.1.2"
__author__ = "Sanjay JK"
__all__ = [
    "DEFAULT_MODEL_IDS",
    "ModelFormat",
    "ModelLoadError",
    "ParseError",
    "ParseResult",
    "PrefilterAI",
    "PrefilterAIError",
]
