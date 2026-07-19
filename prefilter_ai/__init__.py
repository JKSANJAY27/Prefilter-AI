"""
prefilter-ai
============
AI-powered Query Understanding Middleware — sits between natural language
search queries and any search or retrieval backend.

Quick start (simple parser)
---------------------------
>>> from prefilter_ai import PrefilterAI
>>> ai = PrefilterAI(parse_backend="spacy")
>>> result = ai.parse("Sony noise cancelling headphones under $200, not red")
>>> result.fields
{'domain': 'ecommerce', 'product': 'headphones', 'brand': 'Sony', ...}

Quick start (full pipeline — recommended)
-----------------------------------------
>>> from prefilter_ai import PrefilterPipeline
>>> pipeline = PrefilterPipeline(parser="spacy")
>>> result = pipeline.run("gaming laptop under $800")
>>> result.ir          # IntermediateRepresentation with all stages
>>> result.conflicts   # Feasibility warnings
>>> result.sql         # SQL WHERE clause
>>> result.elasticsearch  # Elasticsearch DSL

Multi-turn session
------------------
>>> session = pipeline.new_session()
>>> r1 = session.run("gaming laptops")
>>> r2 = session.run("Only Lenovo, under $1200")
>>> r3 = session.run("Actually 32GB RAM")
"""

from prefilter_ai.config import DEFAULT_MODEL_IDS, ModelFormat
from prefilter_ai.exceptions import ModelLoadError, ParseError, PrefilterAIError
from prefilter_ai.expert import PrefilterAI
from prefilter_ai.pipeline import PrefilterPipeline, PipelineResult, PipelineSession
from prefilter_ai.result import ParseResult
from prefilter_ai.ir import IntermediateRepresentation, IRFilterConstraint, IRPreference
from prefilter_ai.history import PrefilterSession

__version__ = "0.2.0"
__author__ = "Sanjay JK"
__all__ = [
    # Core pipeline (recommended)
    "PrefilterPipeline",
    "PipelineResult",
    "PipelineSession",
    # Legacy single-call API
    "PrefilterAI",
    "ParseResult",
    # IR data model
    "IntermediateRepresentation",
    "IRFilterConstraint",
    "IRPreference",
    # Session management
    "PrefilterSession",
    # Errors
    "ModelLoadError",
    "ParseError",
    "PrefilterAIError",
    # Config
    "DEFAULT_MODEL_IDS",
    "ModelFormat",
]
