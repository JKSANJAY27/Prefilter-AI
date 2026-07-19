"""
Custom exceptions for prefilter-ai.
All library errors inherit from PrefilterAIError so callers can catch
the entire family with a single except clause if desired.
"""


class PrefilterAIError(Exception):
    """Base exception for all prefilter-ai errors."""


class ModelLoadError(PrefilterAIError):
    """Raised when the model or tokenizer cannot be loaded."""


class ParseError(PrefilterAIError):
    """
    Raised when the model output cannot be parsed into a structured dict.
    Carries the raw model output for debugging.
    """

    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output

    def __str__(self):
        base = super().__str__()
        if self.raw_output:
            preview = self.raw_output[:200].replace("\n", "\\n")
            return f"{base} | raw_output={preview!r}"
        return base
