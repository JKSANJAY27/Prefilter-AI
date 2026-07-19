"""
Parser utilities: convert raw model output strings into Python dicts.

Each parser is lenient by design — it attempts a best-effort extraction
before raising ParseError, because small models occasionally produce
slightly malformed output.
"""

from __future__ import annotations

import json
import re

import yaml

from prefilter_ai.config import ModelFormat
from prefilter_ai.exceptions import ParseError

# ── JSON Repair and Parsing ───────────────────────────────────────────


def repair_json_string(s: str) -> str:
    """Best-effort pure Python JSON repair for truncated model outputs."""
    s = s.strip()
    if not s:
        return s

    in_quote = False
    escape = False
    stack = []
    repaired = []

    for char in s:
        repaired.append(char)
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_quote = not in_quote
            continue
        if not in_quote:
            if char == '{':
                stack.append('}')
            elif char == '[':
                stack.append(']')
            elif char == '}':
                if stack and stack[-1] == '}':
                    stack.pop()
            elif char == ']':
                if stack and stack[-1] == ']':
                    stack.pop()

    if in_quote:
        repaired.append('"')

    while stack:
        repaired.append(stack.pop())

    return "".join(repaired)


def _parse_json(text: str) -> dict:
    text = text.strip()

    # 1. Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 2. Try third-party json-repair if available
    try:
        import json_repair
        result = json_repair.repair_json(text, return_objects=True)
        if isinstance(result, dict):
            return result
    except ImportError:
        pass
    except Exception:
        pass

    # 3. Extract first {...} block and try parsing
    match = re.search(r"\{.*", text, re.DOTALL)
    if match:
        candidate = match.group()
        # Clean trailing prose after the closing brace if there is any
        # We find the last matching brace in case there is trailing prose
        last_brace = candidate.rfind('}')
        if last_brace != -1:
            candidate = candidate[:last_brace+1]
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            # Try repairing candidate
            try:
                repaired = repair_json_string(candidate)
                result = json.loads(repaired)
                if isinstance(result, dict):
                    return result
            except Exception:
                pass

    # 4. Try direct repair of the entire text as a last resort
    try:
        repaired = repair_json_string(text)
        # Search for {...} in repaired text
        match = re.search(r"\{.*?\}", repaired, re.DOTALL)
        if match:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
    except Exception:
        pass

    raise ParseError(
        "Could not parse model output as JSON.",
        raw_output=text,
    )


# ── YAML ──────────────────────────────────────────────────────────────


def _parse_yaml(text: str) -> dict:
    text = text.strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:yaml)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)

    try:
        result = yaml.safe_load(text)
        if isinstance(result, dict):
            return result
    except yaml.YAMLError:
        pass

    raise ParseError(
        "Could not parse model output as YAML.",
        raw_output=text,
    )


# ── Dispatcher ────────────────────────────────────────────────────────

_PARSERS = {
    ModelFormat.JSON: _parse_json,
    ModelFormat.YAML: _parse_yaml,
}


def parse_model_output(raw: str, fmt: ModelFormat) -> dict:
    """
    Parse the raw string produced by the model into a Python dict.

    Parameters
    ----------
    raw : str
        The decoded model output (already stripped of the prompt).
    fmt : ModelFormat
        Which format the model was asked to produce.

    Returns
    -------
    dict
        Extracted field → value mapping.

    Raises
    ------
    ParseError
        If the output cannot be parsed after best-effort attempts.
    """
    parser = _PARSERS.get(fmt)
    if parser is None:
        raise ParseError(f"No parser registered for format {fmt!r}.")

    return parser(raw)
