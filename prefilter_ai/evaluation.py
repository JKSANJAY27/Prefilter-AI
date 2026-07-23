"""
evaluation.py — Evaluation Harness for Prefilter AI.

Measures parser precision/recall, latency breakdown, feasibility validation,
and constraint relaxation success to trace search pipeline metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from prefilter_ai.expert import PrefilterAI
from prefilter_ai.relaxer import QueryRelaxer
from prefilter_ai.validator import ConflictDetector


@dataclass
class LatencyProfile:
    extraction_ms: float = 0.0
    ontology_ms: float = 0.0
    validation_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class EvalResult:
    query: str
    ground_truth: dict[str, Any]
    extracted_fields: dict[str, Any]
    precision: float
    recall: float
    f1: float
    latency: LatencyProfile
    conflicts_detected: list[str] = field(default_factory=list)
    relaxed_steps_logged: list[str] = field(default_factory=list)


class EvaluationHarness:
    """Evaluates Prefilter AI query understanding performance and profiles latency."""

    def __init__(self, expert: PrefilterAI | None = None):
        self.expert = expert or PrefilterAI(parse_backend="spacy")
        self.detector = ConflictDetector()
        self.relaxer = QueryRelaxer()

    def evaluate_item(
        self, query: str, ground_truth: dict[str, Any], relax_level: int = 0
    ) -> EvalResult:
        """Run parser, ontology, validation, and optionally relaxation metrics on a single query."""
        latency = LatencyProfile()

        # 1. Measure Extraction
        start = time.perf_counter()
        result = self.expert.parse(query)
        latency.extraction_ms = (time.perf_counter() - start) * 1000

        # Retrieve IR
        ir = result._get_or_create_ir()

        # 2. Run Ontology Preference Inference (Ontology matches already applied in parse() via parser if structured,
        # but let's time the direct inference path)
        start_ont = time.perf_counter()
        from prefilter_ai.ontology import OntologyEngine

        ir = OntologyEngine().infer(ir, query)
        latency.ontology_ms = (time.perf_counter() - start_ont) * 1000

        # 3. Validate Feasibility
        start_val = time.perf_counter()
        self.detector.validate(ir)
        latency.validation_ms = (time.perf_counter() - start_val) * 1000

        latency.total_ms = latency.extraction_ms + latency.ontology_ms + latency.validation_ms

        # 4. Optional relaxation
        relaxed_logs = []
        if relax_level > 0:
            relaxed_ir = self.relaxer.relax(ir, relaxation_level=relax_level)
            relaxed_logs = relaxed_ir.metadata.get("relaxation_logs", [])

        # Calculate metrics (precision/recall of fields)
        extracted = result.fields

        # Compare ground truth keys/values (exclude domain)
        gt_filtered = {k: v for k, v in ground_truth.items() if k != "domain"}
        ext_filtered = {k: v for k, v in extracted.items() if k != "domain"}

        tp = 0
        for k, v in ext_filtered.items():
            if k in gt_filtered and self._match_value(v, gt_filtered[k]):
                tp += 1

        precision = tp / len(ext_filtered) if ext_filtered else 1.0
        recall = tp / len(gt_filtered) if gt_filtered else 1.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return EvalResult(
            query=query,
            ground_truth=ground_truth,
            extracted_fields=extracted,
            precision=precision,
            recall=recall,
            f1=f1,
            latency=latency,
            conflicts_detected=ir.conflicts,
            relaxed_steps_logged=relaxed_logs,
        )

    def evaluate_dataset(
        self, dataset: list[dict[str, Any]], relax_level: int = 0
    ) -> dict[str, Any]:
        """Run evaluation across a list of test records."""
        results = []
        total_p = 0.0
        total_r = 0.0
        total_f1 = 0.0
        total_lat = 0.0
        conflict_count = 0

        for item in dataset:
            query = item["query"]
            gt = item["ground_truth"]
            res = self.evaluate_item(query, gt, relax_level=relax_level)
            results.append(res)

            total_p += res.precision
            total_r += res.recall
            total_f1 += res.f1
            total_lat += res.latency.total_ms
            if res.conflicts_detected:
                conflict_count += 1

        n = len(dataset)
        return {
            "metrics": {
                "avg_precision": total_p / n if n else 0.0,
                "avg_recall": total_r / n if n else 0.0,
                "avg_f1": total_f1 / n if n else 0.0,
                "avg_latency_ms": total_lat / n if n else 0.0,
                "conflict_rate": conflict_count / n if n else 0.0,
            },
            "results": results,
        }

    def _match_value(self, ext_val: Any, gt_val: Any) -> bool:
        """Compare string/list values case-insensitively."""
        if isinstance(ext_val, list) and isinstance(gt_val, list):
            return {str(x).lower() for x in ext_val} == {str(y).lower() for y in gt_val}
        return str(ext_val).lower() == str(gt_val).lower()
