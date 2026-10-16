"""Benchmark evaluation matrix runner measuring Precision, Recall, and Runtime."""
import os
import time
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from reprove.agent.investigator import run_investigation
from reprove.agent.schemas import InvestigationReport
from reprove.verifier import FindingStatus

BENCHMARK_CASES = [
    {
        "id": "BM-01",
        "file_path": "examples/01_clear_leak_scaler_before_split.ipynb",
        "name": "Type 1: Preprocessing & Feature Selection Leak",
        "expected_has_leak": True,
        "expected_status": "CONFIRMED",
    },
    {
        "id": "BM-02",
        "file_path": "examples/02_clean_pipeline_no_leak.ipynb",
        "name": "Negative Control: Encapsulated Pipeline",
        "expected_has_leak": False,
        "expected_status": "CLEAN",
    },
    {
        "id": "BM-03",
        "file_path": "examples/03_subtle_leak_duplicate_split.ipynb",
        "name": "Type 2: Cross-Split Sample Duplication",
        "expected_has_leak": True,
        "expected_status": "SUSPECTED",
    },
]


class BenchmarkResult(BaseModel):
    """Result for a single benchmark evaluation case."""
    case_id: str
    name: str
    file_path: str
    expected_has_leak: bool
    detected_has_leak: bool
    verdict: str
    confirmed_count: int
    suspected_count: int
    clean_count: int
    empirical_delta: Optional[float] = None
    runtime_seconds: float
    is_correct: bool


class BenchmarkEvaluationSummary(BaseModel):
    """Aggregate benchmark evaluation metrics."""
    total_cases: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    specificity: float
    total_runtime_seconds: float
    results: List[BenchmarkResult] = Field(default_factory=list)


def evaluate_benchmarks(
    use_replay: bool = True,
    model: str = "qwen2.5-coder:7b",
    seeds: Optional[List[int]] = None,
) -> BenchmarkEvaluationSummary:
    """
    Executes REPROVE across all benchmark cases and computes detection metrics.
    Strict Invariant: Zero synthetic numbers; all metrics are empirically recorded.
    """
    seeds = seeds or [42, 101, 7]
    results: List[BenchmarkResult] = []
    tp = fp = tn = fn = 0
    start_total = time.perf_counter()

    for case in BENCHMARK_CASES:
        t0 = time.perf_counter()
        report: InvestigationReport = run_investigation(
            file_path=case["file_path"],
            model_name=model,
            use_replay=use_replay,
            seeds=seeds,
        )
        elapsed = time.perf_counter() - t0

        confirmed_cnt = len(report.confirmed_evidence)
        suspected_cnt = len(report.suspected_evidence)
        clean_cnt = len(report.clean_checks)

        # Has leak if any confirmed or suspected
        detected_has_leak = (confirmed_cnt + suspected_cnt) > 0
        expected_has_leak = case["expected_has_leak"]

        verdict = "CLEAN"
        if confirmed_cnt > 0:
            verdict = "CONFIRMED"
        elif suspected_cnt > 0:
            verdict = "SUSPECTED"

        # Classification matrix
        if expected_has_leak and detected_has_leak:
            tp += 1
            is_correct = True
        elif not expected_has_leak and not detected_has_leak:
            tn += 1
            is_correct = True
        elif not expected_has_leak and detected_has_leak:
            fp += 1
            is_correct = False
        else:
            fn += 1
            is_correct = False

        delta = None
        if report.verification_results:
            delta = report.verification_results[0].mean_delta

        results.append(
            BenchmarkResult(
                case_id=case["id"],
                name=case["name"],
                file_path=case["file_path"],
                expected_has_leak=expected_has_leak,
                detected_has_leak=detected_has_leak,
                verdict=verdict,
                confirmed_count=confirmed_cnt,
                suspected_count=suspected_cnt,
                clean_count=clean_cnt,
                empirical_delta=delta,
                runtime_seconds=round(elapsed, 4),
                is_correct=is_correct,
            )
        )

    total_time = time.perf_counter() - start_total

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0

    return BenchmarkEvaluationSummary(
        total_cases=len(BENCHMARK_CASES),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        specificity=round(specificity, 4),
        total_runtime_seconds=round(total_time, 4),
        results=results,
    )
