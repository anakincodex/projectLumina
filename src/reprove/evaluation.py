"""Benchmark runner: compares REPROVE verdicts with the planted ground truth of each case."""
import time
from typing import List, Optional

from pydantic import BaseModel, Field

from reprove.agent.investigator import DEFAULT_MODEL, run_investigation
from reprove.verifier import DEFAULT_SEEDS

BENCHMARK_CASES = [
    {
        "id": "BM-01",
        "file_path": "examples/01_clear_leak_scaler_before_split.ipynb",
        "name": "Type 1: scaler + feature selection before split",
        "expected_has_leak": True,
        "expected_status": "CONFIRMED",
    },
    {
        "id": "BM-02",
        "file_path": "examples/02_clean_pipeline_no_leak.ipynb",
        "name": "Clean: split first, sklearn Pipeline",
        "expected_has_leak": False,
        "expected_status": "CLEAN",
    },
    {
        "id": "BM-03",
        "file_path": "examples/03_subtle_leak_duplicate_split.ipynb",
        "name": "Type 2: duplicated rows before split",
        "expected_has_leak": True,
        "expected_status": "CONFIRMED",
    },
    {
        "id": "BM-04",
        "file_path": "examples/04_test_set_model_selection.ipynb",
        "name": "Type 3: hyperparameter chosen on test set",
        "expected_has_leak": True,
        "expected_status": "SUSPECTED",
    },
    {
        "id": "BM-05",
        "file_path": "examples/stress/05_minmax_scaler_leak.py",
        "name": "Stress: real but harmless scaler leak",
        "expected_has_leak": True,
        "expected_status": "NOT_CONFIRMED",
    },
    {
        "id": "BM-06",
        "file_path": "examples/stress/06_concat_columns_clean.py",
        "name": "Stress: clean code using pd.concat",
        "expected_has_leak": False,
        "expected_status": "CLEAN",
    },
]

FLAGGED = {"CONFIRMED", "SUSPECTED", "NOT_CONFIRMED"}


class BenchmarkResult(BaseModel):
    case_id: str
    name: str
    file_path: str
    expected_has_leak: bool
    expected_status: str
    detected_has_leak: bool
    verdict: str
    mode: str
    confirmed_count: int
    suspected_count: int
    not_confirmed_count: int
    clean_count: int
    empirical_delta: Optional[float] = None
    runtime_seconds: float
    is_correct: bool       # leak flagged / not flagged as expected
    status_match: bool     # exact expected status


class BenchmarkEvaluationSummary(BaseModel):
    total_cases: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    status_matches: int
    precision: float
    recall: float
    f1_score: float
    specificity: float
    total_runtime_seconds: float
    results: List[BenchmarkResult] = Field(default_factory=list)


def evaluate_benchmarks(
    use_replay: bool = True,
    model: str = DEFAULT_MODEL,
    seeds: Optional[List[int]] = None,
) -> BenchmarkEvaluationSummary:
    """Run every case. All numbers come from the audits themselves (live or recorded)."""
    seeds = seeds or list(DEFAULT_SEEDS)
    results: List[BenchmarkResult] = []
    tp = fp = tn = fn = matches = 0
    start = time.perf_counter()

    for case in BENCHMARK_CASES:
        t0 = time.perf_counter()
        report = run_investigation(case["file_path"], model_name=model, use_replay=use_replay, seeds=seeds)
        elapsed = time.perf_counter() - t0

        verdict = report.verdict
        detected = verdict in FLAGGED
        expected = case["expected_has_leak"]
        if expected and detected:
            tp += 1
        elif not expected and not detected:
            tn += 1
        elif detected:
            fp += 1
        else:
            fn += 1
        status_match = verdict == case["expected_status"]
        matches += int(status_match)

        ran = [v for v in report.verification_results if v.verification_ran]
        results.append(BenchmarkResult(
            case_id=case["id"],
            name=case["name"],
            file_path=case["file_path"],
            expected_has_leak=expected,
            expected_status=case["expected_status"],
            detected_has_leak=detected,
            verdict=verdict,
            mode=report.effective_mode.replace("FALLBACK_DETERMINISTIC", "FALLBACK").replace("LLM_OLLAMA", "LLM"),
            confirmed_count=len(report.confirmed_evidence),
            suspected_count=len(report.suspected_evidence),
            not_confirmed_count=len(report.not_confirmed_evidence),
            clean_count=len(report.clean_checks),
            empirical_delta=ran[0].mean_delta if ran else None,
            runtime_seconds=round(elapsed, 3),
            is_correct=(expected == detected),
            status_match=status_match,
        ))

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 1.0

    return BenchmarkEvaluationSummary(
        total_cases=len(BENCHMARK_CASES),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        status_matches=matches,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        specificity=round(specificity, 4),
        total_runtime_seconds=round(time.perf_counter() - start, 3),
        results=results,
    )
