"""Controlled rerun: original vs corrected pipeline across N seeds, with a paired noise test."""
import math
import os
import statistics
from typing import Dict, List, Optional

from scipy import stats as _stats

from reprove.parser import parse_file
from reprove.tools.models import EvidenceObject
from reprove.verifier.models import SIGNIFICANCE_LEVEL, FindingStatus, SeedResult, VerificationResult, format_p
from reprove.verifier.patcher import generate_patch_with_reason
from reprove.verifier.runner import PipelineExecutionError, execute_pipeline

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
MIN_SEEDS = 3

# Preferred primary metric (first one present in BOTH runs wins)
METRIC_PRIORITY = [
    "test_auc", "roc_auc", "auc", "roc_auc_score",
    "test_acc", "accuracy", "acc", "test_accuracy", "accuracy_score", "balanced_accuracy_score",
    "test_f1", "f1", "f1_score",
    "test_r2", "r2", "r2_score",
    "test_score", "score", "average_precision_score", "precision_score", "recall_score",
    "test_rmse", "rmse", "test_mse", "mse", "mean_squared_error",
    "test_mae", "mae", "mean_absolute_error", "log_loss",
]
LOWER_IS_BETTER = {
    "test_rmse", "rmse", "test_mse", "mse", "mean_squared_error",
    "test_mae", "mae", "mean_absolute_error", "log_loss",
}


def _unverified(evidence: EvidenceObject, raw_code: str, reason: str,
                patched_code: str = "", diff: str = "") -> VerificationResult:
    """A SUSPECTED result: the pattern exists, but no rerun numbers could be produced."""
    return VerificationResult(
        evidence_id=evidence.evidence_id,
        status=FindingStatus.SUSPECTED,
        verification_ran=False,
        original_code=raw_code,
        patched_code=patched_code,
        diff=diff,
        reason=f"SUSPECTED (not verified): {reason}",
    )


def _pick_metric(orig: Dict[str, float], patched: Dict[str, float]) -> Optional[str]:
    for name in METRIC_PRIORITY:
        if name in orig and name in patched:
            return name
    common = sorted(set(orig) & set(patched))
    return common[0] if common else None


def verify_leakage(
    file_path: str,
    evidence: EvidenceObject,
    seeds: Optional[List[int]] = None,
    timeout: int = 60,
) -> VerificationResult:
    """
    1. Build a corrected copy of the pipeline (patcher).
    2. Run original and corrected code for every seed in separate processes.
    3. Compare them with a paired test on the per-seed differences.
    Never raises: anything that prevents a measurement becomes SUSPECTED with the reason.
    """
    seeds = list(seeds) if seeds else list(DEFAULT_SEEDS)
    graph = parse_file(file_path)
    raw_code = graph.raw_code
    cwd = os.path.dirname(os.path.abspath(file_path))

    patched_code, diff, why = generate_patch_with_reason(raw_code, evidence)
    if not patched_code:
        return _unverified(evidence, raw_code, f"no automatic fix could be built - {why}.")

    if len(seeds) < MIN_SEEDS:
        return _unverified(
            evidence, raw_code,
            f"only {len(seeds)} seed(s) given; at least {MIN_SEEDS} are needed to estimate seed noise.",
            patched_code, diff,
        )

    seed_runs: List[SeedResult] = []
    raw_pairs: List[tuple] = []
    primary: Optional[str] = None

    for seed in seeds:
        try:
            orig_metrics = execute_pipeline(raw_code, seed, timeout=timeout, cwd=cwd)
        except PipelineExecutionError as e:
            return _unverified(evidence, raw_code, f"the ORIGINAL pipeline failed to run: {e}", patched_code, diff)
        try:
            patch_metrics = execute_pipeline(patched_code, seed, timeout=timeout, cwd=cwd)
        except PipelineExecutionError as e:
            return _unverified(
                evidence, raw_code,
                f"the automatically corrected pipeline failed to run, so the fix does not fit this code: {e}",
                patched_code, diff,
            )

        if primary is None:
            primary = _pick_metric(orig_metrics, patch_metrics)
            if primary is None:
                return _unverified(
                    evidence, raw_code,
                    "no evaluation metric was captured. Store the test score in a variable such as "
                    "`test_acc`/`test_auc`, or compute it with sklearn.metrics.",
                    patched_code, diff,
                )
        if primary not in orig_metrics or primary not in patch_metrics:
            return _unverified(evidence, raw_code, f"metric `{primary}` was missing for seed {seed}.", patched_code, diff)

        o, p = float(orig_metrics[primary]), float(patch_metrics[primary])
        raw_pairs.append((o, p))
        seed_runs.append(SeedResult(seed=seed, original_metric=round(o, 4), patched_metric=round(p, 4), delta=round(p - o, 4)))

    higher_is_better = primary not in LOWER_IS_BETTER
    origs = [o for o, _ in raw_pairs]
    patches = [p for _, p in raw_pairs]
    deltas = [p - o for o, p in raw_pairs]                      # patched - original
    degradation = [-d if higher_is_better else d for d in deltas]  # > 0 means the fix made the score worse

    mean_delta = statistics.mean(deltas)
    mean_degr = statistics.mean(degradation)
    noise = statistics.stdev(degradation)  # paired, ddof=1
    n = len(seeds)
    std_err = noise / math.sqrt(n)
    if noise == 0:
        p_value = 0.0 if mean_degr > 0 else 1.0
    else:
        p_value = float(_stats.ttest_1samp(degradation, 0.0, alternative="greater").pvalue)
    significant = mean_degr > 0 and p_value < SIGNIFICANCE_LEVEL
    direction = "drop" if higher_is_better else "increase"
    worse_seeds = sum(1 for d in degradation if d > 0)

    if significant:
        status = FindingStatus.CONFIRMED
        reason = (
            f"CONFIRMED: fixing the flaw changed `{primary}` from {statistics.mean(origs):.4f} to "
            f"{statistics.mean(patches):.4f} (mean {direction} of {abs(mean_delta):.4f}; worse on "
            f"{worse_seeds}/{n} seeds). One-sided paired t-test: p = {format_p(p_value)} < {SIGNIFICANCE_LEVEL}."
        )
        if noise == 0:
            reason += " Note: every seed gave the identical difference, so seeds may not affect this pipeline."
    else:
        status = FindingStatus.NOT_CONFIRMED
        if mean_degr <= 0:
            detail = "the corrected pipeline did not score worse on average"
        else:
            detail = (
                f"the mean {direction} of {abs(mean_delta):.4f} is not statistically significant "
                f"(paired t-test p = {format_p(p_value)}, worse on {worse_seeds}/{n} seeds)"
            )
        reason = (
            f"NOT_CONFIRMED: the flawed pattern is present, but {detail}, "
            f"so no measurable inflation of `{primary}` was found."
        )

    return VerificationResult(
        evidence_id=evidence.evidence_id,
        status=status,
        verification_ran=True,
        primary_metric_name=primary,
        higher_is_better=higher_is_better,
        original_mean=round(statistics.mean(origs), 4),
        original_std=round(statistics.stdev(origs), 4),
        patched_mean=round(statistics.mean(patches), 4),
        patched_std=round(statistics.stdev(patches), 4),
        mean_delta=round(mean_delta, 4),
        seed_noise_std=round(noise, 4),
        standard_error=round(std_err, 4),
        p_value=round(p_value, 6),
        is_significant=significant,
        seed_runs=seed_runs,
        original_code=raw_code,
        patched_code=patched_code,
        diff=diff,
        reason=reason,
    )
