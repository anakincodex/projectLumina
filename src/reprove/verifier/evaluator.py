"""Evaluates leakage findings by comparing multi-seed reruns of original vs patched pipelines."""
from typing import List, Optional
import numpy as np

from reprove.parser import parse_file
from reprove.tools.models import EvidenceObject
from reprove.verifier.models import FindingStatus, SeedResult, VerificationResult
from reprove.verifier.patcher import generate_patch
from reprove.verifier.runner import execute_pipeline

DEFAULT_SEEDS = [42, 101, 2024, 7, 99]
METRIC_PRIORITY = ["test_auc", "roc_auc", "test_acc", "accuracy", "test_f1", "f1"]


def verify_leakage(
    file_path: str,
    evidence: EvidenceObject,
    seeds: Optional[List[int]] = None,
    timeout: int = 15,
) -> VerificationResult:
    """
    Controlled experiment verification:
    1. Patches original pipeline to eliminate the suspected leak.
    2. Runs original vs patched pipelines across N seeds in sandboxed subprocesses.
    3. Calculates empirical mean, standard deviation, and statistical significance of the delta.
    """
    seeds = seeds or DEFAULT_SEEDS
    graph = parse_file(file_path)
    raw_code = graph.raw_code

    # 1. Generate patch
    patched_code, diff = generate_patch(raw_code, evidence)
    if not patched_code or not diff:
        return VerificationResult(
            status=FindingStatus.SUSPECTED,
            primary_metric_name="unknown",
            original_mean=0.0,
            original_std=0.0,
            patched_mean=0.0,
            patched_std=0.0,
            mean_delta=0.0,
            seed_noise_std=0.0,
            is_significant=False,
            seed_runs=[],
            original_code=raw_code,
            patched_code="",
            diff="",
            reason="Could not automatically construct a deterministic patch for this leak type.",
        )

    # 2. Execute multi-seed runs
    seed_runs: List[SeedResult] = []
    primary_metric: Optional[str] = None

    for seed in seeds:
        orig_metrics = execute_pipeline(raw_code, seed, timeout=timeout)
        patch_metrics = execute_pipeline(patched_code, seed, timeout=timeout)

        # Select primary metric on first seed
        if primary_metric is None:
            for m in METRIC_PRIORITY:
                if m in orig_metrics and m in patch_metrics:
                    primary_metric = m
                    break
            if primary_metric is None:
                # Use whatever common key exists
                common = set(orig_metrics.keys()).intersection(set(patch_metrics.keys()))
                primary_metric = next(iter(common)) if common else "metric"

        orig_val = orig_metrics.get(primary_metric, 0.0)
        patch_val = patch_metrics.get(primary_metric, 0.0)
        delta = patch_val - orig_val

        seed_runs.append(
            SeedResult(
                seed=seed,
                original_metric=round(orig_val, 4),
                patched_metric=round(patch_val, 4),
                delta=round(delta, 4),
            )
        )

    # 3. Calculate statistics
    orig_vals = [r.original_metric for r in seed_runs]
    patch_vals = [r.patched_metric for r in seed_runs]
    deltas = [r.delta for r in seed_runs]

    orig_mean = float(np.mean(orig_vals))
    orig_std = float(np.std(orig_vals))
    patch_mean = float(np.mean(patch_vals))
    patch_std = float(np.std(patch_vals))
    mean_delta = float(np.mean(deltas))

    # Noise standard deviation
    seed_noise_std = float(np.sqrt(orig_std**2 + patch_std**2))

    # 4. Determine confirmation status
    # Leakage is confirmed if the patched score dropped significantly below the inflated original score
    is_drop = mean_delta < -0.04
    exceeds_noise = (abs(mean_delta) > (2.0 * seed_noise_std)) or (abs(mean_delta) > 0.08)

    if is_drop and exceeds_noise:
        status = FindingStatus.CONFIRMED
        is_significant = True
        reason = (
            f"CONFIRMED evaluation leakage: Correcting the flaw caused an empirical metric drop of "
            f"{mean_delta:.4f} in {primary_metric} (from {orig_mean:.4f} down to {patch_mean:.4f}) "
            f"across {len(seeds)} random seeds, exceeding seed noise (sigma={seed_noise_std:.4f})."
        )
    elif abs(mean_delta) <= seed_noise_std:
        status = FindingStatus.NOT_CONFIRMED
        is_significant = False
        reason = (
            f"NOT_CONFIRMED: Metric difference of {mean_delta:.4f} is within normal random seed "
            f"variance (sigma={seed_noise_std:.4f}). No measurable evaluation leakage detected."
        )
    else:
        status = FindingStatus.SUSPECTED
        is_significant = False
        reason = (
            f"SUSPECTED: Observed delta of {mean_delta:.4f} in {primary_metric}, but the shift did not "
            f"decisively exceed the 2*sigma seed noise threshold (sigma={seed_noise_std:.4f})."
        )

    return VerificationResult(
        status=status,
        primary_metric_name=primary_metric,
        original_mean=round(orig_mean, 4),
        original_std=round(orig_std, 4),
        patched_mean=round(patch_mean, 4),
        patched_std=round(patch_std, 4),
        mean_delta=round(mean_delta, 4),
        seed_noise_std=round(seed_noise_std, 4),
        is_significant=is_significant,
        seed_runs=seed_runs,
        original_code=raw_code,
        patched_code=patched_code,
        diff=diff,
        reason=reason,
    )
