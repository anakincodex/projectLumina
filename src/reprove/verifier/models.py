"""Data models for controlled experiment verification, seeds, and empirical metric shifts."""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

SIGNIFICANCE_LEVEL = 0.05
NOISE_RULE = (
    "CONFIRMED only if the corrected pipeline scores worse on average AND a one-sided paired "
    "t-test over the per-seed differences gives p < 0.05 (minimum 3 seeds)."
)


def format_p(p: Optional[float]) -> str:
    """Human-friendly p-value."""
    if p is None:
        return "n/a"
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


class FindingStatus(str, Enum):
    """Status of an audited leakage finding.

    CONFIRMED      - a controlled rerun measured a drop larger than seed noise.
    NOT_CONFIRMED  - the rerun ran, but the drop is within seed noise (or the score did not drop).
    SUSPECTED      - a deterministic tool found the pattern, but the rerun could not be run.
    """
    CONFIRMED = "CONFIRMED"
    SUSPECTED = "SUSPECTED"
    NOT_CONFIRMED = "NOT_CONFIRMED"


class SeedResult(BaseModel):
    """Metrics recorded for a single random seed run."""
    seed: int
    original_metric: float
    patched_metric: float
    delta: float  # patched_metric - original_metric


class VerificationResult(BaseModel):
    """
    Empirical output of a controlled rerun comparing original vs patched pipelines.
    Non-negotiable rule: every number comes from real executed outputs. When a rerun
    could not run, verification_ran is False and all numeric fields stay at 0 with a reason.
    """
    evidence_id: Optional[str] = None
    status: FindingStatus
    verification_ran: bool = False
    primary_metric_name: str = "unknown"
    higher_is_better: bool = True
    original_mean: float = 0.0
    original_std: float = 0.0
    patched_mean: float = 0.0
    patched_std: float = 0.0
    mean_delta: float = 0.0          # mean(patched - original)
    seed_noise_std: float = 0.0      # std (ddof=1) of the paired per-seed differences
    standard_error: float = 0.0      # seed_noise_std / sqrt(n_seeds)
    p_value: Optional[float] = None  # one-sided paired t-test: "the fix makes the score worse"
    is_significant: bool = False
    seed_runs: List[SeedResult] = Field(default_factory=list)
    original_code: str = ""
    patched_code: str = ""
    diff: str = ""
    reason: str = ""
    noise_rule: str = NOISE_RULE
