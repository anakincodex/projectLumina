"""Data models for controlled experiment verification, seeds, and empirical metric shifts."""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class FindingStatus(str, Enum):
    """Status of an audited leakage finding."""
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
    Non-negotiable rule: Every number comes from real executed outputs.
    """
    status: FindingStatus
    primary_metric_name: str
    original_mean: float
    original_std: float
    patched_mean: float
    patched_std: float
    mean_delta: float
    seed_noise_std: float
    is_significant: bool
    seed_runs: List[SeedResult] = Field(default_factory=list)
    original_code: str
    patched_code: str
    diff: str
    reason: str
