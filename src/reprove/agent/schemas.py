"""Pydantic schemas for the investigation agent, its tool calls, and the final report."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from reprove.tools.models import EvidenceObject
from reprove.verifier.models import VerificationResult

ToolName = Literal[
    "check_fit_before_split",
    "check_split_overlap",
    "trace_test_usage",
    "inspect_code_span",
    "finish",
]
LeakType = Literal[
    "TYPE_1_FIT_BEFORE_SPLIT",
    "TYPE_2_CROSS_SPLIT_OVERLAP",
    "TYPE_3_TEST_SET_CONTAMINATION",
    "TYPE_4_TARGET_LEAKAGE",
    "NONE",
]
CORE_TOOLS: List[str] = ["check_fit_before_split", "check_split_overlap", "trace_test_usage"]


class Hypothesis(BaseModel):
    """A hypothesis formulated by the LLM (or the fallback plan) about possible leakage."""
    hypothesis_id: str
    leak_type: LeakType
    description: str
    suspected_line: Optional[int] = None
    tool_to_call: ToolName
    tool_args: Dict[str, Any] = Field(default_factory=dict)


class AgentAction(BaseModel):
    """What the LLM must return at every step (also used as the Ollama structured-output schema)."""
    thought: str = Field(description="Short reasoning about the code and which check to run next")
    hypothesis: Hypothesis


class AgentStep(BaseModel):
    """One executed step of the investigation."""
    step_number: int
    origin: Literal["agent", "coverage", "fallback"] = "agent"
    thought: str
    tool_name: str
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    observation: str
    result: str = ""                 # e.g. "CONFIRMED", "CLEAN", "SUSPECTED", "INCONCLUSIVE", "CONCLUDED"
    evidence_id: Optional[str] = None


class InvestigationReport(BaseModel):
    """
    Final output of an audit.
    Invariant: every finding is an EvidenceObject produced by a deterministic tool, and its status
    (confirmed / not confirmed / suspected) is set by code from the controlled rerun, never by the LLM.
    """
    file_path: str
    execution_mode: str  # "LLM_OLLAMA" | "FALLBACK_DETERMINISTIC" | "REPLAY_CACHE"
    replayed_from: Optional[str] = None      # original mode of a replayed run
    recorded_at: Optional[str] = None
    model_name: Optional[str] = None
    fallback_reason: Optional[str] = None
    seeds: List[int] = Field(default_factory=list)
    steps: List[AgentStep] = Field(default_factory=list)
    hypotheses_evaluated: List[Hypothesis] = Field(default_factory=list)
    confirmed_evidence: List[EvidenceObject] = Field(default_factory=list)
    not_confirmed_evidence: List[EvidenceObject] = Field(default_factory=list)
    suspected_evidence: List[EvidenceObject] = Field(default_factory=list)
    clean_checks: List[EvidenceObject] = Field(default_factory=list)
    inconclusive_checks: List[EvidenceObject] = Field(default_factory=list)
    verification_results: List[VerificationResult] = Field(default_factory=list)

    @property
    def has_confirmed_leaks(self) -> bool:
        return len(self.confirmed_evidence) > 0

    @property
    def verdict(self) -> str:
        """CONFIRMED > SUSPECTED > NOT_CONFIRMED > INCONCLUSIVE > CLEAN."""
        if self.confirmed_evidence:
            return "CONFIRMED"
        if self.suspected_evidence:
            return "SUSPECTED"
        if self.not_confirmed_evidence:
            return "NOT_CONFIRMED"
        if self.inconclusive_checks:
            return "INCONCLUSIVE"
        return "CLEAN"

    @property
    def effective_mode(self) -> str:
        """The mode that actually produced the findings (looks through replay)."""
        return self.replayed_from or self.execution_mode

    def verification_for(self, evidence_id: str) -> Optional[VerificationResult]:
        return next((v for v in self.verification_results if v.evidence_id == evidence_id), None)
