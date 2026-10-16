"""Pydantic schemas for the LLM Investigation Agent, hypotheses, and tool interactions."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from reprove.tools.models import EvidenceObject
from reprove.verifier.models import VerificationResult


class Hypothesis(BaseModel):
    """A hypothesis formulated by the LLM regarding possible evaluation leakage."""
    hypothesis_id: str
    leak_type: str = Field(
        description="One of: TYPE_1_FIT_BEFORE_SPLIT, TYPE_2_CROSS_SPLIT_OVERLAP, TYPE_3_TEST_SET_CONTAMINATION, TYPE_4_TARGET_LEAKAGE, NONE"
    )
    description: str
    suspected_line: Optional[int] = None
    tool_to_call: str = Field(
        description="One of: check_fit_before_split, check_split_overlap, trace_test_usage, inspect_code_span, finish"
    )
    tool_args: Dict[str, Any] = Field(default_factory=dict)


class AgentAction(BaseModel):
    """Action chosen by the LLM investigator at a single step."""
    thought: str = Field(description="Step-by-step reasoning about the code and evidence")
    hypothesis: Hypothesis


class AgentStep(BaseModel):
    """An executed step in the agent's bounded tool-calling loop."""
    step_number: int
    thought: str
    tool_name: str
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    observation: str
    evidence_id: Optional[str] = None


class InvestigationReport(BaseModel):
    """
    Final output of an agent investigation run.
    Strict Invariant: Every claim in confirmed_evidence must hold an evidence_id from a tool.
    """
    file_path: str
    execution_mode: str  # "LLM_OLLAMA", "REPLAY_CACHE", "FALLBACK_DETERMINISTIC"
    steps: List[AgentStep] = Field(default_factory=list)
    hypotheses_evaluated: List[Hypothesis] = Field(default_factory=list)
    confirmed_evidence: List[EvidenceObject] = Field(default_factory=list)
    suspected_evidence: List[EvidenceObject] = Field(default_factory=list)
    clean_checks: List[EvidenceObject] = Field(default_factory=list)
    verification_results: List[VerificationResult] = Field(default_factory=list)

    @property
    def has_confirmed_leaks(self) -> bool:
        return len(self.confirmed_evidence) > 0 or any(
            v.status.value == "CONFIRMED" for v in self.verification_results
        )
