"""Deterministic fallback auditor, used when the open-weight LLM is offline or returns invalid output."""
from typing import List, Optional

from reprove.agent.audit_core import AuditState
from reprove.agent.schemas import Hypothesis, InvestigationReport
from reprove.verifier import DEFAULT_SEEDS

_PLAN = [
    (
        "check_fit_before_split",
        "TYPE_1_FIT_BEFORE_SPLIT",
        "Check whether scalers, encoders, imputers or feature selectors are fitted before the split.",
    ),
    (
        "check_split_overlap",
        "TYPE_2_CROSS_SPLIT_OVERLAP",
        "Hash the rows of the real train/test split and look for test rows copied from training.",
    ),
    (
        "trace_test_usage",
        "TYPE_3_TEST_SET_CONTAMINATION",
        "Check whether the test split is used for training, early stopping or model selection.",
    ),
]


def deterministic_fallback_audit(
    file_path: str,
    seeds: Optional[List[int]] = None,
    timeout: int = 60,
    reason: Optional[str] = None,
    model_name: Optional[str] = None,
) -> InvestigationReport:
    """Run all three core checks in a fixed order, verifying every finding with the controlled rerun."""
    state = AuditState(file_path, seeds=seeds or DEFAULT_SEEDS, timeout=timeout)
    for i, (tool, leak_type, description) in enumerate(_PLAN, start=1):
        hyp = Hypothesis(hypothesis_id=f"hyp_{i}", leak_type=leak_type, description=description, tool_to_call=tool)
        state.hypotheses.append(hyp)
        state.run_tool(tool, thought=f"Fixed plan: {description}", origin="fallback", hypothesis=hyp)
    return state.to_report(
        "FALLBACK_DETERMINISTIC",
        model_name=model_name,
        fallback_reason=reason or "Deterministic plan requested.",
    )
