"""Deterministic fallback auditor when open-weight LLM is offline or unconfigured."""
from typing import List, Optional

from reprove.agent.schemas import AgentStep, Hypothesis, InvestigationReport
from reprove.parser import parse_file
from reprove.tools import (
    ToolStatus,
    check_fit_before_split,
    check_split_overlap_graph,
    trace_test_usage,
)
from reprove.verifier import FindingStatus, verify_leakage

DEFAULT_SEEDS = [42, 101, 7]


def deterministic_fallback_audit(
    file_path: str,
    seeds: Optional[List[int]] = None,
) -> InvestigationReport:
    """
    Executes a structured deterministic audit without requiring an LLM:
    1. Traverses the AST experiment graph.
    2. Runs deterministic tools for Types 1, 2, and 3.
    3. Runs multi-seed verification on any flagged findings.
    """
    seeds = seeds or DEFAULT_SEEDS
    graph = parse_file(file_path)

    steps: List[AgentStep] = []
    hypotheses: List[Hypothesis] = []
    confirmed_evidence = []
    suspected_evidence = []
    clean_checks = []
    verification_results = []

    step_counter = 0

    # 1. Check Type 1: Fit before split
    step_counter += 1
    fit_findings = check_fit_before_split(graph)
    hyp1 = Hypothesis(
        hypothesis_id="hyp_1",
        leak_type="TYPE_1_FIT_BEFORE_SPLIT",
        description="Check if transformers or feature selectors are fitted prior to train/test split.",
        tool_to_call="check_fit_before_split",
    )
    hypotheses.append(hyp1)

    for ev in fit_findings:
        if ev.status == ToolStatus.FOUND_LEAK:
            steps.append(
                AgentStep(
                    step_number=step_counter,
                    thought="Investigating potential Type 1 preprocessing fit before split.",
                    tool_name="check_fit_before_split",
                    observation=ev.summary,
                    evidence_id=ev.evidence_id,
                )
            )
            # Run verification
            v_res = verify_leakage(file_path, ev, seeds=seeds)
            if v_res.status == FindingStatus.CONFIRMED:
                confirmed_evidence.append(ev)
                verification_results.append(v_res)
            else:
                suspected_evidence.append(ev)
                if v_res.diff:
                    verification_results.append(v_res)
        elif ev.status == ToolStatus.CLEAN:
            clean_checks.append(ev)

    # 2. Check Type 2: Split overlap
    step_counter += 1
    overlap_findings = check_split_overlap_graph(graph)
    hyp2 = Hypothesis(
        hypothesis_id="hyp_2",
        leak_type="TYPE_2_CROSS_SPLIT_OVERLAP",
        description="Check if premature duplication or oversampling occurred before split.",
        tool_to_call="check_split_overlap",
    )
    hypotheses.append(hyp2)

    for ev in overlap_findings:
        if ev.status == ToolStatus.FOUND_LEAK:
            steps.append(
                AgentStep(
                    step_number=step_counter,
                    thought="Investigating cross-split sample overlap or premature duplication.",
                    tool_name="check_split_overlap",
                    observation=ev.summary,
                    evidence_id=ev.evidence_id,
                )
            )
            # Run verification
            v_res = verify_leakage(file_path, ev, seeds=seeds)
            if v_res.status == FindingStatus.CONFIRMED:
                confirmed_evidence.append(ev)
                verification_results.append(v_res)
            else:
                suspected_evidence.append(ev)
                if v_res.diff:
                    verification_results.append(v_res)
        elif ev.status == ToolStatus.CLEAN:
            clean_checks.append(ev)

    # 3. Check Type 3: Test set usage
    step_counter += 1
    usage_findings = trace_test_usage(graph)
    hyp3 = Hypothesis(
        hypothesis_id="hyp_3",
        leak_type="TYPE_3_TEST_SET_CONTAMINATION",
        description="Check if test set variables are used in training, tuning, or early stopping.",
        tool_to_call="trace_test_usage",
    )
    hypotheses.append(hyp3)

    for ev in usage_findings:
        if ev.status == ToolStatus.FOUND_LEAK:
            steps.append(
                AgentStep(
                    step_number=step_counter,
                    thought="Checking for test set leakage into model training or selection.",
                    tool_name="trace_test_usage",
                    observation=ev.summary,
                    evidence_id=ev.evidence_id,
                )
            )
            suspected_evidence.append(ev)
        elif ev.status == ToolStatus.CLEAN:
            clean_checks.append(ev)

    return InvestigationReport(
        file_path=file_path,
        execution_mode="FALLBACK_DETERMINISTIC",
        steps=steps,
        hypotheses_evaluated=hypotheses,
        confirmed_evidence=confirmed_evidence,
        suspected_evidence=suspected_evidence,
        clean_checks=clean_checks,
        verification_results=verification_results,
    )
