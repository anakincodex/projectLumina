"""Deterministic tool to trace test set usage in model selection, tuning, or training (Type 3)."""
import uuid
from typing import List, Set

from reprove.parser.models import ExperimentGraph, StageType
from reprove.tools.models import EvidenceObject, ToolStatus


def trace_test_usage(graph: ExperimentGraph) -> List[EvidenceObject]:
    """
    Examines the ExperimentGraph for Type 3 leakage:
    Test set variables passed into training, hyperparameter tuning, or model selection.
    """
    splits = graph.find_splits()
    if not splits:
        return [
            EvidenceObject(
                evidence_id=f"ev_test_usage_{uuid.uuid4().hex[:8]}",
                tool_name="trace_test_usage",
                status=ToolStatus.INCONCLUSIVE,
                confidence=0.0,
                summary="No split nodes found in experiment graph to identify test set variables.",
            )
        ]

    # Collect test variables produced by split operations
    test_variables: Set[str] = set()
    for split_node in splits:
        for out_var in split_node.outputs:
            if "test" in out_var.lower() or "val" in out_var.lower():
                test_variables.add(out_var)

    if not test_variables:
        # Fallback: scikit-learn standard second/fourth outputs from train_test_split
        for split_node in splits:
            if len(split_node.outputs) >= 2:
                test_variables.add(split_node.outputs[1])  # X_test
            if len(split_node.outputs) >= 4:
                test_variables.add(split_node.outputs[3])  # y_test

    findings: List[EvidenceObject] = []

    for node in graph.nodes:
        # We only flag when test variables are fed into training or tuning operations
        if node.stage != StageType.MODEL_TRAINING:
            continue

        # Check if any test variable is passed as an input to the training operation
        contaminating_vars = test_variables.intersection(set(node.inputs))
        if contaminating_vars:
            ev_id = f"ev_test_usage_{uuid.uuid4().hex[:8]}"
            findings.append(
                EvidenceObject(
                    evidence_id=ev_id,
                    tool_name="trace_test_usage",
                    status=ToolStatus.FOUND_LEAK,
                    confidence=1.0,
                    summary=(
                        f"Evaluation Leakage (Type 3): Test set variable(s) {sorted(list(contaminating_vars))} "
                        f"passed into model training/tuning operation '{node.label}' at line {node.code_span.start_line}. "
                        f"The test set is directly informing parameter estimation or model selection."
                    ),
                    offending_span=node.code_span,
                    metrics={
                        "leak_type": "TYPE_3_TEST_SET_CONTAMINATION",
                        "operation_line": node.code_span.start_line,
                        "contaminating_variables": sorted(list(contaminating_vars)),
                        "caller": node.caller,
                        "function": node.function_name,
                    },
                    suggested_fix=(
                        "Use cross-validation on the training set (e.g., cross_val_score or GridSearchCV(cv=5)) "
                        "for tuning and reserve the test set strictly for final evaluation."
                    ),
                )
            )

    if not findings:
        return [
            EvidenceObject(
                evidence_id=f"ev_test_usage_{uuid.uuid4().hex[:8]}",
                tool_name="trace_test_usage",
                status=ToolStatus.CLEAN,
                confidence=1.0,
                summary="Test set variables are strictly isolated to evaluation operations (no training/tuning contamination).",
            )
        ]

    return findings
