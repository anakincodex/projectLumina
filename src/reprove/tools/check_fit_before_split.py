"""Deterministic tool to detect preprocessing operations fitted before train/test split."""
import uuid
from typing import List, Optional

from reprove.parser.models import ExperimentGraph, StageType
from reprove.tools.models import EvidenceObject, ToolStatus


def check_fit_before_split(graph: ExperimentGraph) -> List[EvidenceObject]:
    """
    Examines the ExperimentGraph for Type 1 leakage:
    Transformers (scalers, encoders, imputers, selectors, PCA) fitted before splitting.
    """
    splits = graph.find_splits()
    if not splits:
        return [
            EvidenceObject(
                evidence_id=f"ev_fit_split_{uuid.uuid4().hex[:8]}",
                tool_name="check_fit_before_split",
                status=ToolStatus.INCONCLUSIVE,
                confidence=0.0,
                summary="No train_test_split or cross-validation splitting found in the experiment graph.",
            )
        ]

    findings: List[EvidenceObject] = []

    # Check each split against preceding fits
    for split_node in splits:
        split_line = split_node.code_span.start_line
        split_inputs = set(split_node.inputs)

        for node in graph.nodes:
            # Only consider nodes that execute a fit/fit_transform and occur before split
            if not node.is_fit_call or node.stage != StageType.PREPROCESSING:
                continue

            if node.code_span.start_line < split_line:
                # Direct dataflow or sequential leakage
                class_desc = f" ({node.caller_class})" if node.caller_class else ""
                transformer_name = f"{node.caller or node.function_name or 'Transformer'}{class_desc}"
                ev_id = f"ev_fit_before_split_{uuid.uuid4().hex[:8]}"
                findings.append(
                    EvidenceObject(
                        evidence_id=ev_id,
                        tool_name="check_fit_before_split",
                        status=ToolStatus.FOUND_LEAK,
                        confidence=1.0,
                        summary=(
                            f"Evaluation Leakage (Type 1): Preprocessing transformer '{transformer_name}' "
                            f"is fitted at line {node.code_span.start_line} before the train/test split "
                            f"at line {split_line}. Information from the test fold leaks into training features."
                        ),
                        offending_span=node.code_span,
                        context_span=split_node.code_span,
                        metrics={
                            "leak_type": "TYPE_1_FIT_BEFORE_SPLIT",
                            "transformer": transformer_name,
                            "fit_line": node.code_span.start_line,
                            "split_line": split_line,
                            "fit_inputs": node.inputs,
                            "fit_outputs": node.outputs,
                        },
                        suggested_fix=(
                            f"Refactor '{transformer_name}' to fit only on X_train after splitting, "
                            f"or encapsulate inside a sklearn.pipeline.Pipeline."
                        ),
                    )
                )

    if not findings:
        return [
            EvidenceObject(
                evidence_id=f"ev_fit_split_{uuid.uuid4().hex[:8]}",
                tool_name="check_fit_before_split",
                status=ToolStatus.CLEAN,
                confidence=1.0,
                summary="No preprocessing fits detected prior to dataset splitting. Split precedes transformations.",
            )
        ]

    return findings
