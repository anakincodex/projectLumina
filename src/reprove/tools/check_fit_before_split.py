"""Deterministic tool: preprocessing fitted on the full dataset before the train/test split (Type 1)."""
import ast
import uuid
from typing import List, Optional, Set

from reprove.parser.models import CodeSpan, ExperimentGraph, PipelineNode, StageType
from reprove.tools.models import EvidenceObject, ToolStatus

# Fitting these on the full label vector does not leak test information in a meaningful way.
TARGET_ONLY_ENCODERS = {"LabelEncoder", "LabelBinarizer", "MultiLabelBinarizer"}


def _loc(span: CodeSpan) -> str:
    if span.cell_index is not None:
        return f"cell [{span.cell_index}] line {span.cell_line}"
    return f"line {span.start_line}"


def _split_target_name(graph: ExperimentGraph, split: PipelineNode) -> Optional[str]:
    """Name of the 2nd positional argument of train_test_split (usually `y`), if it is a plain name."""
    try:
        stmt = ast.parse(split.code_span.code_snippet.strip()).body[0]
        call = stmt.value if isinstance(stmt, (ast.Assign, ast.Expr)) else None
        if isinstance(call, ast.Call) and len(call.args) > 1 and isinstance(call.args[1], ast.Name):
            return call.args[1].id
    except Exception:
        pass
    return None


def _combined_span(graph: ExperimentGraph, nodes: List[PipelineNode]) -> CodeSpan:
    first, last = nodes[0].code_span, nodes[-1].code_span
    return CodeSpan(
        file_path=first.file_path,
        start_line=first.start_line,
        end_line=last.end_line,
        cell_index=first.cell_index,
        cell_line=first.cell_line,
        code_snippet="\n".join(n.code_span.code_snippet for n in nodes),
    )


def check_fit_before_split(graph: ExperimentGraph) -> List[EvidenceObject]:
    """
    One finding per experiment: every preprocessing fit (scaler, imputer, encoder, PCA,
    feature selector, ...) that runs before the first train/test split.
    """
    splits = sorted(graph.find_splits(), key=lambda n: n.code_span.start_line)
    if not splits:
        return [
            EvidenceObject(
                evidence_id=f"ev_fit_split_{uuid.uuid4().hex[:8]}",
                tool_name="check_fit_before_split",
                status=ToolStatus.INCONCLUSIVE,
                confidence=0.0,
                summary="No train_test_split / CV splitter found, so fit-before-split could not be checked.",
            )
        ]

    split = splits[0]
    split_line = split.code_span.start_line
    target_name = _split_target_name(graph, split)

    leaky: List[PipelineNode] = []
    for node in graph.nodes:
        if not node.is_fit_call or node.stage != StageType.PREPROCESSING:
            continue
        if node.code_span.start_line >= split_line:
            continue
        if node.caller_class in TARGET_ONLY_ENCODERS:
            continue
        if target_name and target_name in node.outputs:
            continue  # encoding the label vector itself
        leaky.append(node)

    if not leaky:
        return [
            EvidenceObject(
                evidence_id=f"ev_fit_split_{uuid.uuid4().hex[:8]}",
                tool_name="check_fit_before_split",
                status=ToolStatus.CLEAN,
                confidence=1.0,
                summary=f"No preprocessing is fitted before the train/test split at {_loc(split.code_span)}.",
                context_span=split.code_span,
            )
        ]

    names: List[str] = []
    for n in leaky:
        who = n.caller_class or n.caller or n.function_name or "transformer"
        var = f" (`{n.caller}`)" if n.caller and n.caller_class else ""
        names.append(f"{who}{var} at {_loc(n.code_span)}")

    inputs: Set[str] = set()
    for n in leaky:
        inputs.update(n.inputs)

    return [
        EvidenceObject(
            evidence_id=f"ev_fit_before_split_{uuid.uuid4().hex[:8]}",
            tool_name="check_fit_before_split",
            status=ToolStatus.FOUND_LEAK,
            confidence=1.0,
            summary=(
                f"Type 1 leakage pattern: {len(leaky)} preprocessing step(s) are fitted on the full dataset "
                f"before the train/test split at {_loc(split.code_span)}: {'; '.join(names)}. "
                f"Statistics of the test rows therefore shape the training features."
            ),
            offending_span=_combined_span(graph, leaky),
            context_span=split.code_span,
            metrics={
                "leak_type": "TYPE_1_FIT_BEFORE_SPLIT",
                "transformers": [n.caller_class or n.caller or n.function_name for n in leaky],
                "fit_lines": [n.code_span.start_line for n in leaky],
                "split_line": split_line,
                "fit_inputs": sorted(inputs),
            },
            suggested_fix=(
                "Split first, then fit each transformer on the training split only and apply it to the test "
                "split - or wrap the steps in a sklearn Pipeline so fitting happens inside the split."
            ),
        )
    ]
