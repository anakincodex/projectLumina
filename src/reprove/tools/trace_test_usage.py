"""Deterministic tool: the test split is used for training, tuning, early stopping or model selection (Type 3).

Patterns detected (from the AST, with exact lines):
  1. a test variable is passed as an early-stopping / validation set  e.g. eval_set=[(X_test, y_test)]
  2. a test variable is passed to a fit / training call               e.g. clf.fit(X_test, y_test)
  3. a loop fits models AND scores them on the test split             e.g. picking the best C on X_test
There is no automatic correction for these, so findings stay SUSPECTED unless a human verifies them.
"""
import ast
import uuid
from typing import List, Optional, Set

from reprove.parser.models import CodeSpan, ExperimentGraph, StageType
from reprove.tools.models import EvidenceObject, ToolStatus

VALIDATION_KEYWORDS = {"eval_set", "validation_data", "valid_sets", "eval_X", "X_val", "validation_split_data"}


def _loc(span: CodeSpan) -> str:
    if span.cell_index is not None:
        return f"cell [{span.cell_index}] line {span.cell_line}"
    return f"line {span.start_line}"


def _test_variables(graph: ExperimentGraph) -> Set[str]:
    """Names of the test outputs of every split (X_test, y_test, or 2nd/4th output by position)."""
    names: Set[str] = set()
    for split in graph.find_splits():
        outs = split.outputs
        named = {o for o in outs if "test" in o.lower()}
        if named:
            names |= named
        elif len(outs) == 4:
            names |= {outs[1], outs[3]}
        elif len(outs) == 2:
            names.add(outs[1])
    return names


def _loaded_names(node: ast.AST) -> Set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _evidence(graph: ExperimentGraph, span: CodeSpan, summary: str, pattern: str, variables: List[str]) -> EvidenceObject:
    return EvidenceObject(
        evidence_id=f"ev_test_usage_{uuid.uuid4().hex[:8]}",
        tool_name="trace_test_usage",
        status=ToolStatus.FOUND_LEAK,
        confidence=1.0,
        summary=summary,
        offending_span=span,
        metrics={
            "leak_type": "TYPE_3_TEST_SET_CONTAMINATION",
            "pattern": pattern,
            "operation_line": span.start_line,
            "contaminating_variables": sorted(variables),
        },
        suggested_fix=(
            "Keep the test split for one final evaluation only. Choose hyperparameters / checkpoints with a "
            "separate validation split or cross-validation on the training data (e.g. GridSearchCV(cv=5))."
        ),
    )


def trace_test_usage(graph: ExperimentGraph) -> List[EvidenceObject]:
    splits = graph.find_splits()
    if not splits:
        return [
            EvidenceObject(
                evidence_id=f"ev_test_usage_{uuid.uuid4().hex[:8]}",
                tool_name="trace_test_usage",
                status=ToolStatus.INCONCLUSIVE,
                confidence=0.0,
                summary="No split found, so test-set usage could not be traced.",
            )
        ]

    test_vars = _test_variables(graph)
    findings: List[EvidenceObject] = []
    flagged_lines: Set[int] = set()

    try:
        tree = ast.parse(graph.raw_code)
    except SyntaxError:
        tree = None

    # 1) Early stopping / validation sets built from the test split (checked first so that
    #    `model.fit(X_train, y_train, eval_set=[(X_test, y_test)])` is labelled precisely)
    if tree is not None:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg in VALIDATION_KEYWORDS:
                    hit = test_vars & _loaded_names(kw.value)
                    if hit and node.lineno not in flagged_lines:
                        flagged_lines.add(node.lineno)
                        span = graph.span_for_lines(node.lineno, getattr(node, "end_lineno", node.lineno))
                        findings.append(_evidence(
                            graph, span,
                            f"Type 3 leakage pattern: the test split {sorted(hit)} is used as `{kw.arg}` "
                            f"(early stopping / validation) at {_loc(span)}.",
                            "test_as_validation", sorted(hit),
                        ))

    # 2) Test variables passed into a training call
    for node in graph.nodes:
        if node.stage != StageType.MODEL_TRAINING or not node.is_fit_call:
            continue
        if node.code_span.start_line in flagged_lines:
            continue
        hit = test_vars & set(node.inputs)
        if hit:
            flagged_lines.add(node.code_span.start_line)
            findings.append(_evidence(
                graph, node.code_span,
                f"Type 3 leakage pattern: test variable(s) {sorted(hit)} are passed to the training call "
                f"`{node.label}` at {_loc(node.code_span)}.",
                "fit_on_test", sorted(hit),
            ))

    # 3) A loop that fits models and scores them on the test split
    if tree is not None:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.For, ast.While)):
                continue
            fits = [
                c for c in ast.walk(node)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "fit"
            ]
            hit = test_vars & _loaded_names(node)
            inner_flagged = any(node.lineno <= ln <= getattr(node, "end_lineno", node.lineno) for ln in flagged_lines)
            if fits and hit and not inner_flagged:
                flagged_lines.add(node.lineno)
                span = graph.span_for_lines(node.lineno, getattr(node, "end_lineno", node.lineno))
                findings.append(_evidence(
                    graph, span,
                    f"Type 3 leakage pattern: a loop at {_loc(span)} fits models and evaluates them on the "
                    f"test split {sorted(hit)}, so the test set is being used to choose the model or its "
                    f"hyperparameters. The reported test score is then optimistically biased.",
                    "selection_loop_on_test", sorted(hit),
                ))

    if findings:
        return findings

    return [
        EvidenceObject(
            evidence_id=f"ev_test_usage_{uuid.uuid4().hex[:8]}",
            tool_name="trace_test_usage",
            status=ToolStatus.CLEAN,
            confidence=1.0,
            summary=(
                f"Test variables {sorted(test_vars)} are only used for evaluation "
                "(no training, early stopping or selection loop uses them)."
            ),
        )
    ]
