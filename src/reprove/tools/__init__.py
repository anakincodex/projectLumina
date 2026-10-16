"""Deterministic evidence tools module for REPROVE."""
from typing import List

from reprove.parser import parse_file
from reprove.tools.check_fit_before_split import check_fit_before_split
from reprove.tools.check_split_overlap import (
    check_image_folders_overlap,
    check_split_overlap_arrays,
    check_split_overlap_graph,
    compute_array_overlap,
)
from reprove.tools.inspect_code_span import inspect_code_span
from reprove.tools.models import EvidenceObject, ToolStatus
from reprove.tools.trace_test_usage import trace_test_usage

__all__ = [
    "ToolStatus",
    "EvidenceObject",
    "check_fit_before_split",
    "check_split_overlap_arrays",
    "check_split_overlap_graph",
    "check_image_folders_overlap",
    "compute_array_overlap",
    "trace_test_usage",
    "inspect_code_span",
    "run_all_deterministic_checks",
]


def run_all_deterministic_checks(file_path: str) -> List[EvidenceObject]:
    """
    Executes the full suite of static and graph-based deterministic checks
    on a target ML experiment file (.py or .ipynb).
    """
    graph = parse_file(file_path)
    evidence: List[EvidenceObject] = []

    # 1. Type 1: Fit before split
    evidence.extend(check_fit_before_split(graph))

    # 2. Type 2: Premature duplication in graph
    evidence.extend(check_split_overlap_graph(graph))

    # 3. Type 3: Test set usage in training/tuning
    evidence.extend(trace_test_usage(graph))

    return evidence
