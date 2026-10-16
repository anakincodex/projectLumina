"""Deterministic evidence tools module for REPROVE (no LLM involved)."""
from typing import List

from reprove.parser import parse_file
from reprove.tools.check_fit_before_split import check_fit_before_split
from reprove.tools.check_split_overlap import (
    check_image_folders_overlap,
    check_split_overlap,
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
    "check_split_overlap",
    "check_split_overlap_arrays",
    "check_split_overlap_graph",
    "check_image_folders_overlap",
    "compute_array_overlap",
    "trace_test_usage",
    "inspect_code_span",
    "run_all_deterministic_checks",
]


def run_all_deterministic_checks(file_path: str, seed: int = 0) -> List[EvidenceObject]:
    """Run every evidence tool on a .py or .ipynb file (Types 1-3)."""
    graph = parse_file(file_path)
    evidence: List[EvidenceObject] = []
    evidence.extend(check_fit_before_split(graph))
    evidence.extend(check_split_overlap(file_path, graph, seed=seed))
    evidence.extend(trace_test_usage(graph))
    return evidence
