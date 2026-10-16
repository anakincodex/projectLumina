"""Tests for the Controlled Verifier Engine, patcher, and multi-seed evaluator."""
import pytest
from reprove.parser import parse_file
from reprove.tools import check_fit_before_split, check_split_overlap_graph
from reprove.verifier import (
    FindingStatus,
    execute_pipeline,
    generate_patch,
    verify_leakage,
)


def test_patcher_generates_valid_diff_for_type_1():
    """Verify patcher creates a unified diff that eliminates fit-before-split."""
    graph = parse_file("examples/01_clear_leak_scaler_before_split.ipynb")
    evidence = check_fit_before_split(graph)[0]

    patched_code, diff = generate_patch(graph.raw_code, evidence)
    assert patched_code is not None
    assert diff is not None
    assert "--- original_leaked.py" in diff
    assert "+++ reprove_patched.py" in diff
    assert "scaler.fit_transform(X_train)" in patched_code
    assert "scaler.transform(X_test)" in patched_code


def test_patcher_generates_dedup_for_type_2():
    """Verify patcher creates deduplication logic for duplicate split leak."""
    graph = parse_file("examples/03_subtle_leak_duplicate_split.ipynb")
    evidence = check_split_overlap_graph(graph)[0]

    patched_code, diff = generate_patch(graph.raw_code, evidence)
    assert patched_code is not None
    assert diff is not None
    assert "np.unique" in patched_code


def test_execute_pipeline_captures_metrics_and_respects_seed():
    """Verify subprocess runner executes code and extracts metric dictionary."""
    code = (
        "import numpy as np\n"
        "from sklearn.metrics import accuracy_score\n"
        "np.random.seed(42)\n"
        "val = np.random.rand()\n"
        "test_acc = float(val)\n"
        "print(f'Done: {test_acc}')\n"
    )
    metrics_42 = execute_pipeline(code, seed=42)
    metrics_99 = execute_pipeline(code, seed=99)

    assert "test_acc" in metrics_42
    assert "test_acc" in metrics_99
    # Different seeds should produce different random values
    assert metrics_42["test_acc"] != metrics_99["test_acc"]


def test_verify_leakage_confirms_example_1():
    """Verify end-to-end multi-seed verification confirms the planted leak in Example 1."""
    graph = parse_file("examples/01_clear_leak_scaler_before_split.ipynb")
    evidence = check_fit_before_split(graph)[0]

    # Run verification with 3 seeds for fast test execution
    result = verify_leakage(
        "examples/01_clear_leak_scaler_before_split.ipynb",
        evidence,
        seeds=[42, 101, 7],
    )

    assert result.status == FindingStatus.CONFIRMED
    assert result.is_significant is True
    assert result.primary_metric_name in {"test_auc", "roc_auc"}
    assert result.original_mean > 0.70
    assert result.patched_mean < 0.60
    assert result.mean_delta < -0.15
    assert len(result.seed_runs) == 3
    assert len(result.diff) > 0
    assert "CONFIRMED" in result.reason
