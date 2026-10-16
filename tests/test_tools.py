"""Unit and integration tests for deterministic evidence tools."""
import numpy as np
import pytest
from PIL import Image

from reprove.parser import parse_file
from reprove.tools import (
    ToolStatus,
    check_fit_before_split,
    check_image_folders_overlap,
    check_split_overlap_arrays,
    check_split_overlap_graph,
    compute_array_overlap,
    inspect_code_span,
    run_all_deterministic_checks,
    trace_test_usage,
)


def test_tool_01_clear_leak_detection():
    """Verify check_fit_before_split finds the planted leak in Example 1."""
    graph = parse_file("examples/01_clear_leak_scaler_before_split.ipynb")
    findings = check_fit_before_split(graph)

    leaks = [f for f in findings if f.status == ToolStatus.FOUND_LEAK]
    assert len(leaks) >= 1, "Expected at least 1 confirmed leak finding"
    assert any("SelectKBest" in f.summary or "StandardScaler" in f.summary for f in leaks)
    assert leaks[0].confidence == 1.0
    assert leaks[0].offending_span is not None
    assert leaks[0].context_span is not None


def test_tool_02_clean_pipeline_no_false_positives():
    """Verify that all deterministic tools return CLEAN on the negative control notebook."""
    graph = parse_file("examples/02_clean_pipeline_no_leak.ipynb")

    fit_findings = check_fit_before_split(graph)
    assert all(f.status == ToolStatus.CLEAN for f in fit_findings)

    overlap_findings = check_split_overlap_graph(graph)
    assert all(f.status == ToolStatus.CLEAN for f in overlap_findings)

    usage_findings = trace_test_usage(graph)
    assert all(f.status == ToolStatus.CLEAN for f in usage_findings)


def test_tool_03_subtle_leak_split_overlap():
    """Verify that AST graph check and array hashing catch the duplicate samples in Example 3."""
    graph = parse_file("examples/03_subtle_leak_duplicate_split.ipynb")
    graph_findings = check_split_overlap_graph(graph)
    assert any(f.status == ToolStatus.FOUND_LEAK for f in graph_findings)

    # Test array-level hashing on duplicated arrays
    X_raw = np.random.randn(50, 4)
    # Duplicate 10 rows
    X_train = X_raw[:35]
    X_test = np.vstack([X_raw[35:], X_raw[:10]])  # 10 rows in X_test are identical to X_train

    evidence = check_split_overlap_arrays(X_train, X_test)
    assert evidence.status == ToolStatus.FOUND_LEAK
    assert evidence.metrics["duplicate_count"] == 10
    assert evidence.metrics["overlap_percentage"] > 0.0


def test_tool_type_3_test_usage_leak(tmp_path):
    """Verify trace_test_usage catches test set passed into training or tuning."""
    leaked_code = (
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "X_train, X_test, y_train, y_test = train_test_split([1, 2], [3, 4])\n"
        "clf = LogisticRegression()\n"
        "clf.fit(X_test, y_test)\n"  # Contamination!
    )
    p = tmp_path / "type3_leak.py"
    p.write_text(leaked_code, encoding="utf-8")

    graph = parse_file(str(p))
    findings = trace_test_usage(graph)

    leaks = [f for f in findings if f.status == ToolStatus.FOUND_LEAK]
    assert len(leaks) == 1
    assert "X_test" in leaks[0].metrics["contaminating_variables"]
    assert leaks[0].confidence == 1.0


def test_tool_image_folder_overlap(tmp_path):
    """Verify check_image_folders_overlap detects identical images across folders."""
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()

    # Create pattern 1 (horizontal stripes)
    arr1 = np.zeros((32, 32, 3), dtype=np.uint8)
    arr1[::4, :] = 255
    img1 = Image.fromarray(arr1)
    img1.save(train_dir / "img1.png")

    # Identical image in test
    img2 = Image.fromarray(arr1)
    img2.save(test_dir / "img2.png")

    # Distinct pattern (vertical stripes)
    arr3 = np.zeros((32, 32, 3), dtype=np.uint8)
    arr3[:, ::4] = 255
    img3 = Image.fromarray(arr3)
    img3.save(test_dir / "img3.png")

    ev = check_image_folders_overlap(str(train_dir), str(test_dir))
    assert ev.status == ToolStatus.FOUND_LEAK
    assert ev.metrics["duplicate_count"] == 1
    assert ev.metrics["test_image_count"] == 2


def test_inspect_code_span():
    """Verify inspect_code_span extracts correct lines and context."""
    span = inspect_code_span("examples/01_clear_leak_scaler_before_split.ipynb", start_line=10, end_line=12)
    assert span.file_path == "examples/01_clear_leak_scaler_before_split.ipynb"
    assert span.start_line <= 10
    assert span.end_line >= 12
    assert len(span.code_snippet) > 0
