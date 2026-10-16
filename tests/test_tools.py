"""Tests for the deterministic evidence tools."""
import numpy as np
from PIL import Image

from reprove.parser import parse_file
from reprove.tools import (
    ToolStatus,
    check_fit_before_split,
    check_image_folders_overlap,
    check_split_overlap,
    check_split_overlap_arrays,
    check_split_overlap_graph,
    inspect_code_span,
    run_all_deterministic_checks,
    trace_test_usage,
)


def _graph(tmp_path, code, name="case.py"):
    p = tmp_path / name
    p.write_text(code, encoding="utf-8")
    return str(p), parse_file(str(p))


def test_fit_before_split_example_1_is_one_grouped_finding():
    graph = parse_file("examples/01_clear_leak_scaler_before_split.ipynb")
    findings = check_fit_before_split(graph)
    assert len(findings) == 1
    ev = findings[0]
    assert ev.status == ToolStatus.FOUND_LEAK
    assert "StandardScaler" in ev.summary and "SelectKBest" in ev.summary
    assert len(ev.metrics["fit_lines"]) == 2
    assert ev.offending_span is not None and ev.offending_span.cell_index is not None


def test_clean_pipeline_has_no_findings():
    for ev in run_all_deterministic_checks("examples/02_clean_pipeline_no_leak.ipynb"):
        assert ev.status == ToolStatus.CLEAN, ev.summary


def test_label_encoding_the_target_is_not_flagged(tmp_path):
    _, graph = _graph(tmp_path, (
        "import numpy as np\n"
        "from sklearn.preprocessing import LabelEncoder\n"
        "from sklearn.model_selection import train_test_split\n"
        "X = np.random.randn(20, 2); labels = np.array(['a', 'b'] * 10)\n"
        "le = LabelEncoder()\n"
        "y = le.fit_transform(labels)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    ))
    assert check_fit_before_split(graph)[0].status == ToolStatus.CLEAN


def test_split_overlap_hashes_the_real_split_of_example_3():
    path = "examples/03_subtle_leak_duplicate_split.ipynb"
    ev = check_split_overlap(path, parse_file(path), seed=0)[0]
    assert ev.status == ToolStatus.FOUND_LEAK
    assert ev.metrics["check"] == "runtime_row_hash"
    assert ev.metrics["duplicate_count"] > 0


def test_concat_of_columns_is_not_a_duplicate_leak():
    path = "examples/stress/06_concat_columns_clean.py"
    graph = parse_file(path)
    assert check_split_overlap(path, graph)[0].status == ToolStatus.CLEAN
    assert check_split_overlap_graph(graph)[0].status == ToolStatus.CLEAN


def test_split_overlap_arrays():
    X_raw = np.random.randn(50, 4)
    ev = check_split_overlap_arrays(X_raw[:35], np.vstack([X_raw[35:], X_raw[:10]]))
    assert ev.status == ToolStatus.FOUND_LEAK and ev.metrics["duplicate_count"] == 10


def test_test_set_used_for_fit(tmp_path):
    _, graph = _graph(tmp_path, (
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "X_train, X_test, y_train, y_test = train_test_split([[1], [2], [3], [4]], [0, 1, 0, 1])\n"
        "clf = LogisticRegression()\n"
        "clf.fit(X_test, y_test)\n"
    ))
    leaks = [f for f in trace_test_usage(graph) if f.status == ToolStatus.FOUND_LEAK]
    assert len(leaks) == 1 and leaks[0].metrics["pattern"] == "fit_on_test"


def test_test_set_used_for_model_selection_loop():
    graph = parse_file("examples/04_test_set_model_selection.ipynb")
    leaks = [f for f in trace_test_usage(graph) if f.status == ToolStatus.FOUND_LEAK]
    assert len(leaks) == 1 and leaks[0].metrics["pattern"] == "selection_loop_on_test"
    assert leaks[0].offending_span.cell_index is not None


def test_test_set_used_for_early_stopping(tmp_path):
    _, graph = _graph(tmp_path, (
        "from sklearn.model_selection import train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        "model.fit(X_train, y_train, eval_set=[(X_test, y_test)])\n"
    ))
    leaks = [f for f in trace_test_usage(graph) if f.status == ToolStatus.FOUND_LEAK]
    assert leaks and leaks[0].metrics["pattern"] == "test_as_validation"


def test_image_folder_overlap(tmp_path):
    train_dir, test_dir = tmp_path / "train", tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    a[::4, :] = 255
    Image.fromarray(a).save(train_dir / "img1.png")
    Image.fromarray(a).save(test_dir / "img2.png")
    b = np.zeros((32, 32, 3), dtype=np.uint8)
    b[:, ::4] = 255
    Image.fromarray(b).save(test_dir / "img3.png")
    ev = check_image_folders_overlap(str(train_dir), str(test_dir))
    assert ev.status == ToolStatus.FOUND_LEAK
    assert ev.metrics["duplicate_count"] == 1 and ev.metrics["test_image_count"] == 2


def test_inspect_code_span():
    span = inspect_code_span("examples/01_clear_leak_scaler_before_split.ipynb", start_line=10, end_line=12)
    assert span.start_line <= 10 and span.end_line >= 12 and span.code_snippet
