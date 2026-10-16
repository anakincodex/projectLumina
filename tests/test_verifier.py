"""Tests for the patcher, the subprocess runner and the controlled-rerun statistics."""
import ast

from reprove.parser import parse_file
from reprove.tools import check_fit_before_split, check_split_overlap_graph, trace_test_usage
from reprove.tools.models import EvidenceObject, ToolStatus
from reprove.verifier import (
    FindingStatus,
    execute_pipeline,
    generate_patch,
    generate_patch_with_reason,
    verify_leakage,
)


def _type1_evidence(lines):
    return EvidenceObject(evidence_id="ev_t1", tool_name="check_fit_before_split", status=ToolStatus.FOUND_LEAK,
                          confidence=1.0, summary="s", metrics={"leak_type": "TYPE_1_FIT_BEFORE_SPLIT", "fit_lines": lines})


def test_type1_patch_example_1():
    graph = parse_file("examples/01_clear_leak_scaler_before_split.ipynb")
    ev = check_fit_before_split(graph)[0]
    patched, diff = generate_patch(graph.raw_code, ev)
    assert patched and diff
    assert "--- original_leaked.py" in diff and "+++ reprove_patched.py" in diff
    assert "scaler.fit(X_train)" in patched and "scaler.transform(X_test)" in patched
    assert "selector.fit(X_train, y_train)" in patched
    assert "selector = SelectKBest(score_func=f_classif, k=10)" in patched   # user's own k is kept
    ast.parse(patched)


def test_type1_patch_uses_the_users_own_variable_names():
    code = (
        "import numpy as np\n"
        "from sklearn.preprocessing import MinMaxScaler, StandardScaler\n"
        "from sklearn.decomposition import PCA\n"
        "from sklearn.model_selection import train_test_split\n"
        "data = np.random.randn(50, 4); labels = np.arange(50) % 2\n"
        "mm = MinMaxScaler()\n"
        "mm.fit(data)\n"
        "data_n = mm.transform(data)\n"
        "data_p = PCA(n_components=2).fit_transform(data_n)\n"
        "Xtr, Xte, ytr, yte = train_test_split(data_p, labels, random_state=0)\n"
    )
    patched, diff, reason = generate_patch_with_reason(code, _type1_evidence([7, 9]))
    assert patched, reason
    assert "train_test_split(data, labels" in patched
    assert "mm.fit(Xtr)" in patched and "Xte = mm.transform(Xte)" in patched
    assert "_rp_step1 = PCA(n_components=2)" in patched
    exec(compile(patched, "patched", "exec"), {})


def test_type1_patch_refuses_unsupported_code_with_a_reason():
    code = (
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import train_test_split\n"
        "import numpy as np\n"
        "X = np.random.randn(20, 3); y = np.arange(20) % 2\n"
        "Xs = StandardScaler().fit_transform(X)\n"
        "print(Xs.shape)\n"
        "a, b, c, d = train_test_split(Xs, y)\n"
    )
    patched, diff, reason = generate_patch_with_reason(code, _type1_evidence([5]))
    assert patched is None and diff is None
    assert "used elsewhere" in reason


def test_type2_patch_deduplicates_before_split():
    graph = parse_file("examples/03_subtle_leak_duplicate_split.ipynb")
    ev = check_split_overlap_graph(graph)[0]
    patched, diff = generate_patch(graph.raw_code, ev)
    assert patched and ".duplicated()" in patched
    assert "train_test_split(\n    _rp_X, _rp_arr1" in patched
    ast.parse(patched)


def test_type3_has_no_automatic_patch():
    graph = parse_file("examples/04_test_set_model_selection.ipynb")
    ev = trace_test_usage(graph)[0]
    patched, diff, reason = generate_patch_with_reason(graph.raw_code, ev)
    assert patched is None and "not implemented" in reason


def test_runner_captures_variables_and_sklearn_metric_calls():
    code = (
        "import numpy as np\n"
        "from sklearn.metrics import accuracy_score\n"
        "np.random.seed(42)\n"
        "test_acc = float(np.random.rand())\n"
        "print(accuracy_score([1, 0, 1, 1], [1, 0, 0, 1]))\n"   # not stored in a variable
    )
    m42 = execute_pipeline(code, seed=42)
    m99 = execute_pipeline(code, seed=99)
    assert m42["test_acc"] != m99["test_acc"]               # seed injection works
    assert m42["accuracy_score"] == 0.75                    # captured by the harness hook


def test_verify_confirms_example_1():
    graph = parse_file("examples/01_clear_leak_scaler_before_split.ipynb")
    ev = check_fit_before_split(graph)[0]
    result = verify_leakage("examples/01_clear_leak_scaler_before_split.ipynb", ev, seeds=[0, 1, 2])
    assert result.status == FindingStatus.CONFIRMED and result.verification_ran
    assert result.evidence_id == ev.evidence_id
    assert result.primary_metric_name == "test_auc"
    assert result.mean_delta < -0.15 and result.p_value < 0.05
    assert len(result.seed_runs) == 3 and result.diff


def test_verify_needs_at_least_three_seeds():
    graph = parse_file("examples/01_clear_leak_scaler_before_split.ipynb")
    ev = check_fit_before_split(graph)[0]
    result = verify_leakage("examples/01_clear_leak_scaler_before_split.ipynb", ev, seeds=[0])
    assert result.status == FindingStatus.SUSPECTED and not result.verification_ran
    assert "at least 3" in result.reason


def test_verify_never_raises_when_the_code_crashes(tmp_path):
    p = tmp_path / "crashes.py"
    p.write_text(
        "import numpy as np\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import train_test_split\n"
        "X = np.random.randn(30, 3); y = np.arange(30) % 2\n"
        "X2 = StandardScaler().fit_transform(X)\n"
        "a, b, c, d = train_test_split(X2, y)\n"
        "raise RuntimeError('boom')\n",
        encoding="utf-8",
    )
    ev = check_fit_before_split(parse_file(str(p)))[0]
    result = verify_leakage(str(p), ev, seeds=[0, 1, 2])
    assert result.status == FindingStatus.SUSPECTED
    assert "failed to run" in result.reason and result.original_mean == 0.0
