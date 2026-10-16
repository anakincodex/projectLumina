"""The audit must never crash, whatever code it is given; problems become honest statuses."""
from reprove.agent import deterministic_fallback_audit

SEEDS = [0, 1, 2]


def _audit(tmp_path, code, name="case.py"):
    p = tmp_path / name
    p.write_text(code, encoding="utf-8")
    return deterministic_fallback_audit(str(p), seeds=SEEDS)


def test_script_that_crashes_is_suspected_not_a_crash(tmp_path):
    report = _audit(tmp_path, (
        "import numpy as np\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import train_test_split\n"
        "X = np.random.randn(30, 3); y = np.arange(30) % 2\n"
        "Xs = StandardScaler().fit_transform(X)\n"
        "a, b, c, d = train_test_split(Xs, y)\n"
        "undefined_function()\n"
    ))
    assert report.verdict == "SUSPECTED"
    assert any("failed to run" in v.reason for v in report.verification_results)


def test_script_without_a_split_is_inconclusive(tmp_path):
    report = _audit(tmp_path, "import numpy as np\nprint(np.arange(3))\n")
    assert report.verdict == "INCONCLUSIVE"


def test_script_without_a_metric_is_not_given_fake_numbers(tmp_path):
    report = _audit(tmp_path, (
        "import numpy as np\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "X = np.random.randn(40, 3); y = np.arange(40) % 2\n"
        "Xs = StandardScaler().fit_transform(X)\n"
        "X_train, X_test, y_train, y_test = train_test_split(Xs, y)\n"
        "print(LogisticRegression().fit(X_train, y_train).predict(X_test)[:3])\n"   # no metric at all
    ))
    vr = report.verification_results[0]
    assert report.verdict == "SUSPECTED"
    assert not vr.verification_ran and vr.original_mean == 0.0 and "no evaluation metric" in vr.reason


def test_unseen_variable_names_minmax_leak():
    report = deterministic_fallback_audit("examples/stress/05_minmax_scaler_leak.py", seeds=SEEDS)
    assert report.verdict in {"NOT_CONFIRMED", "CONFIRMED"}   # verified either way, never a crash
    assert all(v.verification_ran for v in report.verification_results)
