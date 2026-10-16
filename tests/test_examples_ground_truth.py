"""Tests to verify that benchmark planted-leak notebooks run and behave as expected."""
import nbformat
import numpy as np
import pytest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def extract_code_from_notebook(notebook_path: str) -> str:
    """Read a notebook and combine code cells into a single python script string."""
    nb = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell.source for cell in nb.cells if cell.cell_type == "code"]
    return "\n\n".join(code_cells)


def test_01_clear_leak_execution_and_empirical_drop():
    """Verify that notebook 01 runs and demonstrates massive empirical drop when patched."""
    code = extract_code_from_notebook("examples/01_clear_leak_scaler_before_split.ipynb")
    env = {}
    exec(code, env)

    leaked_acc = env["test_acc"]
    leaked_auc = env["test_auc"]
    assert leaked_auc > 0.70, f"Expected high leaked AUC, got {leaked_auc}"

    # Now verify the CLEAN / PATCHED equivalent (split first, fit only on train)
    np.random.seed(42)
    X = np.random.randn(200, 1000)
    y = np.random.randint(0, 2, size=200)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    selector = SelectKBest(score_func=f_classif, k=10)
    X_train_sel = selector.fit_transform(X_train_s, y_train)
    X_test_sel = selector.transform(X_test_s)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_sel, y_train)

    clean_acc = accuracy_score(y_test, clf.predict(X_test_sel))
    clean_auc = roc_auc_score(y_test, clf.predict_proba(X_test_sel)[:, 1])

    delta_auc = clean_auc - leaked_auc
    # The drop must be statistically decisive (at least -0.15)
    assert delta_auc < -0.15, (
        f"Empirical leak not statistically significant: delta={delta_auc:.4f} "
        f"(leaked={leaked_auc:.4f}, clean={clean_auc:.4f})"
    )


def test_02_clean_pipeline_execution():
    """Verify that notebook 02 runs without error and produces valid metrics."""
    code = extract_code_from_notebook("examples/02_clean_pipeline_no_leak.ipynb")
    env = {}
    exec(code, env)

    assert "test_acc" in env
    assert "test_auc" in env
    assert 0.0 <= env["test_acc"] <= 1.0
    assert 0.0 <= env["test_auc"] <= 1.0


def test_03_subtle_leak_duplicate_split():
    """Verify that notebook 03 runs and has actual duplicate rows across train and test."""
    code = extract_code_from_notebook("examples/03_subtle_leak_duplicate_split.ipynb")
    env = {}
    exec(code, env)

    X_train = env["X_train"]
    X_test = env["X_test"]

    # Convert rows to hashable tuples
    train_rows = {tuple(np.round(row, 6)) for row in X_train}
    test_rows = {tuple(np.round(row, 6)) for row in X_test}

    overlapping_rows = train_rows.intersection(test_rows)
    assert len(overlapping_rows) > 0, "Expected duplicate rows between train and test splits!"
    overlap_pct = (len(overlapping_rows) / len(test_rows)) * 100
    assert overlap_pct >= 10.0, f"Expected at least 10% overlap, got {overlap_pct:.2f}%"
