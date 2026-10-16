"""Tests for the CLI and the Streamlit module."""
import os
import tempfile

import pytest
from typer.testing import CliRunner

from reprove.cli.main import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "audit" in result.stdout and "benchmark" in result.stdout


def test_cli_audit_nonexistent_file():
    result = runner.invoke(app, ["audit", "nonexistent_file.ipynb"])
    assert result.exit_code == 2 and "File not found" in result.stdout


def test_cli_rejects_bad_seeds():
    result = runner.invoke(app, ["audit", "examples/02_clean_pipeline_no_leak.ipynb", "--seeds", "a,b"])
    assert result.exit_code != 0


def test_cli_clean_notebook_replay(recorded_benchmarks):
    result = runner.invoke(app, ["audit", "examples/02_clean_pipeline_no_leak.ipynb", "--replay"])
    assert result.exit_code == 0
    assert "CLEAN EXPERIMENT PIPELINE" in result.stdout
    assert "REPLAY_CACHE" in result.stdout


def test_cli_leaking_notebook_fails_on_leak(recorded_benchmarks):
    result = runner.invoke(app, ["audit", "examples/01_clear_leak_scaler_before_split.ipynb", "--replay"])
    assert result.exit_code == 1
    assert "CONFIRMED EVALUATION LEAK" in result.stdout


def test_cli_writes_reports(recorded_benchmarks):
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(app, [
            "audit", "examples/01_clear_leak_scaler_before_split.ipynb",
            "--replay", "--no-fail-on-leak", "--output-dir", tmpdir,
        ])
        assert result.exit_code == 0
        assert os.path.exists(os.path.join(tmpdir, "report.md"))
        assert os.path.exists(os.path.join(tmpdir, "report.html"))


def test_cli_duplicate_split_is_confirmed(recorded_benchmarks):
    result = runner.invoke(app, ["audit", "examples/03_subtle_leak_duplicate_split.ipynb", "--replay", "--no-fail-on-leak"])
    assert result.exit_code == 0
    assert "CONFIRMED" in result.stdout


def test_cli_test_set_selection_is_suspected(recorded_benchmarks):
    result = runner.invoke(app, ["audit", "examples/04_test_set_model_selection.ipynb", "--replay"])
    assert result.exit_code == 0
    assert "SUSPECTED" in result.stdout and "Not verified" in result.stdout


def test_streamlit_app_importable():
    pytest.importorskip("streamlit")
    import reprove.cli.streamlit_app as st_app
    assert "01 - Type 1: Clear Preprocessing Leak (Hastie-Tibshirani)" in st_app.BENCHMARKS
