"""Unit and integration tests for REPROVE CLI and UI."""
import os
import tempfile
import pytest
from typer.testing import CliRunner

from reprove.cli.main import app

runner = CliRunner()


def test_cli_help():
    """Verify reprove --help displays commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "audit" in result.stdout
    assert "ui" in result.stdout


def test_cli_audit_nonexistent_file():
    """Verify error on nonexistent target file."""
    result = runner.invoke(app, ["audit", "nonexistent_file.ipynb"])
    assert result.exit_code == 2
    assert "File not found" in result.stdout


def test_cli_audit_clean_notebook_replay():
    """Verify clean benchmark audit with replay cache exits 0."""
    result = runner.invoke(
        app,
        ["audit", "examples/02_clean_pipeline_no_leak.ipynb", "--replay"],
    )
    assert result.exit_code == 0
    assert "CLEAN EXPERIMENT PIPELINE" in result.stdout
    assert "CLEAN" in result.stdout


def test_cli_audit_leaking_notebook_replay_fail_on_leak():
    """Verify leaking benchmark exits 1 by default when fail_on_leak=True."""
    result = runner.invoke(
        app,
        ["audit", "examples/01_clear_leak_scaler_before_split.ipynb", "--replay"],
    )
    assert result.exit_code == 1
    assert "CONFIRMED EVALUATION LEAK" in result.stdout
    assert "CONFIRMED" in result.stdout


def test_cli_audit_leaking_notebook_no_fail_on_leak_with_output_dir():
    """Verify audit outputs report.md and report.html when --output-dir is given."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(
            app,
            [
                "audit",
                "examples/01_clear_leak_scaler_before_split.ipynb",
                "--replay",
                "--no-fail-on-leak",
                "--output-dir",
                tmpdir,
            ],
        )
        assert result.exit_code == 0
        assert os.path.exists(os.path.join(tmpdir, "report.md"))
        assert os.path.exists(os.path.join(tmpdir, "report.html"))


def test_cli_audit_duplicate_split_notebook():
    """Verify subtle duplicate split leak audit via CLI produces SUSPECTED verdict."""
    result = runner.invoke(
        app,
        [
            "audit",
            "examples/03_subtle_leak_duplicate_split.ipynb",
            "--replay",
            "--no-fail-on-leak",
        ],
    )
    assert result.exit_code == 0
    assert "SUSPECTED LEAK" in result.stdout
    assert "SUSPECTED" in result.stdout


def test_streamlit_app_importable():
    """Verify Streamlit app module can be imported without error."""
    import reprove.cli.streamlit_app as st_app
    assert hasattr(st_app, "BENCHMARKS")
    assert "01 - Type 1: Clear Preprocessing Leak (Hastie-Tibshirani)" in st_app.BENCHMARKS
