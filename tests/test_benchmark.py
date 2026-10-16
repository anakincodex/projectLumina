"""Tests for benchmark evaluation matrix and evaluation CLI."""
import pytest
from typer.testing import CliRunner

from reprove.cli.main import app
from reprove.evaluation import evaluate_benchmarks

runner = CliRunner()


def test_evaluate_benchmarks_metrics():
    """Verify benchmark matrix yields 100% precision and recall on ground truth."""
    summary = evaluate_benchmarks(use_replay=True)
    assert summary.total_cases == 3
    assert summary.true_positives == 2
    assert summary.true_negatives == 1
    assert summary.false_positives == 0
    assert summary.false_negatives == 0
    assert summary.precision == 1.0
    assert summary.recall == 1.0
    assert summary.f1_score == 1.0
    assert summary.specificity == 1.0
    assert summary.total_runtime_seconds < 5.0


def test_cli_benchmark_command():
    """Verify reprove benchmark CLI execution."""
    result = runner.invoke(app, ["benchmark", "--replay"])
    assert result.exit_code == 0
    assert "REPROVE Benchmark Evaluation Matrix" in result.stdout
    assert "Precision" in result.stdout
    assert "100.0%" in result.stdout
    assert "BM-01" in result.stdout
    assert "BM-02" in result.stdout
    assert "BM-03" in result.stdout
