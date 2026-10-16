"""Tests for the benchmark runner (uses the runs recorded once per test session)."""
from typer.testing import CliRunner

from reprove.cli.main import app
from reprove.evaluation import BENCHMARK_CASES, evaluate_benchmarks

runner = CliRunner()


def test_benchmark_matches_ground_truth(recorded_benchmarks):
    summary = evaluate_benchmarks(use_replay=True)
    assert summary.total_cases == len(BENCHMARK_CASES)
    mismatches = [(r.case_id, r.expected_status, r.verdict) for r in summary.results if not r.status_match]
    assert not mismatches, mismatches
    assert summary.false_positives == 0 and summary.false_negatives == 0


def test_cli_benchmark_command(recorded_benchmarks):
    result = runner.invoke(app, ["benchmark", "--replay"])
    assert result.exit_code == 0
    assert "Benchmark Results" in result.stdout and "Precision" in result.stdout
    for case in BENCHMARK_CASES:
        assert case["id"] in result.stdout
