"""Tests for the Markdown / HTML report generator."""
import os

from reprove.agent import run_investigation
from reprove.report import generate_report, render_html_report, render_markdown_report

EX1 = "examples/01_clear_leak_scaler_before_split.ipynb"


def test_markdown_report_confirmed_leak(recorded_benchmarks):
    md = render_markdown_report(run_investigation(EX1, use_replay=True))
    assert "# 🔬 REPROVE Audit Report" in md
    assert "🔴 VERDICT: EVALUATION LEAKAGE CONFIRMED" in md
    assert "## 🚨 Confirmed Leakage Findings" in md
    assert "Controlled rerun" in md and "paired t-test" in md
    assert "```diff" in md
    assert "replay of a recorded run" in md
    assert "reprove audit examples/01_clear_leak_scaler_before_split.ipynb --model" in md
    assert "scikit-learn" in md


def test_markdown_report_clean(recorded_benchmarks):
    md = render_markdown_report(run_investigation("examples/02_clean_pipeline_no_leak.ipynb", use_replay=True))
    assert "🟢 VERDICT: CLEAN" in md
    assert "## 🚨 Confirmed Leakage Findings" not in md
    assert "Passed Checks" in md


def test_markdown_report_suspected_explains_why(recorded_benchmarks):
    md = render_markdown_report(run_investigation("examples/04_test_set_model_selection.ipynb", use_replay=True))
    assert "SUSPECTED" in md and "not implemented" in md


def test_html_report(recorded_benchmarks):
    html_content = render_html_report(run_investigation(EX1, use_replay=True))
    assert "<!DOCTYPE html>" in html_content
    assert "EVALUATION LEAKAGE CONFIRMED" in html_content
    assert "<table" in html_content and '<pre class="diff">' in html_content


def test_generate_report_writes_to_disk(tmp_path, recorded_benchmarks):
    out = str(tmp_path / "audit_output")
    generate_report(run_investigation(EX1, use_replay=True), output_dir=out)
    assert os.path.getsize(os.path.join(out, "report.md")) > 100
    assert os.path.getsize(os.path.join(out, "report.html")) > 100
