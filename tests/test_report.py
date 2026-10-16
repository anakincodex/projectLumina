"""Unit and integration tests for the Report Generator module."""
import os
import pytest
from reprove.agent import run_investigation
from reprove.report import generate_report, render_html_report, render_markdown_report


def test_markdown_report_example_1_confirmed_leak():
    """Verify markdown report contains all required sections for confirmed leak."""
    report = run_investigation(
        "examples/01_clear_leak_scaler_before_split.ipynb",
        use_replay=True,
    )
    md = render_markdown_report(report)

    assert "# 🔬 REPROVE Audit Report" in md
    assert "🔴 VERDICT: EVALUATION LEAKAGE CONFIRMED" in md
    assert "Confirmed Leakage Findings" in md
    assert "Controlled Rerun Verification Results" in md
    assert "Seed Noise Standard Deviation" in md
    assert "Suggested Correction (Diff)" in md
    assert "```diff" in md
    assert "reprove audit" in md
    assert "scikit-learn Version" in md


def test_markdown_report_example_2_clean():
    """Verify markdown report indicates clean bill of health on negative control."""
    report = run_investigation(
        "examples/02_clean_pipeline_no_leak.ipynb",
        use_replay=True,
    )
    md = render_markdown_report(report)

    assert "🟢 VERDICT: CLEAN AUDIT (NO LEAK DETECTED)" in md
    assert "## 🚨 Confirmed Leakage Findings" not in md
    assert "Passed Checks (Verified Clean)" in md


def test_html_report_rendering():
    """Verify HTML report renders valid semantic markup and styling."""
    report = run_investigation(
        "examples/01_clear_leak_scaler_before_split.ipynb",
        use_replay=True,
    )
    html_content = render_html_report(report)

    assert "<!DOCTYPE html>" in html_content
    assert "REPROVE Audit Report" in html_content
    assert "EVALUATION LEAKAGE CONFIRMED" in html_content
    assert "<table" in html_content
    assert "<pre class=\"diff\">" in html_content


def test_generate_report_writes_to_disk(tmp_path):
    """Verify generate_report writes report.md and report.html to disk."""
    report = run_investigation(
        "examples/01_clear_leak_scaler_before_split.ipynb",
        use_replay=True,
    )
    output_dir = str(tmp_path / "audit_output")
    md_text, html_text = generate_report(report, output_dir=output_dir)

    md_file = os.path.join(output_dir, "report.md")
    html_file = os.path.join(output_dir, "report.html")

    assert os.path.exists(md_file)
    assert os.path.exists(html_file)
    assert os.path.getsize(md_file) > 100
    assert os.path.getsize(html_file) > 100
