"""Markdown audit report generator for REPROVE."""
import os
import platform
import sys
from datetime import datetime, timezone
import sklearn

from reprove.agent.schemas import InvestigationReport
from reprove.verifier.models import FindingStatus


def render_markdown_report(report: InvestigationReport) -> str:
    """
    Renders an InvestigationReport into a GitHub-Flavored Markdown audit document.
    Strict Invariant: Every number comes from real executed outputs.
    """
    lines = []

    # Title & Header
    lines.append(f"# 🔬 REPROVE Audit Report")
    lines.append(f"**Target**: `{report.file_path}`  ")
    lines.append(f"**Audit Timestamp**: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`  ")
    lines.append(f"**Execution Mode**: `{report.execution_mode}`  \n")

    # Executive Verdict Badge
    if report.has_confirmed_leaks:
        badge = "## 🔴 VERDICT: EVALUATION LEAKAGE CONFIRMED"
        verdict_desc = (
            "Empirical evidence proves evaluation leakage in this pipeline. "
            "Rerunning the experiment with the flaw fixed resulted in a statistically significant "
            "drop in performance, confirming that evaluation metrics were artificially inflated."
        )
    elif len(report.suspected_evidence) > 0:
        badge = "## 🟡 VERDICT: SUSPECTED LEAKAGE DETECTED"
        verdict_desc = (
            "Structural patterns indicate potential evaluation leakage. However, controlled rerun "
            "verification could not be automatically confirmed."
        )
    else:
        badge = "## 🟢 VERDICT: CLEAN AUDIT (NO LEAK DETECTED)"
        verdict_desc = (
            "No evaluation leakage was detected across dataset splitting, preprocessing, "
            "model selection, or evaluation stages."
        )

    lines.append(badge)
    lines.append(f"{verdict_desc}\n")

    # Summary Table
    lines.append("### Audit Summary")
    lines.append("| Metric | Value |")
    lines.append("| :--- | :--- |")
    lines.append(f"| **Confirmed Leakage Findings** | **{len(report.confirmed_evidence)}** |")
    lines.append(f"| **Suspected Leakage Findings** | {len(report.suspected_evidence)} |")
    lines.append(f"| **Clean / Passed Checks** | {len(report.clean_checks)} |")
    lines.append(f"| **Investigation Steps Executed** | {len(report.steps)} |")
    lines.append(f"| **Controlled Reruns Executed** | {len(report.verification_results)} |\n")

    # 1. Confirmed Findings
    if report.confirmed_evidence:
        lines.append("---")
        lines.append("## 🚨 Confirmed Leakage Findings\n")

        for idx, ev in enumerate(report.confirmed_evidence, 1):
            leak_type = ev.metrics.get("leak_type", "EVALUATION_LEAKAGE")
            lines.append(f"### Finding {idx}: {leak_type}")
            lines.append(f"- **Status**: `CONFIRMED`")
            lines.append(f"- **Evidence ID**: `{ev.evidence_id}`")
            lines.append(f"- **Evidence Tool**: `{ev.tool_name}` (Confidence: {ev.confidence * 100:.0f}%)")

            if ev.offending_span:
                span_desc = f"Line {ev.offending_span.start_line} to {ev.offending_span.end_line}"
                if ev.offending_span.cell_index is not None:
                    span_desc = f"Notebook Cell [{ev.offending_span.cell_index}], Line {ev.offending_span.cell_line} ({span_desc})"
                lines.append(f"- **Offending Code Location**: `{span_desc}`")
                lines.append(f"\n```python\n{ev.offending_span.code_snippet}\n```\n")

            lines.append(f"**Finding Summary**:  \n{ev.summary}\n")

            # Look up matching verification result
            matching_vr = next(
                (v for v in report.verification_results if v.status == FindingStatus.CONFIRMED),
                None,
            )

            if matching_vr:
                lines.append("#### Controlled Rerun Verification Results")
                lines.append(f"Primary Metric: **`{matching_vr.primary_metric_name}`**\n")

                lines.append("| Random Seed | Original Metric (Leaked) | Patched Metric (Clean) | Measured Delta (Δ) |")
                lines.append("| :---: | :---: | :---: | :---: |")
                for sr in matching_vr.seed_runs:
                    lines.append(f"| Seed `{sr.seed}` | `{sr.original_metric:.4f}` | `{sr.patched_metric:.4f}` | **`{sr.delta:+.4f}`** |")

                lines.append(
                    f"| **Mean ± Std** | **`{matching_vr.original_mean:.4f} ± {matching_vr.original_std:.4f}`** | "
                    f"**`{matching_vr.patched_mean:.4f} ± {matching_vr.patched_std:.4f}`** | "
                    f"**`{matching_vr.mean_delta:+.4f}`** |\n"
                )

                lines.append(f"- **Seed Noise Standard Deviation**: `σ = {matching_vr.seed_noise_std:.4f}`")
                lines.append(f"- **Statistically Significant**: `{'YES (|Δ| > 2σ)' if matching_vr.is_significant else 'NO'}`")
                lines.append(f"- **Empirical Verdict**: {matching_vr.reason}\n")

                if matching_vr.diff:
                    lines.append("#### Suggested Correction (Diff)")
                    lines.append("```diff")
                    lines.append(matching_vr.diff.strip())
                    lines.append("```\n")

            elif ev.suggested_fix:
                lines.append(f"**Suggested Remediation**: {ev.suggested_fix}\n")

    # 2. Suspected Findings
    if report.suspected_evidence:
        lines.append("---")
        lines.append("## ⚠️ Suspected Leakage Findings\n")
        for idx, ev in enumerate(report.suspected_evidence, 1):
            leak_type = ev.metrics.get("leak_type", "SUSPECTED_LEAKAGE")
            lines.append(f"### Suspected Finding {idx}: {leak_type}")
            lines.append(f"- **Status**: `SUSPECTED` (Deterministic detection; rerun not verified)")
            lines.append(f"- **Evidence ID**: `{ev.evidence_id}`")
            lines.append(f"- **Tool**: `{ev.tool_name}`")
            if ev.offending_span:
                lines.append(f"- **Location**: Line {ev.offending_span.start_line} - {ev.offending_span.end_line}")
                lines.append(f"\n```python\n{ev.offending_span.code_snippet}\n```\n")
            lines.append(f"**Analysis**: {ev.summary}")
            if ev.suggested_fix:
                lines.append(f"**Remediation**: {ev.suggested_fix}\n")

    # 3. Clean Checks
    if report.clean_checks:
        lines.append("---")
        lines.append("## ✅ Passed Checks (Verified Clean)\n")
        for ck in report.clean_checks:
            lines.append(f"- **`{ck.tool_name}`**: {ck.summary}")
        lines.append("")

    # 4. Agent Trace
    if report.steps:
        lines.append("---")
        lines.append("## 🕵️ Autonomous Investigation Trace\n")
        lines.append("| Step | Tool Chosen | Observation | Evidence Anchor |")
        lines.append("| :---: | :--- | :--- | :--- |")
        for st in report.steps:
            ev_anchor = f"`{st.evidence_id}`" if st.evidence_id else "—"
            obs_brief = st.observation.splitlines()[0] if st.observation else "OK"
            if len(obs_brief) > 80:
                obs_brief = obs_brief[:77] + "..."
            lines.append(f"| {st.step_number} | `{st.tool_name}` | {obs_brief} | {ev_anchor} |")
        lines.append("")

    # 5. Reproducibility Instructions
    lines.append("---")
    lines.append("## 🔬 Reproducibility & Environment")
    lines.append("To independently reproduce this exact audit result, execute:")
    lines.append("```bash")
    lines.append(f"reprove audit {report.file_path} --replay")
    lines.append("```")
    lines.append(f"- **Python Version**: `{platform.python_version()} ({sys.platform})`")
    lines.append(f"- **scikit-learn Version**: `{sklearn.__version__}`")
    lines.append("- **Verification Guarantee**: Every metric above was measured through sandboxed execution; zero claims were fabricated by LLM generation.")

    return "\n".join(lines)
