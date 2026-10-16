"""Markdown audit report generator for REPROVE."""
import platform
import sys
from datetime import datetime, timezone
from typing import List

import numpy
import sklearn

from reprove import __version__
from reprove.agent.schemas import InvestigationReport
from reprove.tools.models import EvidenceObject
from reprove.verifier.models import NOISE_RULE, format_p

VERDICT_TEXT = {
    "CONFIRMED": (
        "## 🔴 VERDICT: EVALUATION LEAKAGE CONFIRMED",
        "A deterministic tool found the leakage pattern AND a controlled rerun with the flaw fixed scored "
        "significantly worse across seeds, so the reported metric is inflated.",
    ),
    "SUSPECTED": (
        "## 🟡 VERDICT: SUSPECTED LEAKAGE (NOT VERIFIED)",
        "A deterministic tool found a leakage pattern, but the controlled rerun could not be run for it. "
        "The reason is given for each finding.",
    ),
    "NOT_CONFIRMED": (
        "## 🔵 VERDICT: LEAKAGE PATTERN PRESENT, NO MEASURABLE INFLATION",
        "A leakage pattern exists in the code, but fixing it did not change the score beyond seed noise. "
        "It is still worth fixing.",
    ),
    "INCONCLUSIVE": (
        "## ⚪ VERDICT: INCONCLUSIVE",
        "Some checks could not run on this code (see below), and no leakage was found by the checks that did run.",
    ),
    "CLEAN": (
        "## 🟢 VERDICT: CLEAN (NO LEAK DETECTED)",
        "No leakage was detected by the fit-before-split, split-overlap and test-usage checks.",
    ),
}


def location(ev: EvidenceObject) -> str:
    span = ev.offending_span or ev.context_span
    if not span:
        return "n/a"
    lines = f"script lines {span.start_line}-{span.end_line}"
    if span.cell_index is not None:
        return f"notebook cell [{span.cell_index}], line {span.cell_line} ({lines})"
    return lines


def mode_description(report: InvestigationReport) -> str:
    mode = report.effective_mode
    model = f" (model `{report.model_name}`)" if report.model_name else ""
    if mode == "LLM_OLLAMA":
        text = f"open-weight LLM agent via Ollama{model}"
    elif mode == "FALLBACK_DETERMINISTIC":
        text = "deterministic fallback plan (no LLM decisions)"
    else:
        text = mode
    if report.execution_mode == "REPLAY_CACHE":
        when = f", recorded {report.recorded_at}" if report.recorded_at else ""
        text = f"replay of a recorded run - {text}{when}"
    return text


def _finding_section(report: InvestigationReport, ev: EvidenceObject, idx: int, label: str) -> List[str]:
    out = []
    vr = report.verification_for(ev.evidence_id)
    out.append(f"### Finding {idx}: {ev.metrics.get('leak_type', 'LEAKAGE')} - `{label}`")
    out.append(f"- **Evidence ID**: `{ev.evidence_id}` (tool `{ev.tool_name}`)")
    out.append(f"- **Code location**: {location(ev)}")
    if ev.offending_span:
        out.append(f"\n```python\n{ev.offending_span.code_snippet}\n```\n")
    out.append(f"**Tool evidence**: {ev.summary}\n")

    if vr and vr.verification_ran:
        arrow = "higher is better" if vr.higher_is_better else "lower is better"
        out.append(f"#### Controlled rerun (`{vr.primary_metric_name}`, {arrow})")
        out.append("| Seed | Original (as written) | Corrected | Difference |")
        out.append("| :---: | :---: | :---: | :---: |")
        for sr in vr.seed_runs:
            out.append(f"| `{sr.seed}` | `{sr.original_metric:.4f}` | `{sr.patched_metric:.4f}` | `{sr.delta:+.4f}` |")
        out.append(
            f"| **Mean ± std** | **`{vr.original_mean:.4f} ± {vr.original_std:.4f}`** | "
            f"**`{vr.patched_mean:.4f} ± {vr.patched_std:.4f}`** | **`{vr.mean_delta:+.4f}`** |\n"
        )
        p = format_p(vr.p_value)
        out.append(f"- **Std of paired differences**: `{vr.seed_noise_std:.4f}` (standard error `{vr.standard_error:.4f}`)")
        out.append(f"- **One-sided paired t-test p-value**: `{p}`")
        out.append(f"- **Decision rule**: {NOISE_RULE}")
    if vr:
        out.append(f"- **Verification verdict**: {vr.reason}\n")
    if vr and vr.diff:
        out.append("#### Suggested correction (diff generated from your code)")
        out.append("```diff")
        out.append(vr.diff.strip())
        out.append("```\n")
    if ev.suggested_fix:
        out.append(f"**How to fix**: {ev.suggested_fix}\n")
    return out


def render_markdown_report(report: InvestigationReport) -> str:
    """Render an InvestigationReport as GitHub-flavoured Markdown. Every number comes from executed runs."""
    lines: List[str] = []
    lines.append("# 🔬 REPROVE Audit Report")
    lines.append(f"**Target**: `{report.file_path}`  ")
    lines.append(f"**Report generated**: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`  ")
    lines.append(f"**Execution mode**: `{report.execution_mode}` - {mode_description(report)}  ")
    if report.fallback_reason:
        lines.append(f"**Why the fallback plan was used**: {report.fallback_reason}  ")
    if report.seeds:
        lines.append(f"**Seeds**: `{', '.join(str(s) for s in report.seeds)}`  ")
    lines.append("")

    badge, desc = VERDICT_TEXT[report.verdict]
    lines.append(badge)
    lines.append(f"{desc}\n")

    reruns = sum(1 for v in report.verification_results if v.verification_ran)
    lines.append("### Audit Summary")
    lines.append("| Item | Count |")
    lines.append("| :--- | :--- |")
    lines.append(f"| **Confirmed Leakage Findings** | **{len(report.confirmed_evidence)}** |")
    lines.append(f"| Suspected (not verified) | {len(report.suspected_evidence)} |")
    lines.append(f"| Pattern present, no measurable inflation | {len(report.not_confirmed_evidence)} |")
    lines.append(f"| Clean checks | {len(report.clean_checks)} |")
    lines.append(f"| Inconclusive checks | {len(report.inconclusive_checks)} |")
    lines.append(f"| Investigation steps | {len(report.steps)} |")
    lines.append(f"| Controlled reruns executed | {reruns} |\n")

    idx = 0
    if report.confirmed_evidence:
        lines.append("---")
        lines.append("## 🚨 Confirmed Leakage Findings\n")
        for ev in report.confirmed_evidence:
            idx += 1
            lines.extend(_finding_section(report, ev, idx, "CONFIRMED"))
    if report.suspected_evidence:
        lines.append("---")
        lines.append("## ⚠️ Suspected Leakage Findings (not verified)\n")
        for ev in report.suspected_evidence:
            idx += 1
            lines.extend(_finding_section(report, ev, idx, "SUSPECTED"))
    if report.not_confirmed_evidence:
        lines.append("---")
        lines.append("## 🔵 Leakage Patterns Without Measurable Inflation\n")
        for ev in report.not_confirmed_evidence:
            idx += 1
            lines.extend(_finding_section(report, ev, idx, "NOT_CONFIRMED"))

    if report.clean_checks:
        lines.append("---")
        lines.append("## ✅ Passed Checks\n")
        for ck in report.clean_checks:
            lines.append(f"- **`{ck.tool_name}`**: {ck.summary}")
        lines.append("")
    if report.inconclusive_checks:
        lines.append("## ⚪ Inconclusive Checks\n")
        for ck in report.inconclusive_checks:
            lines.append(f"- **`{ck.tool_name}`**: {ck.summary}")
        lines.append("")

    if report.steps:
        lines.append("---")
        lines.append("## 🕵️ Investigation Trace\n")
        lines.append("`agent` = chosen by the LLM, `coverage` = run automatically because the agent skipped it, "
                     "`fallback` = deterministic plan.\n")
        lines.append("| Step | Origin | Tool | Result | Reasoning | Evidence |")
        lines.append("| :---: | :--- | :--- | :--- | :--- | :--- |")
        for st in report.steps:
            thought = st.thought.replace("|", "/").replace("\n", " ")
            if len(thought) > 110:
                thought = thought[:107] + "..."
            anchor = f"`{st.evidence_id}`" if st.evidence_id else "-"
            lines.append(f"| {st.step_number} | {st.origin} | `{st.tool_name}` | {st.result or '-'} | {thought} | {anchor} |")
        lines.append("")

    seeds = ",".join(str(s) for s in report.seeds) if report.seeds else "0,1,2,3,4"
    model = report.model_name or "qwen2.5-coder:7b"
    lines.append("---")
    lines.append("## 🔁 Reproducibility")
    lines.append("Re-run the full audit (tools and controlled reruns execute again):")
    lines.append("```bash")
    lines.append(f"reprove audit {report.file_path} --model {model} --seeds {seeds}")
    lines.append("```")
    lines.append("`--replay` only re-displays a previously recorded run; it does not re-execute anything.\n")
    lines.append(f"- **REPROVE**: `{__version__}`")
    lines.append(f"- **Python**: `{platform.python_version()} ({sys.platform})`")
    lines.append(f"- **scikit-learn**: `{sklearn.__version__}` | **numpy**: `{numpy.__version__}`")
    lines.append(
        "- Every metric above was measured by running the code in a separate Python process with a timeout "
        "(process isolation, not a security sandbox). No number in this report was generated by the LLM."
    )
    return "\n".join(lines)
