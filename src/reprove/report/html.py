"""Self-contained (offline) HTML audit report renderer for REPROVE."""
import html
import os
import platform
from typing import List

import sklearn

from reprove import __version__
from reprove.agent.schemas import InvestigationReport
from reprove.report.markdown import VERDICT_TEXT, location, mode_description
from reprove.tools.models import EvidenceObject
from reprove.verifier.models import NOISE_RULE, format_p

BADGE_COLORS = {
    "CONFIRMED": "#da3633",
    "SUSPECTED": "#d29922",
    "NOT_CONFIRMED": "#1f6feb",
    "INCONCLUSIVE": "#6e7681",
    "CLEAN": "#238636",
}
TAG_CLASS = {"CONFIRMED": "tag-red", "SUSPECTED": "tag-yellow", "NOT_CONFIRMED": "tag-blue"}

e = html.escape


def _finding_card(report: InvestigationReport, ev: EvidenceObject, idx: int, label: str) -> str:
    vr = report.verification_for(ev.evidence_id)
    code = f"<pre><code>{e(ev.offending_span.code_snippet)}</code></pre>" if ev.offending_span else ""
    rerun = ""
    if vr and vr.verification_ran:
        rows = "".join(
            f"<tr><td>{sr.seed}</td><td>{sr.original_metric:.4f}</td><td>{sr.patched_metric:.4f}</td>"
            f"<td><strong>{sr.delta:+.4f}</strong></td></tr>"
            for sr in vr.seed_runs
        )
        p = format_p(vr.p_value)
        rerun = f"""
        <h4>Controlled rerun - metric <code>{e(vr.primary_metric_name)}</code></h4>
        <table>
          <thead><tr><th>Seed</th><th>Original (as written)</th><th>Corrected</th><th>Difference</th></tr></thead>
          <tbody>{rows}
            <tr class="total"><td>Mean ± std</td><td>{vr.original_mean:.4f} ± {vr.original_std:.4f}</td>
            <td>{vr.patched_mean:.4f} ± {vr.patched_std:.4f}</td><td>{vr.mean_delta:+.4f}</td></tr>
          </tbody>
        </table>
        <p><strong>Std of paired differences:</strong> {vr.seed_noise_std:.4f} |
           <strong>One-sided paired t-test p:</strong> {p}</p>
        <p class="muted">{e(NOISE_RULE)}</p>"""
    reason = f"<p><em>{e(vr.reason)}</em></p>" if vr else ""
    diff = (
        f'<details><summary>Suggested correction (diff)</summary><pre class="diff"><code>{e(vr.diff)}</code></pre></details>'
        if vr and vr.diff else ""
    )
    fix = f"<p><strong>How to fix:</strong> {e(ev.suggested_fix)}</p>" if ev.suggested_fix else ""
    return f"""
    <div class="card">
      <h3>Finding {idx}: {e(str(ev.metrics.get('leak_type', 'LEAKAGE')))}
        <span class="tag {TAG_CLASS.get(label, '')}">{label}</span></h3>
      <p><strong>Evidence:</strong> <code>{e(ev.evidence_id)}</code> from <code>{e(ev.tool_name)}</code> |
         <strong>Location:</strong> {e(location(ev))}</p>
      <p>{e(ev.summary)}</p>
      {code}{rerun}{reason}{diff}{fix}
    </div>"""


def render_html_report(report: InvestigationReport) -> str:
    verdict = report.verdict
    title, desc = VERDICT_TEXT[verdict]
    title = title.lstrip("# ").strip()

    cards: List[str] = []
    idx = 0
    for label, items in (
        ("CONFIRMED", report.confirmed_evidence),
        ("SUSPECTED", report.suspected_evidence),
        ("NOT_CONFIRMED", report.not_confirmed_evidence),
    ):
        for ev in items:
            idx += 1
            cards.append(_finding_card(report, ev, idx, label))

    checks = "".join(
        f"<li><code>{e(c.tool_name)}</code>: {e(c.summary)}</li>" for c in report.clean_checks
    ) + "".join(
        f"<li><code>{e(c.tool_name)}</code> (inconclusive): {e(c.summary)}</li>" for c in report.inconclusive_checks
    )
    steps = "".join(
        f"<tr><td>{s.step_number}</td><td>{e(s.origin)}</td><td><code>{e(s.tool_name)}</code></td>"
        f"<td>{e(s.result or '-')}</td><td>{e(s.thought[:160])}</td></tr>"
        for s in report.steps
    )
    fallback = f"<p><strong>Fallback reason:</strong> {e(report.fallback_reason)}</p>" if report.fallback_reason else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>REPROVE Audit Report: {e(os.path.basename(report.file_path))}</title>
<style>
  :root {{ --bg:#0d1117; --card:#161b22; --border:#30363d; --text:#c9d1d9; --muted:#8b949e; --accent:#58a6ff; }}
  body {{ background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
         line-height:1.6; margin:0; padding:30px 16px; }}
  .container {{ max-width:920px; margin:0 auto; }}
  h1,h2,h3,h4 {{ color:#fff; }}
  .badge {{ display:inline-block; padding:8px 16px; border-radius:6px; font-weight:bold; background:{BADGE_COLORS[verdict]}; color:#fff; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:18px; margin-bottom:18px; overflow-x:auto; }}
  table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
  th,td {{ text-align:left; padding:8px; border-bottom:1px solid var(--border); vertical-align:top; }}
  th {{ background:#21262d; color:#f0f6fc; }}
  tr.total td {{ font-weight:bold; background:#21262d; }}
  code {{ background:#21262d; padding:2px 6px; border-radius:4px; font-family:Consolas,monospace; font-size:.9em; }}
  pre {{ background:#090d13; border:1px solid var(--border); border-radius:6px; padding:12px; overflow-x:auto; }}
  pre code {{ background:none; padding:0; }}
  pre.diff {{ color:#7ee787; }}
  summary {{ cursor:pointer; font-weight:600; color:var(--accent); }}
  .tag {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:.75em; color:#fff; vertical-align:middle; }}
  .tag-red {{ background:#da3633; }} .tag-yellow {{ background:#9e6a03; }} .tag-blue {{ background:#1f6feb; }}
  .muted {{ color:var(--muted); font-size:.9em; }}
  .footer {{ margin-top:32px; padding-top:16px; border-top:1px solid var(--border); color:var(--muted); font-size:.85em; }}
</style>
</head>
<body>
<div class="container">
  <h1>🔬 REPROVE Audit Report</h1>
  <p><strong>Target:</strong> <code>{e(report.file_path)}</code></p>
  <p><strong>Execution mode:</strong> <code>{e(report.execution_mode)}</code> - {e(mode_description(report))}</p>
  {fallback}
  <p><strong>Seeds:</strong> {e(', '.join(str(s) for s in report.seeds))}</p>
  <div class="badge">{e(title)}</div>
  <p>{e(desc)}</p>

  <div class="card">
    <h3>Audit summary</h3>
    <table>
      <tr><td>Confirmed findings</td><td><strong>{len(report.confirmed_evidence)}</strong></td></tr>
      <tr><td>Suspected (not verified)</td><td>{len(report.suspected_evidence)}</td></tr>
      <tr><td>Pattern present, no measurable inflation</td><td>{len(report.not_confirmed_evidence)}</td></tr>
      <tr><td>Clean / inconclusive checks</td><td>{len(report.clean_checks)} / {len(report.inconclusive_checks)}</td></tr>
      <tr><td>Investigation steps</td><td>{len(report.steps)}</td></tr>
    </table>
  </div>

  {''.join(cards)}

  <div class="card"><h3>Checks</h3><ul>{checks or '<li>None</li>'}</ul></div>

  <div class="card">
    <h3>Investigation trace</h3>
    <p class="muted">agent = chosen by the LLM, coverage = run automatically because the agent skipped it, fallback = deterministic plan.</p>
    <table><thead><tr><th>#</th><th>Origin</th><th>Tool</th><th>Result</th><th>Reasoning</th></tr></thead>
    <tbody>{steps}</tbody></table>
  </div>

  <div class="footer">
    <p>REPROVE {e(__version__)} | Python {platform.python_version()} | scikit-learn {sklearn.__version__}</p>
    <p>Every metric was measured by executing the code in a separate process; no number was generated by the LLM.</p>
  </div>
</div>
</body>
</html>
"""
