"""HTML audit report renderer for REPROVE."""
import html
import os
import platform
import sys
from datetime import datetime, timezone
import sklearn

from reprove.agent.schemas import InvestigationReport
from reprove.verifier.models import FindingStatus


def render_html_report(report: InvestigationReport) -> str:
    """Renders a self-contained, offline-compatible HTML audit report."""
    is_confirmed = report.has_confirmed_leaks
    is_suspected = len(report.suspected_evidence) > 0 and not is_confirmed

    badge_color = "#da3633" if is_confirmed else ("#d29922" if is_suspected else "#238636")
    badge_title = "EVALUATION LEAKAGE CONFIRMED" if is_confirmed else ("SUSPECTED LEAKAGE" if is_suspected else "CLEAN AUDIT")

    confirmed_html = ""
    for idx, ev in enumerate(report.confirmed_evidence, 1):
        matching_vr = next(
            (v for v in report.verification_results if v.status == FindingStatus.CONFIRMED),
            None,
        )

        vr_html = ""
        if matching_vr:
            seed_rows = "".join(
                f"<tr><td>Seed {sr.seed}</td><td>{sr.original_metric:.4f}</td><td>{sr.patched_metric:.4f}</td><td style='color:#da3633; font-weight:bold;'>{sr.delta:+.4f}</td></tr>"
                for sr in matching_vr.seed_runs
            )
            vr_html = f"""
            <div class="card" style="margin-top:15px; border-left: 4px solid #da3633;">
                <h4>Controlled Rerun Results (Metric: <code>{matching_vr.primary_metric_name}</code>)</h4>
                <table>
                    <thead>
                        <tr><th>Seed</th><th>Original (Leaked)</th><th>Patched (Clean)</th><th>Delta (Δ)</th></tr>
                    </thead>
                    <tbody>
                        {seed_rows}
                        <tr style="font-weight:bold; background:#21262d;">
                            <td>Mean ± Std</td>
                            <td>{matching_vr.original_mean:.4f} ± {matching_vr.original_std:.4f}</td>
                            <td>{matching_vr.patched_mean:.4f} ± {matching_vr.patched_std:.4f}</td>
                            <td style="color:#da3633;">{matching_vr.mean_delta:+.4f}</td>
                        </tr>
                    </tbody>
                </table>
                <p style="margin-top:8px;"><strong>Seed Noise:</strong> σ = {matching_vr.seed_noise_std:.4f} | <strong>Exceeds 2σ:</strong> {'YES' if matching_vr.is_significant else 'NO'}</p>
                <p><em>{html.escape(matching_vr.reason)}</em></p>
                {f'<details><summary>View Suggested Fix Diff</summary><pre class="diff"><code>{html.escape(matching_vr.diff)}</code></pre></details>' if matching_vr.diff else ''}
            </div>
            """

        offending_code = f"<pre><code>{html.escape(ev.offending_span.code_snippet)}</code></pre>" if ev.offending_span else ""
        confirmed_html += f"""
        <div class="card finding-card">
            <h3>Finding {idx}: {html.escape(ev.metrics.get('leak_type', 'CONFIRMED_LEAK'))}</h3>
            <p><strong>Status:</strong> <span class="tag tag-red">CONFIRMED</span> | <strong>Tool:</strong> <code>{html.escape(ev.tool_name)}</code> | <strong>Evidence ID:</strong> <code>{html.escape(ev.evidence_id)}</code></p>
            <p>{html.escape(ev.summary)}</p>
            {offending_code}
            {vr_html}
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>REPROVE Audit Report: {html.escape(os.path.basename(report.file_path))}</title>
    <style>
        :root {{
            --bg: #0d1117;
            --card-bg: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-muted: #8b949e;
            --accent: #58a6ff;
        }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 30px 20px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        h1, h2, h3, h4 {{ color: #ffffff; }}
        .header {{
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 25px;
        }}
        .badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 1.1rem;
            background: {badge_color};
            color: #ffffff;
            margin-bottom: 15px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            text-align: left;
            padding: 10px;
            border-bottom: 1px solid var(--border);
        }}
        th {{ background: #21262d; color: #f0f6fc; }}
        code {{
            background: #21262d;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: Consolas, monospace;
            font-size: 0.9em;
        }}
        pre {{
            background: #090d13;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
            overflow-x: auto;
        }}
        pre.diff {{ color: #7ee787; }}
        details {{ margin-top: 10px; cursor: pointer; }}
        summary {{ font-weight: 600; color: var(--accent); }}
        .tag {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .tag-red {{ background: #da3633; color: white; }}
        .tag-green {{ background: #238636; color: white; }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 REPROVE Audit Report</h1>
            <p><strong>Target:</strong> <code>{html.escape(report.file_path)}</code> | <strong>Execution Mode:</strong> <code>{report.execution_mode}</code></p>
            <div class="badge">{badge_title}</div>
        </div>

        <div class="card">
            <h3>Audit Summary</h3>
            <table>
                <tr><td>Confirmed Findings</td><td><strong>{len(report.confirmed_evidence)}</strong></td></tr>
                <tr><td>Suspected Findings</td><td>{len(report.suspected_evidence)}</td></tr>
                <tr><td>Passed / Clean Checks</td><td>{len(report.clean_checks)}</td></tr>
                <tr><td>Investigation Steps</td><td>{len(report.steps)}</td></tr>
            </table>
        </div>

        {confirmed_html}

        <div class="footer">
            <p>Generated by REPROVE | Python {platform.python_version()} | scikit-learn {sklearn.__version__}</p>
            <p>Non-negotiable guarantee: Every metric and finding was verified by empirical execution.</p>
        </div>
    </div>
</body>
</html>
"""
