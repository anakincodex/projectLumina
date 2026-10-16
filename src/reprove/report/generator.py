"""Unified report generator orchestrating Markdown and HTML generation."""
import os
from typing import Optional, Tuple

from reprove.agent.schemas import InvestigationReport
from reprove.report.html import render_html_report
from reprove.report.markdown import render_markdown_report


def generate_report(
    report: InvestigationReport,
    output_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Renders both Markdown and HTML audit reports from an InvestigationReport.
    If output_dir is provided, saves report.md and report.html to disk.
    Returns (markdown_text, html_text).
    """
    md_text = render_markdown_report(report)
    html_text = render_html_report(report)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        md_path = os.path.join(output_dir, "report.md")
        html_path = os.path.join(output_dir, "report.html")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_text)

    return md_text, html_text
