"""Audit report generator module for REPROVE."""
from reprove.report.generator import generate_report
from reprove.report.html import render_html_report
from reprove.report.markdown import render_markdown_report

__all__ = [
    "generate_report",
    "render_markdown_report",
    "render_html_report",
]
