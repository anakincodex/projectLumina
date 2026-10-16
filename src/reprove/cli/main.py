"""Command-line interface for REPROVE."""
import os
import sys
from typing import List, Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from reprove.agent.fallback import DEFAULT_SEEDS
from reprove.agent.investigator import run_investigation
from reprove.agent.schemas import InvestigationReport
from reprove.report.generator import generate_report
from reprove.verifier import FindingStatus

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

app = typer.Typer(
    name="reprove",
    help="Autonomous open-source agent that audits ML experiments for evaluation leakage.",
    add_completion=False,
)
console = Console(highlight=False)


def _render_terminal_summary(report: InvestigationReport, output_dir: Optional[str] = None):
    """Renders formatted Rich tables and panels to console."""
    console.print()
    # Header Banner
    console.print(
        Panel.fit(
            "[bold cyan]REPROVE[/bold cyan] :mag: [bold white]Autonomous ML Evaluation Leakage Auditor[/bold white]\n"
            "[dim]Principle: The LLM proposes, deterministic tools verify.[/dim]",
            border_style="cyan",
        )
    )

    console.print(f"[bold]Target File:[/bold] [yellow]{report.file_path}[/yellow]")
    console.print(f"[bold]Execution Mode:[/bold] [magenta]{report.execution_mode}[/magenta]")
    console.print(f"[bold]Agent Steps Executed:[/bold] {len(report.steps)}")
    console.print()

    # Step details
    step_table = Table(title="Investigation Trajectory", border_style="dim")
    step_table.add_column("Step", style="dim", width=6)
    step_table.add_column("Agent Thought", style="white", min_width=30)
    step_table.add_column("Tool Executed", style="cyan", width=24)
    step_table.add_column("Evidence Status", style="bold", width=18)

    for step in report.steps:
        ev_label = "[dim]None[/dim]"
        if step.tool_name == "finish":
            ev_label = "[blue]CONCLUDED[/blue]"
        elif "LEAK FOUND" in step.observation or "Leak found" in step.observation:
            ev_label = "[bold red]LEAK DETECTED[/bold red]"
        elif "CLEAN" in step.observation:
            ev_label = "[green]CLEAN[/green]"

        thought_snippet = step.thought
        if len(thought_snippet) > 80:
            thought_snippet = thought_snippet[:77] + "..."

        step_table.add_row(
            str(step.step_number),
            thought_snippet,
            step.tool_name,
            ev_label,
        )
    console.print(step_table)
    console.print()

    # Findings Table
    findings_table = Table(title="Audited Findings Verdict", border_style="bold cyan")
    findings_table.add_column("ID", style="dim", width=12)
    findings_table.add_column("Status", width=16)
    findings_table.add_column("Type / Tool", width=26)
    findings_table.add_column("Description / Empirical Evidence", min_width=40)

    for ev in report.confirmed_evidence:
        l_type = ev.metrics.get("leak_type", ev.tool_name)
        findings_table.add_row(
            ev.evidence_id,
            "[bold white on red] CONFIRMED [/bold white on red]",
            str(l_type),
            f"[bold]{ev.summary}[/bold]",
        )
    for ev in report.suspected_evidence:
        l_type = ev.metrics.get("leak_type", ev.tool_name)
        findings_table.add_row(
            ev.evidence_id,
            "[bold black on yellow] SUSPECTED [/bold black on yellow]",
            str(l_type),
            ev.summary,
        )
    for ev in report.clean_checks:
        l_type = ev.metrics.get("leak_type", ev.tool_name)
        findings_table.add_row(
            ev.evidence_id,
            "[bold white on green] CLEAN [/bold white on green]",
            str(l_type),
            ev.summary,
        )

    console.print(findings_table)
    console.print()

    # Verification Table (if any confirmed/evaluated)
    if report.verification_results:
        verif_table = Table(title="Controlled Counterfactual Verification (Multi-Seed)", border_style="red")
        verif_table.add_column("Metric", width=12)
        verif_table.add_column("Original Metric", justify="right", width=16)
        verif_table.add_column("Patched Metric", justify="right", width=16)
        verif_table.add_column("Measured Delta", justify="right", width=16)
        verif_table.add_column("Noise Margin (2*sigma)", justify="right", width=22)
        verif_table.add_column("Verdict", width=14)

        for vr in report.verification_results:
            status_style = (
                "[bold red]CONFIRMED[/bold red]"
                if vr.status == FindingStatus.CONFIRMED
                else f"[yellow]{vr.status.value}[/yellow]"
            )
            verif_table.add_row(
                vr.primary_metric_name,
                f"{vr.original_mean:.4f} (std {vr.original_std:.4f})",
                f"{vr.patched_mean:.4f} (std {vr.patched_std:.4f})",
                f"[bold red]{vr.mean_delta:+.4f}[/bold red]",
                f"+/- {2 * vr.seed_noise_std:.4f}",
                status_style,
            )
        console.print(verif_table)
        console.print()

    # Verdict summary banner
    if report.confirmed_evidence:
        console.print(
            Panel(
                f"[bold red]AUDIT VERDICT: 🚨 {len(report.confirmed_evidence)} CONFIRMED EVALUATION LEAK(S) FOUND[/bold red]\n"
                f"[red]The evaluation metric is empirically inflated. The model's true generalization performance is lower than reported.[/red]",
                border_style="red",
            )
        )
    elif report.suspected_evidence:
        console.print(
            Panel(
                f"[bold yellow]AUDIT VERDICT: ⚠️ {len(report.suspected_evidence)} SUSPECTED LEAK(S) FOUND (UNCONFIRMED)[/bold yellow]\n"
                f"[yellow]Deterministic checks detected anomalies but empirical rerun was inconclusive.[/yellow]",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(
                "[bold green]AUDIT VERDICT: ✅ CLEAN EXPERIMENT PIPELINE[/bold green]\n"
                "[green]No data leakage or test contamination detected across all deterministic checks.[/green]",
                border_style="green",
            )
        )

    if output_dir:
        console.print(f"\n[bold cyan]Saved Reports:[/bold cyan] {output_dir}/report.md, {output_dir}/report.html")


@app.command()
def audit(
    file_path: str = typer.Argument(..., help="Path to Python script (.py) or Jupyter notebook (.ipynb) to audit."),
    model: str = typer.Option("qwen2.5-coder:7b", "--model", "-m", help="Open-weight LLM model to use via Ollama."),
    replay: bool = typer.Option(False, "--replay", "-r", help="Use local replay cache for instantaneous audit results."),
    seeds: str = typer.Option("42,101,7", "--seeds", "-s", help="Comma-separated random seeds for multi-seed verification."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Directory to save report.md and report.html."),
    fail_on_leak: bool = typer.Option(True, "--fail-on-leak/--no-fail-on-leak", help="Exit with code 1 if confirmed leaks are found."),
):
    """Audits an ML experiment for evaluation leakage with LLM and deterministic tools."""
    if not os.path.exists(file_path):
        console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        raise typer.Exit(code=2)

    parsed_seeds = [int(s.strip()) for s in seeds.split(",") if s.strip()]

    with console.status(f"[bold cyan]Running REPROVE audit on [white]{file_path}[/white]...[/bold cyan]"):
        report = run_investigation(
            file_path=file_path,
            model_name=model,
            use_replay=replay,
            seeds=parsed_seeds,
        )

    if output_dir:
        generate_report(report, output_dir=output_dir)

    _render_terminal_summary(report, output_dir=output_dir)

    if fail_on_leak and len(report.confirmed_evidence) > 0:
        raise typer.Exit(code=1)


@app.command()
def ui(
    port: int = typer.Option(8501, "--port", "-p", help="Port for the Streamlit dashboard."),
):
    """Launches the REPROVE interactive Streamlit web dashboard."""
    import subprocess

    app_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", str(port)]
    console.print(f"[bold cyan]Launching REPROVE Streamlit UI at http://localhost:{port}...[/bold cyan]")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        console.print("\n[yellow]Streamlit UI stopped.[/yellow]")


@app.command()
def benchmark(
    replay: bool = typer.Option(True, "--replay/--no-replay", "-r", help="Use local replay cache for instantaneous execution."),
    model: str = typer.Option("qwen2.5-coder:7b", "--model", "-m", help="Open-weight LLM model to use."),
):
    """Evaluates REPROVE against all ground-truth benchmark experiments and outputs performance metrics."""
    from reprove.evaluation import evaluate_benchmarks

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]REPROVE Benchmark Evaluation Matrix[/bold cyan]\n"
            "[dim]Evaluating Precision, Recall, and Ground-Truth Accuracy across Experiments[/dim]",
            border_style="cyan",
        )
    )

    with console.status("[bold cyan]Evaluating benchmark cases...[/bold cyan]"):
        summary = evaluate_benchmarks(use_replay=replay, model=model)

    table = Table(title="Benchmark Execution Results", border_style="bold cyan")
    table.add_column("Case ID", style="dim", width=8)
    table.add_column("Benchmark Name", min_width=32)
    table.add_column("Expected", width=12)
    table.add_column("Detected Verdict", width=16)
    table.add_column("Empirical Delta", justify="right", width=16)
    table.add_column("Time (s)", justify="right", width=10)
    table.add_column("Status", width=10)

    for res in summary.results:
        exp_str = "[bold red]LEAK[/bold red]" if res.expected_has_leak else "[bold green]CLEAN[/bold green]"
        status_str = "[bold green]CORRECT[/bold green]" if res.is_correct else "[bold red]MISMATCH[/bold red]"
        delta_str = f"{res.empirical_delta:+.4f}" if res.empirical_delta is not None else "—"
        
        verdict_str = res.verdict
        if res.verdict == "CONFIRMED":
            verdict_str = "[bold red]CONFIRMED[/bold red]"
        elif res.verdict == "SUSPECTED":
            verdict_str = "[yellow]SUSPECTED[/yellow]"
        elif res.verdict == "CLEAN":
            verdict_str = "[green]CLEAN[/green]"

        table.add_row(
            res.case_id,
            res.name,
            exp_str,
            verdict_str,
            delta_str,
            f"{res.runtime_seconds:.3f}s",
            status_str,
        )

    console.print(table)
    console.print()

    # Metrics summary panel
    perf_table = Table(title="Aggregate Classification Metrics", border_style="bold green")
    perf_table.add_column("Metric", style="bold white", width=24)
    perf_table.add_column("Value", justify="right", width=16)

    perf_table.add_row("Precision", f"{summary.precision * 100:.1f}%")
    perf_table.add_row("Recall", f"{summary.recall * 100:.1f}%")
    perf_table.add_row("F1 Score", f"{summary.f1_score:.4f}")
    perf_table.add_row("Specificity", f"{summary.specificity * 100:.1f}%")
    perf_table.add_row("Total Runtime", f"{summary.total_runtime_seconds:.3f}s")
    perf_table.add_row("Total Test Cases", str(summary.total_cases))

    console.print(perf_table)
    console.print()


if __name__ == "__main__":
    app()

