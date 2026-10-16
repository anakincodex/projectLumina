"""Command-line interface for REPROVE."""
import logging
import os
import sys
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from reprove.agent.investigator import DEFAULT_MODEL, run_investigation
from reprove.agent.schemas import InvestigationReport
from reprove.report.generator import generate_report
from reprove.report.markdown import mode_description
from reprove.verifier.models import format_p

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Fallback reasons are shown in the report itself; keep library warnings off the console.
logging.getLogger("reprove").setLevel(logging.ERROR)

app = typer.Typer(
    name="reprove",
    help="Open-source agent that audits ML experiments for evaluation leakage.",
    add_completion=False,
)
console = Console(highlight=False)

STATUS_STYLE = {
    "CONFIRMED": "[bold white on red] CONFIRMED [/bold white on red]",
    "SUSPECTED": "[bold black on yellow] SUSPECTED [/bold black on yellow]",
    "NOT_CONFIRMED": "[bold white on blue] NOT CONFIRMED [/bold white on blue]",
    "CLEAN": "[bold white on green] CLEAN [/bold white on green]",
    "INCONCLUSIVE": "[bold white on grey37] INCONCLUSIVE [/bold white on grey37]",
}
RESULT_STYLE = {
    "CONFIRMED": "[bold red]CONFIRMED[/bold red]",
    "SUSPECTED": "[yellow]SUSPECTED[/yellow]",
    "NOT_CONFIRMED": "[blue]NOT CONFIRMED[/blue]",
    "CLEAN": "[green]CLEAN[/green]",
    "INCONCLUSIVE": "[dim]INCONCLUSIVE[/dim]",
    "CONCLUDED": "[cyan]FINISHED[/cyan]",
    "DUPLICATE": "[dim]ALREADY RUN[/dim]",
    "INSPECTED": "[dim]CODE SHOWN[/dim]",
}


def parse_seeds(seeds: str) -> List[int]:
    try:
        parsed = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    except ValueError:
        raise typer.BadParameter("Seeds must be comma-separated integers, e.g. 0,1,2,3,4")
    if not parsed:
        raise typer.BadParameter("Give at least one seed.")
    return parsed


def _render_terminal_summary(report: InvestigationReport, output_dir: Optional[str] = None):
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]REPROVE[/bold cyan] - ML evaluation leakage auditor\n"
            "[dim]The LLM proposes, deterministic tools and controlled reruns verify.[/dim]",
            border_style="cyan",
        )
    )
    console.print(f"[bold]Target:[/bold] [yellow]{report.file_path}[/yellow]")
    console.print(f"[bold]Mode:[/bold] [magenta]{report.execution_mode}[/magenta] - {mode_description(report)}")
    if report.fallback_reason:
        console.print(f"[bold]Fallback reason:[/bold] [yellow]{report.fallback_reason}[/yellow]")
    console.print(f"[bold]Seeds:[/bold] {', '.join(str(s) for s in report.seeds)}")
    console.print()

    steps = Table(title="Investigation Trace", border_style="dim")
    steps.add_column("#", style="dim", width=3)
    steps.add_column("Origin", width=9)
    steps.add_column("Reasoning", min_width=30)
    steps.add_column("Tool", style="cyan", width=24)
    steps.add_column("Result", width=15)
    for st in report.steps:
        thought = st.thought if len(st.thought) <= 90 else st.thought[:87] + "..."
        origin = {"agent": "[bold cyan]agent[/bold cyan]", "coverage": "[magenta]coverage[/magenta]"}.get(st.origin, st.origin)
        steps.add_row(str(st.step_number), origin, thought, st.tool_name, RESULT_STYLE.get(st.result, st.result or "-"))
    console.print(steps)
    console.print("[dim]agent = chosen by the LLM | coverage = run automatically because the agent skipped it | fallback = fixed plan[/dim]")
    console.print()

    findings = Table(title="Findings", border_style="bold cyan")
    findings.add_column("Status", width=17)
    findings.add_column("Type", width=26)
    findings.add_column("Evidence (from deterministic tools)", min_width=40)
    for label, items in (
        ("CONFIRMED", report.confirmed_evidence),
        ("SUSPECTED", report.suspected_evidence),
        ("NOT_CONFIRMED", report.not_confirmed_evidence),
        ("CLEAN", report.clean_checks),
        ("INCONCLUSIVE", report.inconclusive_checks),
    ):
        for ev in items:
            findings.add_row(STATUS_STYLE[label], str(ev.metrics.get("leak_type", ev.tool_name)), ev.summary)
    console.print(findings)
    console.print()

    ran = [v for v in report.verification_results if v.verification_ran]
    if ran:
        vt = Table(title="Controlled Rerun: original vs corrected code", border_style="red")
        vt.add_column("Metric", width=14)
        vt.add_column("Original (mean +/- std)", justify="right", width=22)
        vt.add_column("Corrected (mean +/- std)", justify="right", width=22)
        vt.add_column("Mean difference", justify="right", width=15)
        vt.add_column("p-value", justify="right", width=9)
        vt.add_column("Verdict", width=15)
        for vr in ran:
            p = format_p(vr.p_value)
            vt.add_row(
                vr.primary_metric_name,
                f"{vr.original_mean:.4f} +/- {vr.original_std:.4f}",
                f"{vr.patched_mean:.4f} +/- {vr.patched_std:.4f}",
                f"{vr.mean_delta:+.4f}",
                p,
                RESULT_STYLE.get(vr.status.value, vr.status.value),
            )
        console.print(vt)
        console.print("[dim]CONFIRMED = corrected code scores worse on average, one-sided paired t-test p < 0.05 over the seeds.[/dim]")
        console.print()

    for vr in report.verification_results:
        if not vr.verification_ran:
            console.print(f"[yellow]Not verified:[/yellow] {vr.reason}")

    verdict = report.verdict
    if verdict == "CONFIRMED":
        console.print(Panel(
            f"[bold red]VERDICT: {len(report.confirmed_evidence)} CONFIRMED EVALUATION LEAK(S)[/bold red]\n"
            "[red]Fixing the flaw made the score significantly worse: the reported metric is inflated.[/red]",
            border_style="red"))
    elif verdict == "SUSPECTED":
        console.print(Panel(
            f"[bold yellow]VERDICT: {len(report.suspected_evidence)} SUSPECTED LEAK(S) - NOT VERIFIED[/bold yellow]\n"
            "[yellow]A tool found the pattern, but an automatic rerun was not possible (reason above).[/yellow]",
            border_style="yellow"))
    elif verdict == "NOT_CONFIRMED":
        console.print(Panel(
            "[bold blue]VERDICT: LEAKAGE PATTERN PRESENT, NO MEASURABLE INFLATION[/bold blue]\n"
            "[blue]The pattern exists, but fixing it did not change the score beyond seed noise.[/blue]",
            border_style="blue"))
    elif verdict == "INCONCLUSIVE":
        console.print(Panel(
            "[bold]VERDICT: INCONCLUSIVE[/bold]\nSome checks could not run on this code.", border_style="white"))
    else:
        console.print(Panel(
            "[bold green]VERDICT: CLEAN EXPERIMENT PIPELINE[/bold green]\n"
            "[green]No leakage found by the fit-before-split, split-overlap and test-usage checks.[/green]",
            border_style="green"))

    if output_dir:
        console.print(f"\n[bold cyan]Saved reports:[/bold cyan] {output_dir}/report.md, {output_dir}/report.html")


@app.command()
def audit(
    file_path: str = typer.Argument(..., help="Python script (.py) or Jupyter notebook (.ipynb) to audit."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Open-weight model served by Ollama."),
    replay: bool = typer.Option(False, "--replay", "-r", help="Show the recorded run for this exact code (instant). Runs live if none exists."),
    record: bool = typer.Option(False, "--record", help="Save this run so it can be replayed later with --replay."),
    seeds: str = typer.Option("0,1,2,3,4", "--seeds", "-s", help="Comma-separated seeds for the controlled rerun (>= 3)."),
    timeout: int = typer.Option(60, "--timeout", help="Seconds allowed for each single run of the experiment."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Directory to write report.md and report.html."),
    fail_on_leak: bool = typer.Option(True, "--fail-on-leak/--no-fail-on-leak", help="Exit with code 1 if a leak is CONFIRMED."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show agent warnings (LLM retries, fallbacks)."),
):
    """Audit an ML experiment for evaluation leakage."""
    if verbose:
        logging.getLogger("reprove").setLevel(logging.INFO)
        logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
    if not os.path.exists(file_path):
        console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        raise typer.Exit(code=2)

    parsed_seeds = parse_seeds(seeds)
    with console.status(f"[bold cyan]Auditing [white]{file_path}[/white] (controlled reruns can take a minute)...[/bold cyan]"):
        report = run_investigation(
            file_path=file_path,
            model_name=model,
            use_replay=replay,
            seeds=parsed_seeds,
            record=record,
            timeout=timeout,
        )

    if output_dir:
        generate_report(report, output_dir=output_dir)
    _render_terminal_summary(report, output_dir=output_dir)
    if record:
        console.print("[dim]Recorded for replay (use --replay).[/dim]")

    if fail_on_leak and report.has_confirmed_leaks:
        raise typer.Exit(code=1)


@app.command()
def ui(port: int = typer.Option(8501, "--port", "-p", help="Port for the Streamlit dashboard.")):
    """Launch the Streamlit dashboard."""
    import subprocess

    app_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
    console.print(f"[bold cyan]Launching REPROVE UI at http://localhost:{port} ...[/bold cyan]")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path, "--server.port", str(port)])
    except KeyboardInterrupt:
        console.print("\n[yellow]UI stopped.[/yellow]")


@app.command()
def benchmark(
    replay: bool = typer.Option(True, "--replay/--no-replay", "-r", help="Use recorded runs where available."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Open-weight model served by Ollama."),
    seeds: str = typer.Option("0,1,2,3,4", "--seeds", "-s", help="Comma-separated seeds."),
):
    """Run REPROVE on every benchmark case and compare with the planted ground truth."""
    from reprove.evaluation import evaluate_benchmarks

    console.print()
    console.print(Panel.fit(
        "[bold cyan]REPROVE Benchmark[/bold cyan]\n[dim]Planted-leak notebooks + scripts written to stress the tool[/dim]",
        border_style="cyan"))

    with console.status("[bold cyan]Running benchmark cases...[/bold cyan]"):
        summary = evaluate_benchmarks(use_replay=replay, model=model, seeds=parse_seeds(seeds))

    table = Table(title="Benchmark Results", border_style="bold cyan")
    table.add_column("Case", style="dim", no_wrap=True)
    table.add_column("Name", ratio=1)
    table.add_column("Expected", no_wrap=True)
    table.add_column("Verdict", no_wrap=True)
    table.add_column("Mean diff", justify="right", no_wrap=True)
    table.add_column("Match", no_wrap=True)
    for r in summary.results:
        table.add_row(
            r.case_id,
            r.name,
            r.expected_status.replace("NOT_CONFIRMED", "NOT CONF."),
            RESULT_STYLE.get(r.verdict, r.verdict).replace("NOT CONFIRMED", "NOT CONF."),
            f"{r.empirical_delta:+.4f}" if r.empirical_delta is not None else "-",
            "[green]yes[/green]" if r.status_match else ("[yellow]flagged[/yellow]" if r.is_correct else "[red]NO[/red]"),
        )
    console.print(table)
    modes = sorted({r.mode for r in summary.results})
    console.print(f"[dim]Mode(s) used: {', '.join(modes)}[/dim]")

    perf = Table(title=f"Detection on {summary.total_cases} cases", border_style="bold green")
    perf.add_column("Metric", style="bold white", width=26)
    perf.add_column("Value", justify="right", width=12)
    perf.add_row("Leaky cases flagged", f"{summary.true_positives}/{summary.true_positives + summary.false_negatives}")
    perf.add_row("Clean cases left clean", f"{summary.true_negatives}/{summary.true_negatives + summary.false_positives}")
    perf.add_row("Exact status matches", f"{summary.status_matches}/{summary.total_cases}")
    perf.add_row("Precision", f"{summary.precision * 100:.1f}%")
    perf.add_row("Recall", f"{summary.recall * 100:.1f}%")
    perf.add_row("Total runtime", f"{summary.total_runtime_seconds:.1f}s")
    console.print(perf)
    console.print("[dim]Small benchmark written by the team - use it as a sanity check, not as a general accuracy claim.[/dim]")
    console.print()


if __name__ == "__main__":
    app()
