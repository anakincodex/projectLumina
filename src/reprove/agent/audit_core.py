"""Shared audit engine used by both the LLM agent and the deterministic fallback.

It executes evidence tools, runs the controlled rerun for every tool finding, files each finding
under the status the CODE decides (never the LLM), de-duplicates repeated tool calls, and can run
any core check the agent skipped ("coverage") so an early `finish` can never produce a false CLEAN.
"""
from typing import Any, Dict, List, Optional

from reprove.agent.schemas import CORE_TOOLS, AgentStep, Hypothesis, InvestigationReport
from reprove.parser import parse_file
from reprove.tools import (
    EvidenceObject,
    ToolStatus,
    check_fit_before_split,
    check_split_overlap,
    trace_test_usage,
)
from reprove.verifier import DEFAULT_SEEDS, FindingStatus, VerificationResult, verify_leakage

COVERAGE_THOUGHT = "Coverage guarantee: this core check had not been run yet, so REPROVE ran it automatically."


class AuditState:
    def __init__(self, file_path: str, seeds: Optional[List[int]] = None, timeout: int = 60):
        self.file_path = file_path
        self.seeds = list(seeds) if seeds else list(DEFAULT_SEEDS)
        self.timeout = timeout
        self.graph = parse_file(file_path)
        self.steps: List[AgentStep] = []
        self.hypotheses: List[Hypothesis] = []
        self.confirmed: List[EvidenceObject] = []
        self.not_confirmed: List[EvidenceObject] = []
        self.suspected: List[EvidenceObject] = []
        self.clean: List[EvidenceObject] = []
        self.inconclusive: List[EvidenceObject] = []
        self.verifications: List[VerificationResult] = []
        self._tool_observations: Dict[str, str] = {}

    # ------------------------------------------------------------------ helpers
    def numbered_code(self, max_lines: int = 220) -> str:
        lines = self.graph.raw_code.splitlines()
        shown = [f"{i:4d} | {line}" for i, line in enumerate(lines[:max_lines], start=1)]
        if len(lines) > max_lines:
            shown.append(f"... ({len(lines) - max_lines} more lines not shown)")
        return "\n".join(shown)

    def graph_summary(self) -> str:
        out = []
        for node in self.graph.nodes:
            if node.stage.value == "OTHER":
                continue
            out.append(f"- [{node.stage.value}] line {node.code_span.start_line}: {node.label}")
        return "\n".join(out) or "(no pipeline stages recognised)"

    def remaining_core_tools(self) -> List[str]:
        return [t for t in CORE_TOOLS if t not in self._tool_observations]

    def _verify(self, ev: EvidenceObject) -> VerificationResult:
        try:
            vr = verify_leakage(self.file_path, ev, seeds=self.seeds, timeout=self.timeout)
        except Exception as e:  # the audit must never crash because of one finding
            vr = VerificationResult(
                evidence_id=ev.evidence_id,
                status=FindingStatus.SUSPECTED,
                reason=f"SUSPECTED (not verified): verification raised {type(e).__name__}: {e}",
            )
        vr.evidence_id = ev.evidence_id
        return vr

    def _run_core_tool(self, tool_name: str) -> List[EvidenceObject]:
        if tool_name == "check_fit_before_split":
            return check_fit_before_split(self.graph)
        if tool_name == "check_split_overlap":
            return check_split_overlap(self.file_path, self.graph, seed=self.seeds[0], timeout=self.timeout)
        if tool_name == "trace_test_usage":
            return trace_test_usage(self.graph)
        raise ValueError(tool_name)

    # ------------------------------------------------------------------ public
    def run_tool(
        self,
        tool_name: str,
        thought: str,
        origin: str = "agent",
        hypothesis: Optional[Hypothesis] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        suspected_line: Optional[int] = None,
    ) -> AgentStep:
        tool_args = dict(tool_args or {})
        step_no = len(self.steps) + 1

        if tool_name == "inspect_code_span":
            line = suspected_line or tool_args.get("line") or tool_args.get("start_line") or 1
            try:
                line = int(line)
            except (TypeError, ValueError):
                line = 1
            lines = self.graph.raw_code.splitlines()
            lo, hi = max(1, line - 4), min(len(lines), line + 4)
            snippet = "\n".join(f"{i:4d} | {lines[i - 1]}" for i in range(lo, hi + 1))
            step = AgentStep(step_number=step_no, origin=origin, thought=thought, tool_name=tool_name,
                             tool_args={"line": line}, observation=f"Code around line {line}:\n{snippet}",
                             result="INSPECTED")
            self.steps.append(step)
            return step

        if tool_name not in CORE_TOOLS:
            step = AgentStep(step_number=step_no, origin=origin, thought=thought, tool_name=tool_name,
                             observation=f"Unknown tool '{tool_name}'.", result="ERROR")
            self.steps.append(step)
            return step

        if tool_name in self._tool_observations:
            step = AgentStep(step_number=step_no, origin=origin, thought=thought, tool_name=tool_name,
                             observation="Already executed earlier in this audit; result unchanged:\n"
                                         + self._tool_observations[tool_name],
                             result="DUPLICATE")
            self.steps.append(step)
            return step

        try:
            evidence = self._run_core_tool(tool_name)
        except Exception as e:
            evidence = [EvidenceObject(
                evidence_id=f"ev_error_{tool_name}", tool_name=tool_name, status=ToolStatus.INCONCLUSIVE,
                confidence=0.0, summary=f"The tool failed: {type(e).__name__}: {e}",
            )]

        notes, results, anchor = [], [], None
        for ev in evidence:
            if ev.status == ToolStatus.FOUND_LEAK:
                vr = self._verify(ev)
                self.verifications.append(vr)
                if vr.status == FindingStatus.CONFIRMED:
                    self.confirmed.append(ev)
                elif vr.status == FindingStatus.NOT_CONFIRMED:
                    self.not_confirmed.append(ev)
                else:
                    self.suspected.append(ev)
                results.append(vr.status.value)
                anchor = anchor or ev.evidence_id
                notes.append(f"[{ev.evidence_id}] {ev.summary}\nVerification -> {vr.reason}")
            elif ev.status == ToolStatus.CLEAN:
                self.clean.append(ev)
                results.append("CLEAN")
                notes.append(f"[{ev.evidence_id}] CLEAN: {ev.summary}")
            else:
                self.inconclusive.append(ev)
                results.append("INCONCLUSIVE")
                notes.append(f"[{ev.evidence_id}] INCONCLUSIVE: {ev.summary}")

        order = ["CONFIRMED", "SUSPECTED", "NOT_CONFIRMED", "INCONCLUSIVE", "CLEAN"]
        result = next((r for r in order if r in results), "CLEAN")
        observation = "\n".join(notes)
        self._tool_observations[tool_name] = observation

        step = AgentStep(step_number=step_no, origin=origin, thought=thought, tool_name=tool_name,
                         tool_args=tool_args, observation=observation, result=result, evidence_id=anchor)
        self.steps.append(step)
        return step

    def add_finish_step(self, thought: str, origin: str = "agent") -> AgentStep:
        step = AgentStep(step_number=len(self.steps) + 1, origin=origin, thought=thought,
                         tool_name="finish", observation="Investigation concluded by the agent.",
                         result="CONCLUDED")
        self.steps.append(step)
        return step

    def run_coverage(self, origin: str = "coverage") -> List[AgentStep]:
        """Run every core check the agent did not call."""
        return [self.run_tool(t, thought=COVERAGE_THOUGHT, origin=origin) for t in self.remaining_core_tools()]

    def to_report(self, mode: str, model_name: Optional[str] = None,
                  fallback_reason: Optional[str] = None) -> InvestigationReport:
        return InvestigationReport(
            file_path=self.file_path,
            execution_mode=mode,
            model_name=model_name,
            fallback_reason=fallback_reason,
            seeds=self.seeds,
            steps=self.steps,
            hypotheses_evaluated=self.hypotheses,
            confirmed_evidence=self.confirmed,
            not_confirmed_evidence=self.not_confirmed,
            suspected_evidence=self.suspected,
            clean_checks=self.clean,
            inconclusive_checks=self.inconclusive,
            verification_results=self.verifications,
        )
