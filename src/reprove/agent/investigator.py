"""LLM Investigation Agent with Ollama, bounded tool-calling loop, and replay cache."""
import json
import logging
import os
from typing import List, Optional

import ollama
from pydantic import ValidationError

from reprove.agent.fallback import DEFAULT_SEEDS, deterministic_fallback_audit
from reprove.agent.replay_cache import get_cached_report, save_cached_report
from reprove.agent.schemas import AgentAction, AgentStep, Hypothesis, InvestigationReport
from reprove.parser import parse_file
from reprove.tools import (
    ToolStatus,
    check_fit_before_split,
    check_split_overlap_graph,
    inspect_code_span,
    trace_test_usage,
)
from reprove.verifier import FindingStatus, verify_leakage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are REPROVE, an open-source autonomous agent that audits ML experiments for evaluation leakage.
CORE PRINCIPLE: The LLM proposes, tools verify. You must NEVER claim or confirm a leak without calling a deterministic evidence tool.

Available tools:
- "check_fit_before_split": Deterministically checks if scalers, encoders, or selectors were fitted before train_test_split.
- "check_split_overlap": Checks if duplicate or resampled samples exist across train/test splits.
- "trace_test_usage": Checks if test set variables were passed into training, tuning (GridSearchCV), or early stopping.
- "inspect_code_span": Retrieves exact surrounding source code lines around a line number.
- "finish": Concludes the audit when all hypotheses have been tested.

Respond STRICTLY with a JSON object matching this schema:
{
  "thought": "<reasoning about the code and what check to perform>",
  "hypothesis": {
    "hypothesis_id": "<e.g. hyp_1>",
    "leak_type": "<TYPE_1_FIT_BEFORE_SPLIT | TYPE_2_CROSS_SPLIT_OVERLAP | TYPE_3_TEST_SET_CONTAMINATION | TYPE_4_TARGET_LEAKAGE | NONE>",
    "description": "<what is suspected and why>",
    "suspected_line": <int or null>,
    "tool_to_call": "<check_fit_before_split | check_split_overlap | trace_test_usage | inspect_code_span | finish>",
    "tool_args": {}
  }
}
"""


def _format_graph_summary(graph) -> str:
    """Format nodes and stages into a clean summary for the LLM."""
    lines = [f"File: {graph.source_file}", "Experiment Graph Pipeline Nodes:"]
    for node in graph.nodes:
        span = f"L{node.code_span.start_line}"
        if node.code_span.cell_index is not None:
            span = f"Cell[{node.code_span.cell_index}]:L{node.code_span.cell_line}"
        lines.append(f"- [{node.stage.value}] ({span}): {node.label}")
    return "\n".join(lines)


def run_investigation(
    file_path: str,
    model_name: str = "qwen2.5-coder:7b",
    use_replay: bool = False,
    seeds: Optional[List[int]] = None,
    max_steps: int = 8,
) -> InvestigationReport:
    """
    Orchestrates the investigation:
    1. Replay cache check for instant demo reproducibility.
    2. Open-weight LLM tool-calling loop (<= max_steps).
    3. Automatic fallback to deterministic plan if Ollama is unavailable.
    """
    seeds = seeds or DEFAULT_SEEDS
    graph = parse_file(file_path)
    raw_code = graph.raw_code

    # 1. Replay cache check
    if use_replay:
        cached = get_cached_report(file_path, raw_code)
        if cached:
            return cached

    # 2. Check Ollama connectivity
    try:
        client = ollama.Client(timeout=1.5)
        client.list()
        is_ollama_available = True
    except Exception as e:
        logger.warning(f"Ollama not available ({e}). Falling back to deterministic plan.")
        is_ollama_available = False

    if not is_ollama_available:
        report = deterministic_fallback_audit(file_path, seeds=seeds)
        save_cached_report(file_path, raw_code, report)
        return report

    # 3. LLM Investigation Loop
    graph_summary = _format_graph_summary(graph)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Please audit this ML experiment for evaluation leakage:\n\n{graph_summary}\n\nFormulate your first hypothesis and select a tool to execute.",
        },
    ]

    steps: List[AgentStep] = []
    hypotheses: List[Hypothesis] = []
    confirmed_evidence = []
    suspected_evidence = []
    clean_checks = []
    verification_results = []

    for step_num in range(1, max_steps + 1):
        try:
            response = client.chat(
                model=model_name,
                messages=messages,
                format="json",
            )
            content = response["message"]["content"]
            action_data = json.loads(content)
            action = AgentAction.model_validate(action_data)
        except Exception as e:
            logger.warning(f"Failed to parse LLM response at step {step_num}: {e}. Falling back.")
            report = deterministic_fallback_audit(file_path, seeds=seeds)
            save_cached_report(file_path, raw_code, report)
            return report

        hyp = action.hypothesis
        hypotheses.append(hyp)
        tool_name = hyp.tool_to_call.strip().lower()

        if tool_name == "finish":
            steps.append(
                AgentStep(
                    step_number=step_num,
                    thought=action.thought,
                    tool_name="finish",
                    observation="Investigation concluded by agent.",
                )
            )
            break

        # Execute chosen tool
        observation = ""
        evidence_id = None

        if tool_name == "check_fit_before_split":
            findings = check_fit_before_split(graph)
            for ev in findings:
                if ev.status == ToolStatus.FOUND_LEAK:
                    observation += ev.summary + "\n"
                    evidence_id = ev.evidence_id
                    v_res = verify_leakage(file_path, ev, seeds=seeds)
                    if v_res.status == FindingStatus.CONFIRMED:
                        confirmed_evidence.append(ev)
                        verification_results.append(v_res)
                    else:
                        suspected_evidence.append(ev)
                else:
                    clean_checks.append(ev)
                    observation += ev.summary + "\n"

        elif tool_name == "check_split_overlap":
            findings = check_split_overlap_graph(graph)
            for ev in findings:
                if ev.status == ToolStatus.FOUND_LEAK:
                    observation += ev.summary + "\n"
                    evidence_id = ev.evidence_id
                    v_res = verify_leakage(file_path, ev, seeds=seeds)
                    if v_res.status == FindingStatus.CONFIRMED:
                        confirmed_evidence.append(ev)
                        verification_results.append(v_res)
                    else:
                        suspected_evidence.append(ev)
                else:
                    clean_checks.append(ev)
                    observation += ev.summary + "\n"

        elif tool_name == "trace_test_usage":
            findings = trace_test_usage(graph)
            for ev in findings:
                if ev.status == ToolStatus.FOUND_LEAK:
                    observation += ev.summary + "\n"
                    evidence_id = ev.evidence_id
                    suspected_evidence.append(ev)
                else:
                    clean_checks.append(ev)
                    observation += ev.summary + "\n"

        elif tool_name == "inspect_code_span":
            line = hyp.suspected_line or 1
            span = inspect_code_span(file_path, start_line=line)
            observation = f"Code around L{line}:\n{span.code_snippet}"

        else:
            observation = f"Unknown tool: {tool_name}"

        steps.append(
            AgentStep(
                step_number=step_num,
                thought=action.thought,
                tool_name=tool_name,
                tool_args=hyp.tool_args,
                observation=observation.strip(),
                evidence_id=evidence_id,
            )
        )

        # Feed tool result back to agent conversation
        messages.append({"role": "assistant", "content": content})
        messages.append(
            {
                "role": "user",
                "content": f"Tool Execution Result ({tool_name}):\n{observation}\n\nUpdate your hypothesis or conclude with 'finish'.",
            }
        )

    report = InvestigationReport(
        file_path=file_path,
        execution_mode="LLM_OLLAMA",
        steps=steps,
        hypotheses_evaluated=hypotheses,
        confirmed_evidence=confirmed_evidence,
        suspected_evidence=suspected_evidence,
        clean_checks=clean_checks,
        verification_results=verification_results,
    )

    save_cached_report(file_path, raw_code, report)
    return report
