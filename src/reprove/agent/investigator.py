"""LLM investigation agent: an open-weight model (Ollama) chooses which evidence tool to run next.

The LLM proposes, tools verify:
  * the model only picks tools and writes its reasoning; every finding comes from a deterministic
    tool, and its status comes from the controlled rerun (see audit_core.py);
  * replies are constrained to a JSON schema (Ollama structured outputs) and validated with pydantic;
    an invalid reply is retried once, then the run falls back to the deterministic plan and is
    marked FALLBACK_DETERMINISTIC with the reason;
  * when the agent finishes, any core check it skipped is run automatically ("coverage").
"""
import logging
import os
from typing import List, Optional, Tuple

import ollama
from pydantic import ValidationError

from reprove.agent.audit_core import AuditState
from reprove.agent.fallback import deterministic_fallback_audit
from reprove.agent.replay_cache import get_cached_report, save_cached_report
from reprove.agent.schemas import CORE_TOOLS, AgentAction, InvestigationReport
from reprove.parser import parse_file
from reprove.verifier import DEFAULT_SEEDS

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5-coder:7b"
HEALTH_CHECK_TIMEOUT = 3.0                                          # only for "is Ollama running?"
LLM_TIMEOUT = float(os.environ.get("REPROVE_LLM_TIMEOUT", "300"))   # each chat call (model load can be slow)

SYSTEM_PROMPT = """You are REPROVE, an auditor that checks machine-learning experiments for evaluation leakage.
The code you audit is shown with line numbers. You cannot run code yourself: you choose which
deterministic evidence tool to run next, read its result, and decide what to check next.
You never decide whether a leak is confirmed - the tools and a controlled rerun do that.

Tools:
- check_fit_before_split: are scalers/imputers/encoders/PCA/feature selectors fitted before train_test_split? (Type 1)
- check_split_overlap: hashes the rows of the real train/test split to find test rows copied from training. (Type 2)
- trace_test_usage: is the test split used for training, early stopping, or choosing models/hyperparameters? (Type 3)
- inspect_code_span: show the code around `suspected_line`, if you need to look closer.
- finish: stop when every relevant check has been run.

Rules: run each relevant check once (repeating a check gives the same result). Base every
hypothesis on specific lines of the code. Reply with ONE JSON object only, matching the schema:
{"thought": "...", "hypothesis": {"hypothesis_id": "hyp_1", "leak_type": "...", "description": "...",
 "suspected_line": <int or null>, "tool_to_call": "...", "tool_args": {}}}"""


def _ollama_problem(model_name: str) -> Optional[str]:
    """Return None if Ollama is reachable and the model is pulled, otherwise a human-readable reason."""
    try:
        listing = ollama.Client(timeout=HEALTH_CHECK_TIMEOUT).list()
    except Exception as e:
        return f"Ollama is not reachable ({type(e).__name__}). Start it with `ollama serve`."
    models = listing.get("models", []) if isinstance(listing, dict) else getattr(listing, "models", [])
    names = set()
    for m in models or []:
        name = m.get("model") or m.get("name") if isinstance(m, dict) else (getattr(m, "model", None) or getattr(m, "name", None))
        if name:
            names.add(name)
            if name.endswith(":latest"):
                names.add(name[: -len(":latest")])
    if model_name not in names:
        return f"Model '{model_name}' is not pulled in Ollama. Run: ollama pull {model_name}"
    return None


def _ask(client: "ollama.Client", model: str, messages: list) -> Tuple[Optional[AgentAction], Optional[str], Optional[str]]:
    """One LLM step with schema-constrained output and one retry. Returns (action, raw_content, error)."""
    schema = AgentAction.model_json_schema()
    error = None
    for attempt in range(2):
        msgs = list(messages)
        if attempt == 1:
            msgs.append({
                "role": "user",
                "content": f"Your previous reply was invalid ({error}). Reply with ONLY one JSON object matching the schema.",
            })
        try:
            response = client.chat(model=model, messages=msgs, format=schema, options={"temperature": 0})
            content = response["message"]["content"]
            return AgentAction.model_validate_json(content), content, None
        except ValidationError as e:
            error = f"schema validation failed: {e.errors()[0].get('msg', 'invalid')}"
        except Exception as e:  # timeout, connection reset, model error...
            error = f"{type(e).__name__}: {str(e)[:200]}"
        logger.warning("LLM step attempt %d failed: %s", attempt + 1, error)
    return None, None, error


def _run_llm_audit(file_path: str, model_name: str, seeds: List[int], max_steps: int, timeout: int) -> InvestigationReport:
    client = ollama.Client(timeout=LLM_TIMEOUT)
    state = AuditState(file_path, seeds=seeds, timeout=timeout)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Audit this ML experiment for evaluation leakage.\n\nFile: {file_path}\n\n"
                f"Code (line numbers on the left):\n{state.numbered_code()}\n\n"
                f"Pipeline stages found by static analysis:\n{state.graph_summary()}\n\n"
                "State your first hypothesis and choose a tool."
            ),
        },
    ]

    failure: Optional[str] = None
    for step_num in range(1, max_steps + 1):
        action, content, error = _ask(client, model_name, messages)
        if action is None:
            failure = f"LLM reply invalid twice at step {step_num}: {error}"
            break

        hyp = action.hypothesis
        state.hypotheses.append(hyp)
        if hyp.tool_to_call == "finish":
            state.add_finish_step(action.thought)
            break

        step = state.run_tool(
            hyp.tool_to_call,
            thought=action.thought,
            origin="agent",
            hypothesis=hyp,
            tool_args=hyp.tool_args,
            suspected_line=hyp.suspected_line,
        )
        remaining = state.remaining_core_tools()
        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": (
                f"Result of {hyp.tool_to_call} (status: {step.result}):\n{step.observation[:2500]}\n\n"
                f"Core checks not run yet: {remaining or 'none'}. "
                "Update your hypothesis and choose the next tool, or 'finish' if all relevant checks are done."
            ),
        })

    if failure:
        state.run_coverage(origin="fallback")
        return state.to_report("FALLBACK_DETERMINISTIC", model_name=model_name, fallback_reason=failure)

    state.run_coverage(origin="coverage")
    return state.to_report("LLM_OLLAMA", model_name=model_name)


def run_investigation(
    file_path: str,
    model_name: str = DEFAULT_MODEL,
    use_replay: bool = False,
    seeds: Optional[List[int]] = None,
    max_steps: int = 8,
    record: bool = False,
    timeout: int = 60,
) -> InvestigationReport:
    """
    1. --replay: return the recorded report for this exact code, if one exists.
    2. Otherwise run the LLM agent (or the deterministic fallback if Ollama/model is unavailable).
    3. --record: save the fresh report for later replay.
    """
    seeds = list(seeds) if seeds else list(DEFAULT_SEEDS)
    graph = parse_file(file_path)

    if use_replay:
        cached = get_cached_report(file_path, graph.raw_code)
        if cached is not None:
            return cached
        logger.info("No recorded run for this exact code; running live.")

    problem = _ollama_problem(model_name)
    if problem:
        logger.warning("%s Falling back to the deterministic plan.", problem)
        report = deterministic_fallback_audit(file_path, seeds=seeds, timeout=timeout, reason=problem, model_name=model_name)
    else:
        report = _run_llm_audit(file_path, model_name, seeds, max_steps, timeout)

    if record:
        save_cached_report(file_path, graph.raw_code, report)
    return report
