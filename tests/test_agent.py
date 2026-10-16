"""Tests for the investigation agent: schemas, coverage guard, retries/fallback, replay cache."""
import pytest
from pydantic import ValidationError

import reprove.agent.investigator as inv
from reprove.agent import (
    AgentAction,
    AgentStep,
    InvestigationReport,
    cache_dir,
    deterministic_fallback_audit,
    get_cached_report,
    run_investigation,
    save_cached_report,
)
from reprove.parser import parse_file

EX1 = "examples/01_clear_leak_scaler_before_split.ipynb"
EX2 = "examples/02_clean_pipeline_no_leak.ipynb"
SEEDS = [0, 1, 2]


def _action(tool, leak_type="NONE", line=None):
    return AgentAction.model_validate({
        "thought": f"call {tool}",
        "hypothesis": {"hypothesis_id": "h", "leak_type": leak_type, "description": "d",
                       "suspected_line": line, "tool_to_call": tool, "tool_args": {}},
    })


def test_schema_validation_rejects_unknown_tools():
    ok = _action("check_fit_before_split", "TYPE_1_FIT_BEFORE_SPLIT", 15)
    assert ok.hypothesis.tool_to_call == "check_fit_before_split"
    with pytest.raises(ValidationError):
        _action("delete_all_files")
    with pytest.raises(ValidationError):
        AgentAction.model_validate({"thought": "no hypothesis"})


def test_fallback_confirms_example_1():
    report = deterministic_fallback_audit(EX1, seeds=SEEDS)
    assert report.execution_mode == "FALLBACK_DETERMINISTIC"
    assert report.verdict == "CONFIRMED"
    assert len(report.confirmed_evidence) == 1          # scaler + selector grouped into ONE finding
    vr = report.verification_for(report.confirmed_evidence[0].evidence_id)
    assert vr is not None and vr.verification_ran and len(vr.seed_runs) == 3


def test_fallback_clean_example_2():
    report = deterministic_fallback_audit(EX2, seeds=SEEDS)
    assert report.verdict == "CLEAN"
    assert len(report.clean_checks) == 3


def test_agent_finishing_early_cannot_hide_a_leak(monkeypatch):
    """The LLM says 'finish' immediately; coverage must still run every core check."""
    monkeypatch.setattr(inv, "_ollama_problem", lambda m: None)
    monkeypatch.setattr(inv, "_ask", lambda client, model, msgs: (_action("finish"), "{}", None))
    report = run_investigation(EX1, seeds=SEEDS)
    assert report.execution_mode == "LLM_OLLAMA"
    assert report.verdict == "CONFIRMED"
    assert [s.origin for s in report.steps] == ["agent", "coverage", "coverage", "coverage"]


def test_agent_repeated_tool_is_not_rerun(monkeypatch):
    script = iter([_action("check_fit_before_split", "TYPE_1_FIT_BEFORE_SPLIT"),
                   _action("check_fit_before_split", "TYPE_1_FIT_BEFORE_SPLIT"),
                   _action("finish")])
    monkeypatch.setattr(inv, "_ollama_problem", lambda m: None)
    monkeypatch.setattr(inv, "_ask", lambda client, model, msgs: (next(script), "{}", None))
    report = run_investigation(EX1, seeds=SEEDS)
    assert [s.result for s in report.steps][:2] == ["CONFIRMED", "DUPLICATE"]
    assert len(report.confirmed_evidence) == 1


def test_invalid_llm_output_falls_back_with_reason(monkeypatch):
    monkeypatch.setattr(inv, "_ollama_problem", lambda m: None)
    monkeypatch.setattr(inv, "_ask", lambda client, model, msgs: (None, None, "schema validation failed"))
    report = run_investigation(EX1, seeds=SEEDS)
    assert report.execution_mode == "FALLBACK_DETERMINISTIC"
    assert "invalid" in report.fallback_reason
    assert report.verdict == "CONFIRMED"


def test_llm_timeout_is_not_the_health_check_timeout():
    assert inv.LLM_TIMEOUT >= 60
    assert inv.HEALTH_CHECK_TIMEOUT < 10


def test_replay_cache_roundtrip_keeps_original_mode():
    report = InvestigationReport(
        file_path="examples/dummy.py",
        execution_mode="LLM_OLLAMA",
        model_name="qwen2.5-coder:7b",
        steps=[AgentStep(step_number=1, thought="t", tool_name="check_fit_before_split", observation="ok")],
    )
    path = save_cached_report("examples/dummy.py", "print('hi')", report)
    assert path.startswith(cache_dir())
    cached = get_cached_report("examples/dummy.py", "print('hi')")
    assert cached.execution_mode == "REPLAY_CACHE"
    assert cached.replayed_from == "LLM_OLLAMA"
    assert cached.recorded_at
    # Different code => no stale replay, even with the same file name
    assert get_cached_report("examples/dummy.py", "print('changed')") is None


def test_ordinary_runs_do_not_write_the_cache():
    import os
    before = set(os.listdir(cache_dir())) if os.path.exists(cache_dir()) else set()
    run_investigation(EX2, seeds=SEEDS)
    after = set(os.listdir(cache_dir())) if os.path.exists(cache_dir()) else set()
    assert before == after


def test_replay_returns_recorded_run():
    graph = parse_file(EX1)
    save_cached_report(EX1, graph.raw_code, deterministic_fallback_audit(EX1, seeds=SEEDS))
    replay = run_investigation(EX1, use_replay=True)
    assert replay.execution_mode == "REPLAY_CACHE"
    assert replay.replayed_from == "FALLBACK_DETERMINISTIC"
    assert replay.has_confirmed_leaks
