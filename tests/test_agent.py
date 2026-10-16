"""Tests for LLM Investigation Agent schemas, deterministic fallback, and replay cache."""
import pytest
from pydantic import ValidationError

from reprove.agent import (
    AgentAction,
    AgentStep,
    Hypothesis,
    InvestigationReport,
    deterministic_fallback_audit,
    get_cached_report,
    run_investigation,
    save_cached_report,
)
from reprove.parser import parse_file


def test_schema_agent_action_validation():
    """Verify AgentAction schema validates valid tool calls and rejects malformed fields."""
    valid_data = {
        "thought": "I suspect SelectKBest was fitted before split.",
        "hypothesis": {
            "hypothesis_id": "hyp_1",
            "leak_type": "TYPE_1_FIT_BEFORE_SPLIT",
            "description": "SelectKBest fit_transform called before train_test_split",
            "suspected_line": 15,
            "tool_to_call": "check_fit_before_split",
            "tool_args": {},
        },
    }
    action = AgentAction.model_validate(valid_data)
    assert action.hypothesis.tool_to_call == "check_fit_before_split"
    assert action.hypothesis.suspected_line == 15

    # Missing required field
    with pytest.raises(ValidationError):
        AgentAction.model_validate({"thought": "Just a thought without hypothesis"})


def test_deterministic_fallback_audit_example_1():
    """Verify deterministic fallback audit detects and confirms leak in Example 1."""
    report = deterministic_fallback_audit(
        "examples/01_clear_leak_scaler_before_split.ipynb",
        seeds=[42],
    )
    assert report.execution_mode == "FALLBACK_DETERMINISTIC"
    assert report.has_confirmed_leaks is True
    assert len(report.confirmed_evidence) >= 1
    assert any("SelectKBest" in e.summary or "StandardScaler" in e.summary for e in report.confirmed_evidence)
    assert len(report.verification_results) >= 1
    assert report.verification_results[0].status.value == "CONFIRMED"


def test_deterministic_fallback_audit_example_2_clean():
    """Verify deterministic fallback audit confirms zero leaks on clean pipeline."""
    report = deterministic_fallback_audit(
        "examples/02_clean_pipeline_no_leak.ipynb",
        seeds=[42],
    )
    assert report.has_confirmed_leaks is False
    assert len(report.confirmed_evidence) == 0
    assert len(report.clean_checks) >= 2


def test_replay_cache_roundtrip(tmp_path):
    """Verify that reports can be cached and retrieved seamlessly."""
    dummy_report = InvestigationReport(
        file_path="examples/dummy.py",
        execution_mode="LLM_OLLAMA",
        steps=[
            AgentStep(
                step_number=1,
                thought="Checking fits",
                tool_name="check_fit_before_split",
                observation="Clean",
            )
        ],
    )
    dummy_code = "print('hello world')"
    save_cached_report("examples/dummy.py", dummy_code, dummy_report)

    cached = get_cached_report("examples/dummy.py", dummy_code)
    assert cached is not None
    assert cached.execution_mode == "REPLAY_CACHE"
    assert len(cached.steps) == 1
    assert cached.steps[0].tool_name == "check_fit_before_split"


def test_run_investigation_with_replay():
    """Verify run_investigation honors replay mode and executes without Ollama."""
    # Seed the cache with Example 1
    report = deterministic_fallback_audit(
        "examples/01_clear_leak_scaler_before_split.ipynb",
        seeds=[42],
    )
    graph = parse_file("examples/01_clear_leak_scaler_before_split.ipynb")
    save_cached_report("examples/01_clear_leak_scaler_before_split.ipynb", graph.raw_code, report)

    # Replay run
    replay_report = run_investigation(
        "examples/01_clear_leak_scaler_before_split.ipynb",
        use_replay=True,
    )
    assert replay_report.execution_mode == "REPLAY_CACHE"
    assert replay_report.has_confirmed_leaks is True
