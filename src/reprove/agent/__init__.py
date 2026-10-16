"""Investigation Agent module for REPROVE."""
from reprove.agent.fallback import deterministic_fallback_audit
from reprove.agent.investigator import run_investigation
from reprove.agent.replay_cache import get_cached_report, save_cached_report
from reprove.agent.schemas import (
    AgentAction,
    AgentStep,
    Hypothesis,
    InvestigationReport,
)

__all__ = [
    "run_investigation",
    "deterministic_fallback_audit",
    "get_cached_report",
    "save_cached_report",
    "Hypothesis",
    "AgentAction",
    "AgentStep",
    "InvestigationReport",
]
