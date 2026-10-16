"""Investigation Agent module for REPROVE."""
from reprove.agent.audit_core import AuditState
from reprove.agent.fallback import deterministic_fallback_audit
from reprove.agent.investigator import DEFAULT_MODEL, run_investigation
from reprove.agent.replay_cache import cache_dir, get_cached_report, save_cached_report
from reprove.agent.schemas import (
    CORE_TOOLS,
    AgentAction,
    AgentStep,
    Hypothesis,
    InvestigationReport,
)

__all__ = [
    "run_investigation",
    "deterministic_fallback_audit",
    "AuditState",
    "DEFAULT_MODEL",
    "CORE_TOOLS",
    "cache_dir",
    "get_cached_report",
    "save_cached_report",
    "Hypothesis",
    "AgentAction",
    "AgentStep",
    "InvestigationReport",
]
