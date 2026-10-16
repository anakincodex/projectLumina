"""Data models for deterministic evidence objects and tool findings."""
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from reprove.parser.models import CodeSpan


class ToolStatus(str, Enum):
    """Execution status returned by a deterministic evidence tool."""
    FOUND_LEAK = "FOUND_LEAK"
    CLEAN = "CLEAN"
    INCONCLUSIVE = "INCONCLUSIVE"


class EvidenceObject(BaseModel):
    """
    An immutable, empirically grounded evidence record produced by a deterministic tool.
    Non-negotiable rule: Every finding in REPROVE must cite a valid EvidenceObject.
    """
    evidence_id: str
    tool_name: str
    status: ToolStatus
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    offending_span: Optional[CodeSpan] = None
    context_span: Optional[CodeSpan] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    suggested_fix: Optional[str] = None

    @property
    def is_confirmed_leak(self) -> bool:
        """True if the tool deterministically proved an evaluation leak."""
        return self.status == ToolStatus.FOUND_LEAK
