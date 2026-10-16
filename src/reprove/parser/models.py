"""Data models for code spans, pipeline stages, and the Experiment Graph."""
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class StageType(str, Enum):
    """Categorized lifecycle stages of an ML experiment."""
    DATA_LOADING = "DATA_LOADING"
    SPLITTING = "SPLITTING"
    PREPROCESSING = "PREPROCESSING"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    MODEL_TRAINING = "MODEL_TRAINING"
    EVALUATION = "EVALUATION"
    OTHER = "OTHER"


class CodeSpan(BaseModel):
    """Represents an exact, line-anchored span of code in a source file or notebook."""
    file_path: str
    start_line: int
    end_line: int
    cell_index: Optional[int] = None
    cell_line: Optional[int] = None
    code_snippet: str

    def __str__(self) -> str:
        loc = f"L{self.start_line}-{self.end_line}"
        if self.cell_index is not None:
            loc = f"Cell[{self.cell_index}]:L{self.cell_line} ({loc})"
        return f"{self.file_path}:{loc}"


class PipelineNode(BaseModel):
    """A discrete operation or statement in the ML experiment pipeline."""
    node_id: str
    label: str
    stage: StageType
    code_span: CodeSpan
    caller: Optional[str] = None
    caller_class: Optional[str] = None
    function_name: Optional[str] = None
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)

    @property
    def is_fit_call(self) -> bool:
        """Check if this operation fits parameters onto data."""
        if not self.function_name:
            return False
        return self.function_name in {"fit", "fit_transform", "fit_predict", "fit_resample"}


class ExperimentGraph(BaseModel):
    """Line-indexed, topologically ordered graph of experiment stages and operations."""
    source_file: str
    raw_code: str
    nodes: List[PipelineNode] = Field(default_factory=list)

    def get_nodes_by_stage(self, stage: StageType) -> List[PipelineNode]:
        """Return all nodes matching a specific experiment stage."""
        return [node for node in self.nodes if node.stage == stage]

    def find_splits(self) -> List[PipelineNode]:
        """Return all splitting nodes."""
        return self.get_nodes_by_stage(StageType.SPLITTING)

    def find_fits(self) -> List[PipelineNode]:
        """Return all fit / fit_transform nodes."""
        return [node for node in self.nodes if node.is_fit_call]

    def find_node_by_id(self, node_id: str) -> Optional[PipelineNode]:
        """Find a node by its unique ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None
