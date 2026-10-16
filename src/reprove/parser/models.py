"""Data models for code spans, pipeline stages, and the Experiment Graph."""
from enum import Enum
from typing import Dict, List, Optional, Tuple
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
    # Unified-script line -> (notebook cell index, line inside that cell). Empty for .py files.
    line_to_cell: Dict[int, Tuple[int, int]] = Field(default_factory=dict)

    def span_for_lines(self, start_line: int, end_line: Optional[int] = None) -> CodeSpan:
        """Build a CodeSpan for any line range of the unified script (keeps notebook cell mapping)."""
        lines = self.raw_code.splitlines()
        end_line = end_line or start_line
        start_line = max(1, start_line)
        end_line = min(max(start_line, end_line), max(1, len(lines)))
        cell_index = cell_line = None
        loc = self.line_to_cell.get(start_line)
        if loc:
            cell_index, cell_line = loc
        return CodeSpan(
            file_path=self.source_file,
            start_line=start_line,
            end_line=end_line,
            cell_index=cell_index,
            cell_line=cell_line,
            code_snippet="\n".join(lines[start_line - 1 : end_line]),
        )

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
