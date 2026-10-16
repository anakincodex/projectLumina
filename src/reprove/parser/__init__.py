"""AST and notebook parsing module for REPROVE."""
import os
from reprove.parser.ast_tracer import build_experiment_graph
from reprove.parser.models import CodeSpan, ExperimentGraph, PipelineNode, StageType
from reprove.parser.notebook import NotebookLineMapping, ParsedNotebook, load_notebook

__all__ = [
    "parse_file",
    "build_experiment_graph",
    "load_notebook",
    "ExperimentGraph",
    "PipelineNode",
    "CodeSpan",
    "StageType",
    "NotebookLineMapping",
    "ParsedNotebook",
]


def parse_file(file_path: str) -> ExperimentGraph:
    """
    Parse a Python script (.py) or Jupyter Notebook (.ipynb) and return a
    line-anchored, stage-classified ExperimentGraph.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.endswith(".ipynb"):
        parsed_nb = load_notebook(file_path)
        return build_experiment_graph(
            source_code=parsed_nb.unified_script,
            file_path=file_path,
            notebook_mapping=parsed_nb.mapping,
        )
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        return build_experiment_graph(
            source_code=source,
            file_path=file_path,
            notebook_mapping=None,
        )
