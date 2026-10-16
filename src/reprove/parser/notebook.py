"""Jupyter Notebook parser and cell-to-script line coordinate mapper."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import nbformat


@dataclass
class NotebookLineMapping:
    """Tracks coordinates of each unified script line back to its original notebook cell."""
    # Maps unified script line (1-indexed) -> (cell_index, line_in_cell)
    line_to_cell: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    cell_sources: Dict[int, str] = field(default_factory=dict)

    def get_notebook_loc(self, script_line: int) -> Optional[Tuple[int, int]]:
        """Return (cell_index, line_in_cell) for a given unified script line."""
        return self.line_to_cell.get(script_line)


@dataclass
class ParsedNotebook:
    """A notebook compiled into a unified Python script with exact cell mappings."""
    file_path: str
    unified_script: str
    mapping: NotebookLineMapping
    raw_cells_count: int


def load_notebook(notebook_path: str) -> ParsedNotebook:
    """
    Read an .ipynb file using nbformat and combine code cells into a unified
    Python script while recording line-to-cell coordinate mappings.
    """
    nb = nbformat.read(notebook_path, as_version=4)
    mapping = NotebookLineMapping()

    script_lines: List[str] = []
    current_script_line = 1

    for cell_idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        source = cell.source or ""
        mapping.cell_sources[cell_idx] = source
        cell_lines = source.splitlines()

        if not cell_lines:
            # Empty code cell: pad a newline
            script_lines.append("")
            mapping.line_to_cell[current_script_line] = (cell_idx, 1)
            current_script_line += 1
            continue

        for cell_line_num, line in enumerate(cell_lines, start=1):
            # If the line contains IPython magics (e.g. %matplotlib, !pip), comment it out
            # so standard python AST parser parses it cleanly without errors
            stripped = line.strip()
            if stripped.startswith("%") or stripped.startswith("!"):
                line = f"# [IPython Magic]: {line}"

            script_lines.append(line)
            mapping.line_to_cell[current_script_line] = (cell_idx, cell_line_num)
            current_script_line += 1

        # Add a blank separating line between cells for clean AST separation
        script_lines.append("")
        mapping.line_to_cell[current_script_line] = (cell_idx, len(cell_lines) + 1)
        current_script_line += 1

    unified_script = "\n".join(script_lines)
    return ParsedNotebook(
        file_path=notebook_path,
        unified_script=unified_script,
        mapping=mapping,
        raw_cells_count=len(nb.cells),
    )
