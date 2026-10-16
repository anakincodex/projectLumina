"""Helper evidence tool to inspect exact code spans with surrounding context."""
import os
from typing import Optional
from reprove.parser import parse_file
from reprove.parser.models import CodeSpan


def inspect_code_span(
    file_path: str,
    start_line: int,
    end_line: Optional[int] = None,
    context_lines: int = 3,
) -> CodeSpan:
    """
    Extracts the exact code span from a file (or notebook) including context lines.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    graph = parse_file(file_path)
    all_lines = graph.raw_code.splitlines()

    end_line = end_line or start_line
    ctx_start = max(1, start_line - context_lines)
    ctx_end = min(len(all_lines), end_line + context_lines)

    snippet = "\n".join(all_lines[ctx_start - 1 : ctx_end])

    return CodeSpan(
        file_path=file_path,
        start_line=ctx_start,
        end_line=ctx_end,
        code_snippet=snippet,
    )
