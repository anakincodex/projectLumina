"""Counterfactual patcher: rewrites a leaky pipeline into its corrected form.

Supported automatic corrections (anything else is reported honestly as "not verified"):
  * Type 1 - preprocessing fitted before the split. The fit/transform statements that feed the
    split are removed, the split is applied to the raw data, and each transformer is re-fitted on
    the training split only (then applied to the test split), in the original order.
    Recognised forms:   out = obj.fit_transform(x[, y])
                        out = obj.fit(x[, y]).transform(x)
                        obj.fit(x[, y])  ...  out = obj.transform(x)
  * Type 2 - duplicate rows across splits. Exact duplicate feature rows are dropped right before
    the split.
The patch is computed from the AST of the user's own code, using the variable names it actually uses.
"""
import ast
import difflib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from reprove.tools.models import EvidenceObject

TYPE_1 = {"TYPE_1_FIT_BEFORE_SPLIT"}
TYPE_2 = {"TYPE_2_CROSS_SPLIT_OVERLAP", "TYPE_2_PREMATURE_DUPLICATION"}
TYPE_3 = {"TYPE_3_TEST_SET_CONTAMINATION"}


class PatchUnavailable(Exception):
    """Raised when the code does not match a pattern the patcher can safely rewrite."""


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def generate_patch(raw_code: str, evidence: EvidenceObject) -> Tuple[Optional[str], Optional[str]]:
    """Backward-compatible API: returns (patched_code, unified_diff) or (None, None)."""
    patched, diff, _reason = generate_patch_with_reason(raw_code, evidence)
    return patched, diff


def generate_patch_with_reason(
    raw_code: str, evidence: EvidenceObject
) -> Tuple[Optional[str], Optional[str], str]:
    """Returns (patched_code, unified_diff, reason). On failure the first two are None."""
    leak_type = str(evidence.metrics.get("leak_type", ""))
    try:
        if leak_type in TYPE_1:
            patched = _patch_fit_before_split(raw_code, evidence)
        elif leak_type in TYPE_2:
            patched = _patch_duplicate_split(raw_code)
        elif leak_type in TYPE_3:
            raise PatchUnavailable(
                "automatic correction for test-set model selection is not implemented; "
                "use a validation split or cross-validation on the training data for selection"
            )
        else:
            raise PatchUnavailable(f"no automatic correction exists for leak type '{leak_type or 'unknown'}'")
    except PatchUnavailable as e:
        return None, None, str(e)
    except SyntaxError as e:
        return None, None, f"code could not be parsed ({e})"

    if not patched or patched == raw_code:
        return None, None, "the patcher produced no change"

    try:
        ast.parse(patched)
    except SyntaxError as e:  # never run a broken patch
        return None, None, f"generated patch is not valid Python ({e})"

    diff = "".join(
        difflib.unified_diff(
            raw_code.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile="original_leaked.py",
            tofile="reprove_patched.py",
        )
    )
    return patched, diff, "ok"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _iter_blocks(node: ast.AST):
    """Yield every statement list (module body, function bodies, if/for/with bodies...)."""
    for attr in ("body", "orelse", "finalbody"):
        block = getattr(node, attr, None)
        if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
            yield block
            for stmt in block:
                yield from _iter_blocks(stmt)
    for handler in getattr(node, "handlers", []) or []:
        yield from _iter_blocks(handler)


def _is_split_call(call: ast.AST) -> bool:
    if not isinstance(call, ast.Call):
        return False
    f = call.func
    return (isinstance(f, ast.Name) and f.id == "train_test_split") or (
        isinstance(f, ast.Attribute) and f.attr == "train_test_split"
    )


def _find_split(tree: ast.Module) -> Tuple[List[ast.stmt], int]:
    """Return (block, index) of the first `... = train_test_split(...)` statement."""
    best = None
    for block in _iter_blocks(tree):
        for i, stmt in enumerate(block):
            if isinstance(stmt, ast.Assign) and _is_split_call(stmt.value):
                if best is None or stmt.lineno < best[0][best[1]].lineno:
                    best = (block, i)
    if best is None:
        raise PatchUnavailable("no `... = train_test_split(...)` assignment was found")
    return best


def _target_names(stmt: ast.Assign) -> List[str]:
    if len(stmt.targets) != 1:
        return []
    t = stmt.targets[0]
    if isinstance(t, (ast.Tuple, ast.List)) and all(isinstance(e, ast.Name) for e in t.elts):
        return [e.id for e in t.elts]
    return []


def _src(code: str, node: ast.AST) -> str:
    seg = ast.get_source_segment(code, node)
    if seg is None:
        raise PatchUnavailable("could not read source segment")
    return seg


def _indent_of(lines: List[str], lineno: int) -> str:
    line = lines[lineno - 1]
    return line[: len(line) - len(line.lstrip())]


def _replace_nodes(lines: List[str], replacements: List[Tuple[ast.AST, str]]) -> None:
    """Replace the source text of AST nodes in-place (last node first so offsets stay valid)."""
    for node, text in sorted(replacements, key=lambda r: (r[0].lineno, r[0].col_offset), reverse=True):
        if node.lineno != node.end_lineno:
            raise PatchUnavailable("multi-line split arguments are not supported")
        raw = lines[node.lineno - 1].encode("utf-8")
        new = raw[: node.col_offset] + text.encode("utf-8") + raw[node.end_col_offset:]
        lines[node.lineno - 1] = new.decode("utf-8")


# --------------------------------------------------------------------------------------
# Type 1: preprocessing fitted before the split
# --------------------------------------------------------------------------------------
@dataclass
class _Op:
    kind: str                  # "fit_transform" | "fit" | "transform"
    index: int                 # position in block
    stmt: ast.stmt
    obj_src: str               # e.g. "scaler" or "StandardScaler()"
    x_name: Optional[str]
    y_name: Optional[str]
    out: Optional[str] = None
    fit_op: Optional["_Op"] = field(default=None, repr=False)


def _call_xy(call: ast.Call) -> Tuple[Optional[str], Optional[str]]:
    args = list(call.args)
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    x = args[0] if args else kw.get("X")
    y = args[1] if len(args) > 1 else kw.get("y")
    if x is None or not isinstance(x, ast.Name):
        raise PatchUnavailable("transformer input is not a plain variable name")
    if y is not None and not isinstance(y, ast.Name):
        raise PatchUnavailable("transformer target argument is not a plain variable name")
    return x.id, (y.id if y is not None else None)


def _parse_op(code: str, stmt: ast.stmt, index: int) -> Optional[_Op]:
    # out = obj.fit_transform(x[, y])   |   out = obj.fit(x[, y]).transform(x)   |   out = obj.transform(x)
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        call = stmt.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
            attr = call.func.attr
            if attr == "fit_transform":
                x, y = _call_xy(call)
                return _Op("fit_transform", index, stmt, _src(code, call.func.value), x, y, stmt.targets[0].id)
            if attr == "transform":
                inner = call.func.value
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "fit"
                ):
                    x, y = _call_xy(inner)
                    return _Op("fit_transform", index, stmt, _src(code, inner.func.value), x, y, stmt.targets[0].id)
                x, _ = _call_xy(call)
                return _Op("transform", index, stmt, _src(code, call.func.value), x, None, stmt.targets[0].id)
    # obj.fit(x[, y])
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        if isinstance(call.func, ast.Attribute) and call.func.attr == "fit":
            x, y = _call_xy(call)
            return _Op("fit", index, stmt, _src(code, call.func.value), x, y)
    return None


def _patch_fit_before_split(code: str, evidence: EvidenceObject) -> str:
    tree = ast.parse(code)
    block, si = _find_split(tree)
    split = block[si]
    outputs = _target_names(split)
    call = split.value
    if len(outputs) != 4 or len(call.args) != 2 or any(isinstance(a, ast.Starred) for a in call.args):
        raise PatchUnavailable("only `X_train, X_test, y_train, y_test = train_test_split(X, y, ...)` is supported")
    x_arg, y_arg = call.args
    if not isinstance(x_arg, ast.Name):
        raise PatchUnavailable("the first argument of train_test_split is not a plain variable")
    x_tr, x_te, y_tr, y_te = outputs

    fit_lines = set(int(n) for n in evidence.metrics.get("fit_lines", []) or [])
    if not fit_lines and evidence.offending_span:
        fit_lines = set(range(evidence.offending_span.start_line, evidence.offending_span.end_line + 1))

    # Parse every candidate fit/transform statement before the split in the same block
    ops: List[_Op] = []
    for i in range(si):
        stmt = block[i]
        try:
            op = _parse_op(code, stmt, i)
        except PatchUnavailable:
            if any(stmt.lineno <= ln <= stmt.end_lineno for ln in fit_lines):
                raise
            op = None  # unrelated statement we do not need to understand
        if op is not None:
            ops.append(op)

    # Link each `transform` to the most recent `fit` of the same object
    last_fit: Dict[str, _Op] = {}
    for op in ops:
        if op.kind == "fit":
            last_fit[op.obj_src] = op
        elif op.kind == "transform":
            op.fit_op = last_fit.get(op.obj_src)

    # Walk the data path backwards from the split input: which statements produced it?
    chain: List[_Op] = []
    current, pos = x_arg.id, si
    while True:
        producer = None
        for op in reversed(ops):
            if op.index < pos and op.out == current and op.kind in {"fit_transform", "transform"}:
                producer = op
                break
        if producer is None:
            break
        if producer.kind == "transform" and producer.fit_op is None:
            break  # transform with an object fitted elsewhere: not a pre-split fit
        chain.append(producer)
        current, pos = producer.x_name, producer.index
    chain.reverse()
    root = current

    if not chain:
        raise PatchUnavailable("the data passed to train_test_split is not produced by a fitted transformer")

    removed = {op.index for op in chain} | {op.fit_op.index for op in chain if op.fit_op is not None}
    removed_lines = set()
    for i in removed:
        removed_lines.update(range(block[i].lineno, block[i].end_lineno + 1))

    flagged_outside = fit_lines - removed_lines
    if flagged_outside:
        raise PatchUnavailable(
            f"the fit at line(s) {sorted(flagged_outside)} does not feed the split input; cannot build a faithful fix"
        )

    # Target arguments must be the split's y
    y_name = y_arg.id if isinstance(y_arg, ast.Name) else None
    for op in chain:
        fit = op.fit_op or op
        if fit.y_name is not None and fit.y_name != y_name:
            raise PatchUnavailable(f"transformer is fitted with target `{fit.y_name}`, not the split target")

    # Intermediate variables must not be needed by any statement we keep before the split
    intermediates = {op.out for op in chain if op.out and op.out != root}
    for i in range(si):
        if i in removed:
            continue
        used = {n.id for n in ast.walk(block[i]) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        clash = used & intermediates
        if clash:
            raise PatchUnavailable(f"variable(s) {sorted(clash)} are used elsewhere before the split")

    lines = code.splitlines()
    indent = _indent_of(lines, split.lineno)

    # 1) Build the re-fit block (applied in original order, training split only)
    new_lines = [f"{indent}# --- [REPROVE PATCH] preprocessing re-fitted on the training split only ---"]
    for n, op in enumerate(chain):
        fit = op.fit_op or op
        obj = op.obj_src
        if not obj.isidentifier():
            new_lines.append(f"{indent}_rp_step{n} = {obj}")
            obj = f"_rp_step{n}"
        y_part = f", {y_tr}" if fit.y_name is not None else ""
        new_lines.append(f"{indent}{obj}.fit({x_tr}{y_part})")
        new_lines.append(f"{indent}{x_tr} = {obj}.transform({x_tr})")
        new_lines.append(f"{indent}{x_te} = {obj}.transform({x_te})")
    new_lines.append(f"{indent}# --- [END REPROVE PATCH] ---")

    # 2) Split the raw data instead of the pre-transformed data
    _replace_nodes(lines, [(x_arg, root)])

    # 3) Insert the re-fit block after the split statement
    insert_at = split.end_lineno
    lines[insert_at:insert_at] = new_lines

    # 4) Comment out the removed statements (they are above the split, so indexes are unchanged)
    for i in sorted(removed, reverse=True):
        st = block[i]
        for ln in range(st.lineno, st.end_lineno + 1):
            original = lines[ln - 1]
            ind = original[: len(original) - len(original.lstrip())]
            lines[ln - 1] = f"{ind}# [REPROVE: moved below the split] {original.strip()}"

    return "\n".join(lines) + ("\n" if code.endswith("\n") else "")


# --------------------------------------------------------------------------------------
# Type 2: duplicate rows across the split
# --------------------------------------------------------------------------------------
def _patch_duplicate_split(code: str) -> str:
    tree = ast.parse(code)
    block, si = _find_split(tree)
    split = block[si]
    call = split.value
    args = [a for a in call.args if not isinstance(a, ast.Starred)]
    if not args or len(args) != len(call.args):
        raise PatchUnavailable("train_test_split arguments could not be analysed")

    lines = code.splitlines()
    indent = _indent_of(lines, split.lineno)

    x_src = _src(code, args[0])
    pre = [
        f"{indent}# --- [REPROVE PATCH] drop exact duplicate feature rows before splitting ---",
        f"{indent}import numpy as _rp_np",
        f"{indent}import pandas as _rp_pd",
        f"{indent}_rp_X = {x_src}",
        f"{indent}_rp_keep = ~_rp_pd.DataFrame(_rp_np.asarray(_rp_X).reshape(len(_rp_X), -1)).duplicated().to_numpy()",
        f"{indent}_rp_X = _rp_X[_rp_keep]",
    ]
    replacements = [(args[0], "_rp_X")]
    for k, other in enumerate(args[1:], start=1):
        pre.append(f"{indent}_rp_arr{k} = ({_src(code, other)})[_rp_keep]")
        replacements.append((other, f"_rp_arr{k}"))
    pre.append(f"{indent}# --- [END REPROVE PATCH] ---")

    _replace_nodes(lines, replacements)
    lines[split.lineno - 1 : split.lineno - 1] = pre
    return "\n".join(lines) + ("\n" if code.endswith("\n") else "")
