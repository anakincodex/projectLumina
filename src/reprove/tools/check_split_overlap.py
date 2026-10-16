"""Deterministic tool: duplicate samples shared by the train and test splits (Type 2).

Primary check (runtime): run the experiment once with a harness that saves the arrays returned by
train_test_split, then hash every row and count test rows that also appear in training.
Fallback (static hint): if the script cannot be run, look for duplication / oversampling calls
before the split. A static hint is never reported as proof.
"""
import hashlib
import os
import shutil
import tempfile
import uuid
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from reprove.parser.models import CodeSpan, ExperimentGraph
from reprove.tools.models import EvidenceObject, ToolStatus

# Calls that copy or synthesise rows. ("concat" is deliberately NOT here: joining columns is normal.)
DUPLICATION_SIGNATURES = (
    "vstack", "np.repeat", "np.tile", "resample(", "RandomOverSampler", "SMOTE", "ADASYN",
    "fit_resample", "replace=True",
)


def _loc(span: CodeSpan) -> str:
    if span.cell_index is not None:
        return f"cell [{span.cell_index}] line {span.cell_line}"
    return f"line {span.start_line}"


def hash_tabular_row(row: np.ndarray, precision: int = 6) -> str:
    """Deterministic MD5 hash of one row (numeric rows are rounded; other rows hashed by repr)."""
    arr = np.asarray(row)
    if np.issubdtype(arr.dtype, np.number):
        return hashlib.md5(np.round(arr.astype(float), decimals=precision).tobytes()).hexdigest()
    return hashlib.md5(repr(tuple(arr.ravel().tolist())).encode("utf-8")).hexdigest()


def compute_array_overlap(
    X_train: Union[np.ndarray, pd.DataFrame],
    X_test: Union[np.ndarray, pd.DataFrame],
    precision: int = 6,
) -> Tuple[int, float, List[str]]:
    """Returns (number of test rows also in train, % of test rows, up to 5 sample hashes)."""
    arr_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
    arr_test = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else np.asarray(X_test)
    if arr_train.ndim == 1:
        arr_train = arr_train.reshape(-1, 1)
    if arr_test.ndim == 1:
        arr_test = arr_test.reshape(-1, 1)

    train_hashes: Set[str] = {hash_tabular_row(r, precision) for r in arr_train}
    overlapping = [h for h in (hash_tabular_row(r, precision) for r in arr_test) if h in train_hashes]
    pct = (len(overlapping) / len(arr_test) * 100.0) if len(arr_test) else 0.0
    return len(overlapping), pct, overlapping[:5]


def check_split_overlap_arrays(
    X_train: Union[np.ndarray, pd.DataFrame],
    X_test: Union[np.ndarray, pd.DataFrame],
    source_span: Optional[CodeSpan] = None,
) -> EvidenceObject:
    """Deterministic check on actual train/test arrays."""
    dup_count, pct, sample = compute_array_overlap(X_train, X_test)
    ev_id = f"ev_overlap_{uuid.uuid4().hex[:8]}"
    if dup_count > 0:
        return EvidenceObject(
            evidence_id=ev_id,
            tool_name="check_split_overlap",
            status=ToolStatus.FOUND_LEAK,
            confidence=1.0,
            summary=(
                f"Type 2 leakage pattern: {dup_count} of {len(X_test)} test rows ({pct:.1f}%) are exact copies "
                f"of training rows, so the model is partly tested on data it was trained on."
            ),
            offending_span=source_span,
            metrics={
                "leak_type": "TYPE_2_CROSS_SPLIT_OVERLAP",
                "duplicate_count": dup_count,
                "test_size": len(X_test),
                "overlap_percentage": round(pct, 2),
                "sample_hashes": sample,
            },
            suggested_fix=(
                "Remove duplicate rows before splitting, oversample only the training split, or use a "
                "group-aware split (GroupShuffleSplit / GroupKFold) when rows share an entity."
            ),
        )
    return EvidenceObject(
        evidence_id=ev_id,
        tool_name="check_split_overlap",
        status=ToolStatus.CLEAN,
        confidence=1.0,
        summary=f"No test row is an exact copy of a training row (0 of {len(X_test)}).",
        offending_span=None,
        context_span=source_span,
        metrics={"duplicate_count": 0, "test_size": len(X_test), "overlap_percentage": 0.0},
    )


def _static_hints(graph: ExperimentGraph, split_line: int) -> List[Tuple[CodeSpan, List[str]]]:
    hints = []
    for node in graph.nodes:
        if node.code_span.start_line >= split_line:
            continue
        snippet = node.code_span.code_snippet
        matched = [sig for sig in DUPLICATION_SIGNATURES if sig in snippet]
        if matched:
            hints.append((node.code_span, matched))
    return hints


def check_split_overlap_graph(graph: ExperimentGraph) -> List[EvidenceObject]:
    """Static hint only: duplication / oversampling calls before the split (used when the code cannot run)."""
    splits = sorted(graph.find_splits(), key=lambda n: n.code_span.start_line)
    if not splits:
        return [
            EvidenceObject(
                evidence_id=f"ev_overlap_{uuid.uuid4().hex[:8]}",
                tool_name="check_split_overlap",
                status=ToolStatus.INCONCLUSIVE,
                confidence=0.0,
                summary="No dataset split found, so train/test overlap could not be checked.",
            )
        ]
    split = splits[0]
    hints = _static_hints(graph, split.code_span.start_line)
    if not hints:
        return [
            EvidenceObject(
                evidence_id=f"ev_overlap_ast_{uuid.uuid4().hex[:8]}",
                tool_name="check_split_overlap",
                status=ToolStatus.CLEAN,
                confidence=0.5,
                summary="Static scan: no duplication or oversampling call before the split (rows were not hashed).",
                context_span=split.code_span,
            )
        ]
    where = "; ".join(f"{', '.join(sig)} at {_loc(span)}" for span, sig in hints)
    return [
        EvidenceObject(
            evidence_id=f"ev_overlap_ast_{uuid.uuid4().hex[:8]}",
            tool_name="check_split_overlap",
            status=ToolStatus.FOUND_LEAK,
            confidence=0.5,
            summary=(
                f"Possible Type 2 leakage (static hint only, rows not hashed): row duplication/oversampling "
                f"before the split at {_loc(split.code_span)} ({where})."
            ),
            offending_span=hints[0][0],
            context_span=split.code_span,
            metrics={
                "leak_type": "TYPE_2_PREMATURE_DUPLICATION",
                "check": "static_hint",
                "split_line": split.code_span.start_line,
                "detected_signatures": sorted({s for _, sig in hints for s in sig}),
            },
            suggested_fix="Deduplicate before splitting and apply oversampling to the training split only.",
        )
    ]


def check_split_overlap(
    file_path: str,
    graph: ExperimentGraph,
    seed: int = 0,
    timeout: int = 60,
) -> List[EvidenceObject]:
    """
    Runtime row-hash check on the REAL split produced by the experiment (one run, given seed).
    Falls back to the static hint if the script cannot be executed.
    """
    from reprove.verifier.runner import PipelineExecutionError, execute_pipeline  # local import avoids a cycle

    splits = sorted(graph.find_splits(), key=lambda n: n.code_span.start_line)
    if not splits:
        return check_split_overlap_graph(graph)
    split = splits[0]

    tmp_dir = tempfile.mkdtemp(prefix="reprove_split_")
    dump_path = os.path.join(tmp_dir, "split.npz")
    try:
        try:
            execute_pipeline(
                graph.raw_code,
                seed,
                timeout=timeout,
                cwd=os.path.dirname(os.path.abspath(file_path)),
                split_dump_path=dump_path,
            )
            run_error = None
        except PipelineExecutionError as e:
            run_error = str(e)

        if not os.path.exists(dump_path):
            fallback = check_split_overlap_graph(graph)
            note = (
                f" Runtime row hashing was unavailable ({run_error.splitlines()[0] if run_error else 'train_test_split was not called at runtime'})."
            )
            for ev in fallback:
                ev.summary += note
            return fallback

        data: Dict[str, np.ndarray] = dict(np.load(dump_path, allow_pickle=True))
        evidence = check_split_overlap_arrays(data["X_train"], data["X_test"], source_span=split.code_span)
        if evidence.status == ToolStatus.FOUND_LEAK:
            hints = _static_hints(graph, split.code_span.start_line)
            evidence.metrics.update({"check": "runtime_row_hash", "seed": seed, "split_line": split.code_span.start_line})
            evidence.summary += f" (Rows hashed from the actual split at {_loc(split.code_span)}, seed {seed}.)"
            if hints:
                evidence.offending_span = hints[0][0]
                evidence.context_span = split.code_span
                evidence.summary += " Likely source: " + "; ".join(
                    f"{', '.join(sig)} at {_loc(span)}" for span, sig in hints
                ) + "."
        else:
            evidence.summary += f" (Rows hashed from the actual split at {_loc(split.code_span)}, seed {seed}.)"
        return [evidence]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def check_image_folders_overlap(
    train_dir: str, test_dir: str, max_hamming_distance: int = 2
) -> EvidenceObject:
    """Perceptual hash (pHash) check for image datasets stored in train/test folders."""
    from PIL import Image
    import imagehash

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        return EvidenceObject(
            evidence_id=f"ev_img_overlap_{uuid.uuid4().hex[:8]}",
            tool_name="check_split_overlap",
            status=ToolStatus.INCONCLUSIVE,
            confidence=0.0,
            summary=f"Image directories not found: {train_dir} or {test_dir}",
        )

    def get_hashes(folder: str) -> Dict[str, "imagehash.ImageHash"]:
        hashes = {}
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    p = os.path.join(root, file)
                    try:
                        with Image.open(p) as img:
                            hashes[p] = imagehash.phash(img)
                    except Exception:
                        continue
        return hashes

    train_hashes = get_hashes(train_dir)
    test_hashes = get_hashes(test_dir)

    duplicates: List[Tuple[str, str, int]] = []
    for test_path, t_hash in test_hashes.items():
        for train_path, tr_hash in train_hashes.items():
            dist = t_hash - tr_hash
            if dist <= max_hamming_distance:
                duplicates.append((train_path, test_path, dist))
                break

    ev_id = f"ev_img_overlap_{uuid.uuid4().hex[:8]}"
    if duplicates:
        pct = (len(duplicates) / len(test_hashes) * 100.0) if test_hashes else 0.0
        return EvidenceObject(
            evidence_id=ev_id,
            tool_name="check_split_overlap",
            status=ToolStatus.FOUND_LEAK,
            confidence=1.0,
            summary=(
                f"Type 2 leakage pattern (images): {len(duplicates)} test images ({pct:.1f}%) are duplicates or "
                f"near-duplicates (pHash distance <= {max_hamming_distance}) of training images."
            ),
            metrics={
                "leak_type": "TYPE_2_IMAGE_PHASH_OVERLAP",
                "duplicate_count": len(duplicates),
                "test_image_count": len(test_hashes),
                "overlap_percentage": round(pct, 2),
            },
            suggested_fix="Deduplicate images across splits or split by patient/scene/source ID.",
        )

    return EvidenceObject(
        evidence_id=ev_id,
        tool_name="check_split_overlap",
        status=ToolStatus.CLEAN,
        confidence=1.0,
        summary="No duplicate or near-duplicate images between train and test folders (pHash).",
    )
