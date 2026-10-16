"""Deterministic tool to detect duplicate samples and split overlap across train/test splits."""
import hashlib
import os
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import imagehash

from reprove.parser.models import CodeSpan, ExperimentGraph, StageType
from reprove.tools.models import EvidenceObject, ToolStatus


def hash_tabular_row(row: np.ndarray, precision: int = 6) -> str:
    """Compute a deterministic MD5 hash for a numeric/tabular vector."""
    rounded = np.round(row, decimals=precision)
    return hashlib.md5(rounded.tobytes()).hexdigest()


def compute_array_overlap(
    X_train: Union[np.ndarray, pd.DataFrame],
    X_test: Union[np.ndarray, pd.DataFrame],
    precision: int = 6,
) -> Tuple[int, float, List[str]]:
    """
    Computes exact duplicate sample count and percentage between train and test arrays.
    Returns (duplicate_count, overlap_percentage, sample_overlapping_hashes).
    """
    arr_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
    arr_test = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else np.asarray(X_test)

    train_hashes: Set[str] = {hash_tabular_row(arr_train[i], precision) for i in range(len(arr_train))}
    test_hashes: List[str] = [hash_tabular_row(arr_test[i], precision) for i in range(len(arr_test))]

    overlapping = [h for h in test_hashes if h in train_hashes]
    dup_count = len(overlapping)
    overlap_pct = (dup_count / len(test_hashes) * 100.0) if len(test_hashes) > 0 else 0.0

    return dup_count, overlap_pct, overlapping[:5]


def check_split_overlap_arrays(
    X_train: Union[np.ndarray, pd.DataFrame],
    X_test: Union[np.ndarray, pd.DataFrame],
    source_span: Optional[CodeSpan] = None,
) -> EvidenceObject:
    """
    Deterministic check on actual train/test split data arrays.
    """
    dup_count, overlap_pct, sample_hashes = compute_array_overlap(X_train, X_test)

    ev_id = f"ev_overlap_{uuid.uuid4().hex[:8]}"
    if dup_count > 0:
        return EvidenceObject(
            evidence_id=ev_id,
            tool_name="check_split_overlap",
            status=ToolStatus.FOUND_LEAK,
            confidence=1.0,
            summary=(
                f"Evaluation Leakage (Type 2): Found {dup_count} duplicate samples "
                f"({overlap_pct:.2f}% of test set) shared identically between train and test splits. "
                f"The model can artificially memorize training rows and score high on identical test rows."
            ),
            offending_span=source_span,
            metrics={
                "leak_type": "TYPE_2_CROSS_SPLIT_OVERLAP",
                "duplicate_count": dup_count,
                "test_size": len(X_test),
                "overlap_percentage": round(overlap_pct, 2),
                "sample_hashes": sample_hashes,
            },
            suggested_fix=(
                "Deduplicate the dataset prior to splitting, perform group-based splitting "
                "(GroupKFold) if records share an entity ID, or apply resampling strictly after splitting."
            ),
        )

    return EvidenceObject(
        evidence_id=ev_id,
        tool_name="check_split_overlap",
        status=ToolStatus.CLEAN,
        confidence=1.0,
        summary="No duplicate samples detected between train and test splits (0.00% overlap).",
        metrics={"duplicate_count": 0, "test_size": len(X_test), "overlap_percentage": 0.0},
    )


def check_split_overlap_graph(graph: ExperimentGraph) -> List[EvidenceObject]:
    """
    AST analysis for Type 2 leakage: detects premature duplication, oversampling,
    or data augmentation applied prior to the train/test split.
    """
    splits = graph.find_splits()
    if not splits:
        return [
            EvidenceObject(
                evidence_id=f"ev_overlap_{uuid.uuid4().hex[:8]}",
                tool_name="check_split_overlap",
                status=ToolStatus.INCONCLUSIVE,
                confidence=0.0,
                summary="No dataset split found in graph to evaluate split overlap.",
            )
        ]

    findings: List[EvidenceObject] = []
    split_node = splits[0]
    split_line = split_node.code_span.start_line

    # Inspect nodes preceding split for duplication / oversampling signatures
    duplication_keywords = {"vstack", "concat", "resample", "RandomOverSampler", "SMOTE", "duplicate"}

    for node in graph.nodes:
        if node.code_span.start_line >= split_line:
            continue

        snippet = node.code_span.code_snippet.lower()
        matched = [kw for kw in duplication_keywords if kw.lower() in snippet]
        if matched:
            ev_id = f"ev_overlap_ast_{uuid.uuid4().hex[:8]}"
            findings.append(
                EvidenceObject(
                    evidence_id=ev_id,
                    tool_name="check_split_overlap",
                    status=ToolStatus.FOUND_LEAK,
                    confidence=0.90,
                    summary=(
                        f"Suspected Type 2 Leakage: Premature data augmentation/resampling operation "
                        f"({', '.join(matched)}) detected at line {node.code_span.start_line} before "
                        f"splitting at line {split_line}. Identical samples risk entering both train and test splits."
                    ),
                    offending_span=node.code_span,
                    context_span=split_node.code_span,
                    metrics={
                        "leak_type": "TYPE_2_PREMATURE_DUPLICATION",
                        "operation_line": node.code_span.start_line,
                        "split_line": split_line,
                        "detected_signatures": matched,
                    },
                    suggested_fix="Move oversampling, SMOTE, or record duplication to after the train/test split.",
                )
            )

    if not findings:
        return [
            EvidenceObject(
                evidence_id=f"ev_overlap_ast_{uuid.uuid4().hex[:8]}",
                tool_name="check_split_overlap",
                status=ToolStatus.CLEAN,
                confidence=0.85,
                summary="No premature duplication or oversampling operations detected prior to split in AST.",
            )
        ]

    return findings


def check_image_folders_overlap(
    train_dir: str, test_dir: str, max_hamming_distance: int = 2
) -> EvidenceObject:
    """
    Perceptual hash (pHash) analysis for image datasets across train and test directories.
    """
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        return EvidenceObject(
            evidence_id=f"ev_img_overlap_{uuid.uuid4().hex[:8]}",
            tool_name="check_split_overlap",
            status=ToolStatus.INCONCLUSIVE,
            confidence=0.0,
            summary=f"Image directories not found: {train_dir} or {test_dir}",
        )

    def get_hashes(folder: str) -> Dict[str, imagehash.ImageHash]:
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
        overlap_pct = (len(duplicates) / len(test_hashes) * 100.0) if test_hashes else 0.0
        return EvidenceObject(
            evidence_id=ev_id,
            tool_name="check_split_overlap",
            status=ToolStatus.FOUND_LEAK,
            confidence=1.0,
            summary=(
                f"Perceptual Image Leakage (Type 2): Found {len(duplicates)} duplicate/near-duplicate images "
                f"({overlap_pct:.2f}% of test set) shared across train and test directories."
            ),
            metrics={
                "leak_type": "TYPE_2_IMAGE_PHASH_OVERLAP",
                "duplicate_count": len(duplicates),
                "test_image_count": len(test_hashes),
                "overlap_percentage": round(overlap_pct, 2),
            },
            suggested_fix="Deduplicate images across splits or partition by patient/scene ID.",
        )

    return EvidenceObject(
        evidence_id=ev_id,
        tool_name="check_split_overlap",
        status=ToolStatus.CLEAN,
        confidence=1.0,
        summary="No duplicate or near-duplicate images detected between train and test folders via pHash.",
    )
