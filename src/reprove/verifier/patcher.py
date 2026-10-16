"""AST and source code patcher that generates corrected pipelines for leakage flaws."""
import difflib
import re
from typing import Optional, Tuple
from reprove.tools.models import EvidenceObject


def generate_patch(
    raw_code: str, evidence: EvidenceObject
) -> Tuple[Optional[str], Optional[str]]:
    """
    Given the original code and a confirmed leak EvidenceObject,
    generates a patched clean version of the code and a unified diff.
    Returns (patched_code, unified_diff_str).
    """
    leak_type = evidence.metrics.get("leak_type", "")
    patched_code: Optional[str] = None

    if leak_type == "TYPE_1_FIT_BEFORE_SPLIT" or "fit_before_split" in evidence.evidence_id:
        # Patch for Type 1: Fit after split on X_train only
        patched_code = _patch_fit_before_split(raw_code, evidence)

    elif (
        leak_type in {"TYPE_2_CROSS_SPLIT_OVERLAP", "TYPE_2_PREMATURE_DUPLICATION"}
        or "overlap" in evidence.evidence_id
    ):
        # Patch for Type 2: Deduplicate data prior to splitting
        patched_code = _patch_duplicate_split(raw_code, evidence)

    if not patched_code or patched_code == raw_code:
        return None, None

    # Generate unified diff
    orig_lines = raw_code.splitlines(keepends=True)
    patch_lines = patched_code.splitlines(keepends=True)
    diff = "".join(
        difflib.unified_diff(
            orig_lines,
            patch_lines,
            fromfile="original_leaked.py",
            tofile="reprove_patched.py",
        )
    )

    return patched_code, diff


def _patch_fit_before_split(code: str, evidence: EvidenceObject) -> Optional[str]:
    """
    Rewrites preprocessing fitted on full data into clean train-only fitting.
    Handles single-line and multi-line train_test_split calls.
    """
    lines = code.splitlines()

    # Find the range of lines encompassing the train_test_split call
    split_start_idx = -1
    split_end_idx = -1
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        if "train_test_split" in line:
            split_start_idx = idx
            # Search forward for the matching closing parenthesis
            for j in range(idx, len(lines)):
                if ")" in lines[j]:
                    split_end_idx = j
                    break
            break

    if split_start_idx == -1 or split_end_idx == -1:
        return None

    # Extract test_size or parameters from the original split call if present
    full_split_text = " ".join(lines[split_start_idx : split_end_idx + 1])
    test_size_match = re.search(r"test_size\s*=\s*([0-9.]+)", full_split_text)
    test_size = test_size_match.group(1) if test_size_match else "0.25"

    random_state_match = re.search(r"random_state\s*=\s*([0-9]+)", full_split_text)
    random_state = random_state_match.group(1) if random_state_match else "42"

    # Replacement clean block
    patch_block = (
        f"# --- [REPROVE PATCH: Split raw data first, fit transformers on X_train only] ---\n"
        f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state={random_state})\n\n"
        f"scaler = StandardScaler()\n"
        f"X_train = scaler.fit_transform(X_train)\n"
        f"X_test = scaler.transform(X_test)\n\n"
        f"selector = SelectKBest(score_func=f_classif, k=10)\n"
        f"X_train = selector.fit_transform(X_train, y_train)\n"
        f"X_test = selector.transform(X_test)\n"
        f"# --------------------------------------------------------------------------------\n"
    )

    cleaned_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Always preserve imports
        if stripped.startswith("import ") or stripped.startswith("from "):
            cleaned_lines.append(line)
            i += 1
            continue

        # Skip original leaked preprocessing calls
        if (
            "StandardScaler" in line
            or "SelectKBest" in line
            or "scaler.fit" in line
            or "selector.fit" in line
            or "X_scaled" in line
            or "X_selected" in line
        ):
            i += 1
            continue

        # When reaching the split call, replace its entire multi-line span with patch_block
        if i == split_start_idx:
            cleaned_lines.append(patch_block)
            i = split_end_idx + 1
            continue

        cleaned_lines.append(line)
        i += 1

    return "\n".join(cleaned_lines)


def _patch_duplicate_split(code: str, evidence: EvidenceObject) -> Optional[str]:
    """
    Inserts a deduplication step prior to train_test_split.
    """
    lines = code.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        if "train_test_split" in line:
            dedup_patch = (
                "\n# --- [REPROVE PATCH: Deduplicate data before splitting] ---\n"
                "_, _unique_indices = np.unique(X, axis=0, return_index=True)\n"
                "X = X[_unique_indices]\n"
                "y = y[_unique_indices]\n"
                "# ------------------------------------------------------------\n"
            )
            lines.insert(idx, dedup_patch)
            return "\n".join(lines)

    return None
