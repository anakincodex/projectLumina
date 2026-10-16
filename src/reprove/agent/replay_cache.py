"""Replay Cache for deterministic, sub-second live demo execution and offline reproducibility."""
import hashlib
import json
import os
from typing import Optional

from reprove.agent.schemas import InvestigationReport

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CACHE_DIR = os.path.join(REPO_ROOT, ".reprove_cache")


def get_cache_key(file_path: str, raw_code: str) -> str:
    """Generate a deterministic hash key from the file basename and raw content."""
    base = os.path.basename(file_path).replace(".ipynb", "").replace(".py", "")
    code_hash = hashlib.sha256(raw_code.strip().encode("utf-8")).hexdigest()[:16]
    return f"{base}_{code_hash}"


def get_cached_report(file_path: str, raw_code: str) -> Optional[InvestigationReport]:
    """Retrieve pre-recorded investigation report if available in cache."""
    cache_key = get_cache_key(file_path, raw_code)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")

    # Also check without hash suffix for base examples
    base_name = os.path.basename(file_path).replace(".ipynb", "").replace(".py", "")
    fallback_cache_file = os.path.join(CACHE_DIR, f"{base_name}.json")

    target_file = None
    if os.path.exists(cache_file):
        target_file = cache_file
    elif os.path.exists(fallback_cache_file):
        target_file = fallback_cache_file

    if not target_file:
        return None

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        report = InvestigationReport.model_validate(data)
        report.execution_mode = "REPLAY_CACHE"
        return report
    except Exception:
        return None


def save_cached_report(file_path: str, raw_code: str, report: InvestigationReport) -> None:
    """Save an authentic investigation report to local cache for demo replay."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = get_cache_key(file_path, raw_code)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")

    base_name = os.path.basename(file_path).replace(".ipynb", "").replace(".py", "")
    canonical_file = os.path.join(CACHE_DIR, f"{base_name}.json")

    data = report.model_dump(mode="json")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with open(canonical_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
