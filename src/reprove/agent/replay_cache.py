"""Replay cache: re-show a previously RECORDED audit instantly (for live demos).

Rules that keep replay honest:
  * a report is written only when explicitly recorded (`reprove audit ... --record`), never by tests
    or ordinary runs;
  * the key is a hash of the exact code (+ cache schema version) - no filename-only fallback, so an
    edited notebook can never be answered with an old result;
  * the replayed report keeps the mode of the run that produced it (`replayed_from`).
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Optional

from reprove.agent.schemas import InvestigationReport

CACHE_SCHEMA_VERSION = "v2"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def cache_dir() -> str:
    """REPROVE_CACHE_DIR env var > repo root (editable install) > current directory."""
    env = os.environ.get("REPROVE_CACHE_DIR")
    if env:
        return env
    if os.path.exists(os.path.join(_REPO_ROOT, "pyproject.toml")):
        return os.path.join(_REPO_ROOT, ".reprove_cache")
    return os.path.join(os.getcwd(), ".reprove_cache")


def get_cache_key(file_path: str, raw_code: str) -> str:
    base = os.path.splitext(os.path.basename(file_path))[0]
    normalized = raw_code.replace("\r\n", "\n").strip()
    digest = hashlib.sha256(f"{CACHE_SCHEMA_VERSION}\n{normalized}".encode("utf-8")).hexdigest()[:16]
    return f"{base}_{digest}"


def get_cached_report(file_path: str, raw_code: str) -> Optional[InvestigationReport]:
    path = os.path.join(cache_dir(), f"{get_cache_key(file_path, raw_code)}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            report = InvestigationReport.model_validate(json.load(f))
    except Exception:
        return None
    if report.execution_mode != "REPLAY_CACHE":
        report.replayed_from = report.execution_mode
    report.execution_mode = "REPLAY_CACHE"
    return report


def save_cached_report(file_path: str, raw_code: str, report: InvestigationReport) -> str:
    """Record a report for later replay. Returns the cache file path."""
    os.makedirs(cache_dir(), exist_ok=True)
    path = os.path.join(cache_dir(), f"{get_cache_key(file_path, raw_code)}.json")
    data = report.model_dump(mode="json")
    data["recorded_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path
