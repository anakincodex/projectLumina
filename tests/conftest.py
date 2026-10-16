"""Shared test setup.

* Tests never touch the real replay cache: REPROVE_CACHE_DIR points to a temporary folder.
* Tests never call a real Ollama server (results must not depend on a local model).
* Benchmark cases are audited once per session (deterministic fallback, the default 5 seeds) and
  recorded, so the replay-based tests are fast.
"""
import os

import pytest



@pytest.fixture(scope="session", autouse=True)
def isolated_environment(tmp_path_factory):
    cache = tmp_path_factory.mktemp("reprove_cache")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("REPROVE_CACHE_DIR", str(cache))
        import reprove.agent.investigator as inv
        mp.setattr(inv, "_ollama_problem", lambda model: "Ollama disabled in the test suite.")
        yield str(cache)


@pytest.fixture(scope="session")
def recorded_benchmarks(isolated_environment):
    """Record a fallback audit for every benchmark case into the temporary cache."""
    from reprove.agent import run_investigation
    from reprove.evaluation import BENCHMARK_CASES
    from reprove.verifier import DEFAULT_SEEDS

    reports = {}
    for case in BENCHMARK_CASES:
        reports[case["id"]] = run_investigation(case["file_path"], seeds=DEFAULT_SEEDS, record=True)
    return reports
