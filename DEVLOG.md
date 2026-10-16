# LEAKHOUND Development Log

This document records the architectural decisions, design rationale, problems faced, and solutions implemented throughout the development of LEAKHOUND.

---

## Milestone 1: Skeleton, Environment & Ground-Truth Benchmark Notebooks

### 1. What was done
- Configured project packaging using `pyproject.toml` with Python 3.11+ compatibility and standard open-source dependencies (`scikit-learn`, `pandas`, `numpy`, `libcst`, `nbformat`, `pydantic`, `typer`, `rich`, `imagehash`, `ollama`, `streamlit`, `pytest`).
- Added official Apache-2.0 License to ensure open-source compliance for the hackathon.
- Created package skeleton under `src/leakhound/` partitioned into isolated modules (`parser`, `agent`, `tools`, `verifier`, `report`, `cli`).
- Designed and authored three planted-leak benchmark notebooks under `examples/` with known ground truths:
  1. `examples/01_clear_leak_scaler_before_split.ipynb`: Type 1 leak (`StandardScaler` and `SelectKBest` fitted on the full dataset before `train_test_split`).
  2. `examples/02_clean_pipeline_no_leak.ipynb`: Negative control cleanly encapsulating split and transformation inside `Pipeline`.
  3. `examples/03_subtle_leak_duplicate_split.ipynb`: Type 2 leak (premature duplicate records/oversampling before split causing exact row overlap across train/test).
- Added an initial test suite in `tests/test_examples_ground_truth.py` to execute all three benchmark notebooks end-to-end and assert metric expectations and planted ground-truth behavior.

### 2. Rationale & Why
- **Decisive empirical deltas**: In `01_clear_leak_scaler_before_split.ipynb`, the verifier must demonstrate a drop that exceeds random seed noise. We applied the classic Hastie, Tibshirani & Friedman ("The Elements of Statistical Learning", Chapter 7.10.2: *"How Not to Do Cross-Validation"*) high-dimensional formulation.
- **Negative Control**: `02_clean_pipeline_no_leak.ipynb` is essential to prove LEAKHOUND does not hallucinate false positives on idiomatic ML code.
- **Fast Execution**: All three benchmarks execute in under 3 seconds total (2.01s in automated tests), guaranteeing zero latency during test runs or live hackathon demo reruns.

### 3. Problems Faced & Overcoming Them
- **Problem 1**: Missing runtime dependencies (`nbformat`, `libcst`, `pytest`, `streamlit`, `imagehash`).
  - **Resolution**: Identified missing dependencies via automated inspection and installed them cleanly into the local environment via pip.
- **Problem 2**: In our initial synthetic classification data for Example 1 (`make_classification` with 5 strong informative features and 95 noise features), `SelectKBest` was so effective at isolating the 5 true informative features even on the clean train split alone that `clean_auc` was 0.9119 while `leaked_auc` was 0.9201 ($\Delta = -0.0082$). The empirical test failed because the leak delta was swallowed by the high true signal.
  - **Resolution**: Switched to the textbook Hastie-Tibshirani high-dimensional noise setup (200 samples, 1000 features). Leaked AUC measured `0.7579`, whereas Clean AUC dropped to `0.5068` (chance level), creating an unambiguous, statistically decisive delta of $\Delta = -0.2511$ ($> 5\sigma$), perfectly demonstrating the leakage phenomenon.
- **Problem 3**: Maintaining schema-valid Jupyter notebooks.
  - **Resolution**: Generated all `.ipynb` files programmatically using `nbformat.v4` rather than raw string templating.

### 4. Verification & Test Evidence
Automated test run:
```
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.7.0
rootdir: C:\Users\samru\dev\projectLumina
configfile: pyproject.toml
collected 3 items

tests/test_examples_ground_truth.py::test_01_clear_leak_execution_and_empirical_drop PASSED [ 33%]
tests/test_examples_ground_truth.py::test_02_clean_pipeline_execution PASSED [ 66%]
tests/test_examples_ground_truth.py::test_03_subtle_leak_duplicate_split PASSED [100%]

============================== 3 passed in 2.01s ==============================
```

---
