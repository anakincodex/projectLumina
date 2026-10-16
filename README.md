#  REPROVE

> **The LLM proposes, tools verify.**
> An open-source autonomous agent that audits Machine Learning experiments for evaluation leakage, confirms findings with deterministic evidence tools and controlled reruns, and produces reproducible audit reports.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-green.svg)](pyproject.toml)

---

##  What is REPROVE?

Evaluation leakage quietly ruins ML experiments: inflated validation/test metrics create a false sense of performance that collapses in production.

Unlike generic code scanners that produce hallucinated or noisy warnings, **REPROVE** operates on a strict non-negotiable principle:
**No finding may be labelled `CONFIRMED` unless a deterministic tool or a controlled rerun produced empirical evidence.**

Every claim must be anchored to concrete evidence. If a check cannot run, it is reported as `NOT_VERIFIED` with the exact blocker explained.

---

## 🔬 Supported Leak Types (Scope)

1. **Preprocessing Fitted Before Train/Test Split (Type 1)**: Scalers, imputers, encoders, PCA, or feature selection fitted on the whole dataset prior to splitting.
2. **Train/Test Sample Duplication & Cross-Split Overlap (Type 2)**: Exact or near-duplicate rows (tabular hashing) or identical samples caused by premature oversampling or repeated group records across splits.
3. **Test Set Contamination in Selection/Tuning (Type 3)**: Test set used inside hyperparameter tuning (`GridSearchCV`), early stopping loops, or checkpoint selection.
4. **Target Leakage (Type 4 - Detection Only)**: Features with unrealistically high predictive association with the target label.

---

##  Architecture

```
                               ┌────────────────────────────────┐
                               │   ML Notebook or Script (.py)   │
                               └──────────────┬─────────────────┘
                                              │
                                              ▼
                               ┌────────────────────────────────┐
                               │     AST Parser & Tracer        │
                               │   (Extracts Experiment Graph)  │
                               └──────────────┬─────────────────┘
                                              │
                                              ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                              Investigation Agent                                       │
 │           (Open-weight LLM e.g. Qwen2.5-Coder via Ollama / Replay Cache)               │
 │                                                                                        │
 │   1. Proposes Leak Hypotheses                                                          │
 │   2. Orchestrates Bounded Deterministic Tool Execution (<= 8 steps)                    │
 └─────────────────┬───────────────────────────────────────────────────┬──────────────────┘
                   │                                                   │
                   ▼                                                   ▼
 ┌───────────────────────────────────┐               ┌────────────────────────────────────┐
 │       Deterministic Tools         │               │     Controlled Verifier Engine     │
 │  • check_fit_before_split         │               │  • Patches pipeline (AST rewrite)  │
 │  • check_split_overlap            │               │  • Sandboxed rerun (Orig vs Patch) │
 │  • trace_test_usage               │               │  • Multi-seed delta & significance │
 └─────────────────┬─────────────────┘               └─────────────────┬──────────────────┘
                   │                                                   │
                   └─────────────────────┬─────────────────────────────┘
                                         ▼
                         ┌───────────────────────────────┐
                         │      Audit Report (report.md) │
                         │  • CONFIRMED / SUSPECTED      │
                         │  • AST Code Spans & Diffs     │
                         │  • Empirical Metric Deltas    │
                         │  • Reproducibility Recipe     │
                         └───────────────────────────────┘
```

---

##  Quick Start

### Installation

```bash
git clone https://github.com/anakincodex/projectLumina.git REPROVE
cd REPROVE
pip install -e .
```

### Run an Audit

```bash
REPROVE audit examples/01_clear_leak_scaler_before_split.ipynb --model qwen2.5-coder:7b
```

Or launch the interactive demo UI:

```bash
streamlit run src/REPROVE/cli/streamlit_app.py
```

---

##  Project Structure

```
REPROVE/
├── src/REPROVE/
│   ├── parser/         # AST (ast/libcst) + nbformat -> experiment graph
│   ├── agent/          # Ollama / Qwen2.5-Coder investigator + replay cache
│   ├── tools/          # Deterministic evidence checkers (no LLM)
│   ├── verifier/       # Controlled multi-seed runner & AST patcher
│   ├── report/         # Markdown & HTML audit report generator
│   └── cli/            # Typer CLI & Streamlit demo application
├── examples/           # Ground truth planted-leak benchmark notebooks
├── tests/              # Test suite (pytest)
├── docs/               # Architecture diagrams and demo script
├── DEVELOPMENT_LOG.md  # Detailed log of decisions, issues, & resolutions
├── LICENSE             # Apache-2.0
├── pyproject.toml
└── README.md
```

---

##  License

REPROVE is distributed under the [Apache License 2.0](LICENSE).
