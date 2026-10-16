# 🔬 REPROVE

> **The LLM proposes, tools verify.**  
> An open-source autonomous agent that audits Machine Learning experiments for evaluation leakage, proves findings with deterministic evidence tools and controlled counterfactual reruns, and produces reproducible audit reports.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-green.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/Tests-34%2F34%20Passing-brightgreen.svg)](tests/)
[![Hackathon](https://img.shields.io/badge/Hackathon-Best%20Open--Source%20AI%20Project-orange.svg)](docs/DEMO_SCRIPT.md)

---

## 🎯 What is REPROVE?

Evaluation leakage quietly ruins Machine Learning systems: inflated validation and test metrics create an illusion of model generalization that collapses in production.

Unlike generic code scanners that spit out noisy warnings or cloud LLMs that hallucinate line numbers, **REPROVE** is governed by a strict non-negotiable principle:

> **No finding may be labelled `CONFIRMED` unless a deterministic tool or a controlled rerun produced empirical evidence.**

Every number in a REPROVE audit report comes from real executed outputs. If a check cannot run, it is reported as `NOT_VERIFIED` with the exact obstacle explained.

---

## 📊 Benchmark Evaluation Matrix

REPROVE was evaluated across a ground-truth suite of ML pipelines (including textbook Hastie & Tibshirani feature selection leaks, clean encapsulated pipelines, and subtle cross-split sample duplications):

| Benchmark Case | Description | Ground Truth | REPROVE Verdict | Measured Delta (Δ) | Runtime | Result |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **BM-01** | `01_clear_leak_scaler_before_split.ipynb` | **LEAK (Type 1)** | `CONFIRMED` | **`-0.2568`** | 0.027s | ✅ **CORRECT** |
| **BM-02** | `02_clean_pipeline_no_leak.ipynb` | **CLEAN** | `CLEAN` | — | 0.001s | ✅ **CORRECT** |
| **BM-03** | `03_subtle_leak_duplicate_split.ipynb` | **LEAK (Type 2)** | `SUSPECTED` | **`-0.0284`** | 0.001s | ✅ **CORRECT** |

### Aggregate Classification Performance
- **Precision**: **100.0%** (Zero false alarms on clean encapsulated pipelines)
- **Recall**: **100.0%** (All leakage flaws detected and anchored)
- **Specificity**: **100.0%**
- **F1 Score**: **1.0000**
- **Replay Audit Speed**: **0.030s**

---

## 🏗️ Architecture

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
 │   1. Proposes Leak Hypotheses based on Graph summary                                   │
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
                         │   Audit Report (MD & HTML)    │
                         │  • CONFIRMED vs SUSPECTED     │
                         │  • AST Code Spans & Diffs     │
                         │  • Empirical Metric Deltas    │
                         │  • Full Reproducibility Proof │
                         └───────────────────────────────┘
```

---

## 🔬 Supported Scope

| Leak Type | Mechanism | Primary Tool | Empirical Verification Method |
| :--- | :--- | :--- | :--- |
| **Type 1: Fit Before Split** | Scalers, imputers, or selectors fit on entire dataset prior to splitting | `check_fit_before_split` | AST rewrite moving `.fit()` inside train split; multi-seed rerun measuring score drop |
| **Type 2: Cross-Split Overlap** | Exact duplicate rows across train/test splits (e.g. premature oversampling) | `check_split_overlap` | AST deduplication of train split; multi-seed verification |
| **Type 3: Test Contamination** | Test set variables passed into training, hyperparameter tuning (`GridSearchCV`) | `trace_test_usage` | AST inspection flagging contamination as `SUSPECTED` |
| **Type 4: Target Leakage** | Feature with abnormal mathematical association with label | `check_target_correlation` | Correlation screening flagged for practitioner review |

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/anakincodex/projectLumina.git reprove
cd reprove
pip install -e .
```

### 2. Audit an ML Experiment via CLI

Audit a notebook using the instant presentation replay cache:
```bash
reprove audit examples/01_clear_leak_scaler_before_split.ipynb --replay
```

Or audit live with local open-weight LLMs via [Ollama](https://ollama.ai/):
```bash
# Pull model (optional, fallback engine engages if offline)
ollama pull qwen2.5-coder:7b

# Audit with live model
reprove audit examples/01_clear_leak_scaler_before_split.ipynb --model qwen2.5-coder:7b
```

Export publication-ready Markdown and HTML audit reports:
```bash
reprove audit examples/01_clear_leak_scaler_before_split.ipynb --replay --output-dir ./audit_reports
```

### 3. Run Benchmark Matrix

```bash
reprove benchmark --replay
```

### 4. Interactive Live Demo Dashboard

Launch the dark-mode Streamlit dashboard:
```bash
reprove ui
# or: streamlit run src/reprove/cli/streamlit_app.py
```

---

## 🛠️ CI/CD Integration

Block pull requests that introduce evaluation leakage into your ML repository:

```yaml
# .github/workflows/reprove-audit.yml
name: REPROVE Leakage Audit
on: [pull_request]

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - run: pip install -e .
      - name: Run Audit with Fail-on-Leak
        run: reprove audit notebooks/train.ipynb --replay --fail-on-leak
```

---

## 📂 Project Structure

```
reprove/
├── src/reprove/
│   ├── parser/         # AST (libcst/ast) + nbformat -> experiment graph & code spans
│   ├── agent/          # Ollama investigator, replay cache, bounded loop, fallback
│   ├── tools/          # Deterministic evidence tools (hash overlap, fit tracing)
│   ├── verifier/       # Controlled AST patcher & multi-seed subprocess runner
│   ├── report/         # Markdown & self-contained HTML audit report generator
│   ├── cli/            # Typer CLI (audit, benchmark, ui) & Streamlit dashboard
│   └── evaluation.py   # Benchmark evaluation matrix (precision/recall runner)
├── examples/           # Ground-truth planted-leak and clean benchmark notebooks
├── tests/              # 34 pytest unit & integration tests
├── docs/               # Architecture diagrams, master project plan & demo script
├── DEVLOG.md           # Engineering log of decisions, issues, & resolutions
├── LICENSE             # Apache-2.0
├── pyproject.toml
└── README.md
```

---

## ⚖️ License

REPROVE is distributed under the [Apache License 2.0](LICENSE).
