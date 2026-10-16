# 🔬 REPROVE

> **The LLM proposes, tools verify.**
> An open-source agent that audits machine-learning experiments for **evaluation leakage**, proves each finding with deterministic tools and a **controlled rerun of the experiment with the flaw fixed**, and writes a reproducible audit report.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-green.svg)](pyproject.toml)
[![Hackathon](https://img.shields.io/badge/Hackathon-Best%20Open--Source%20AI%20Project-orange.svg)](docs/DEMO_SCRIPT.md)

---

## 🎯 What it does

Leakage makes a model look better than it is: test-set information sneaks into training, the score is inflated, and the model disappoints in production. REPROVE:

1. **Investigates** a notebook or script with an open-weight LLM (Qwen2.5-Coder via [Ollama](https://ollama.com), runs locally, no API keys). The LLM reads the code and **chooses** which evidence tool to run next.
2. **Gathers evidence** with deterministic tools (AST analysis and row hashing of the real train/test split). No LLM output is ever treated as evidence.
3. **Verifies** by re-running the experiment over 5 seeds, original vs automatically corrected code, and testing whether the corrected version is significantly worse (one-sided paired t-test, p < 0.05).
4. **Reports** each finding as `CONFIRMED`, `NOT_CONFIRMED` or `SUSPECTED`, with code locations, tool evidence, per-seed results, the suggested fix as a diff, and reproduce commands.

### What the statuses mean

| Status | Meaning |
| :--- | :--- |
| `CONFIRMED` | A tool found the leakage pattern **and** fixing it made the score significantly worse across seeds. The reported metric is inflated. |
| `NOT_CONFIRMED` | The pattern is in the code, but fixing it did not change the score beyond seed noise. Still worth fixing. |
| `SUSPECTED` | A tool found the pattern, but no controlled rerun was possible (no automatic fix for that pattern, the code failed, or no metric was found). The reason is always stated. |

The status is assigned by code from the rerun statistics, **never by the LLM**.

---

## 🏗️ How it works

```
 notebook / script ──► Parser (ast + nbformat) ──► experiment graph with exact cell/line spans
                                   │
                                   ▼
        Investigation agent: open-weight LLM (Ollama, JSON-schema constrained output)
        picks a tool ─► reads the result ─► picks the next tool ... (max 8 steps)
        + coverage guard: any core check the agent skipped is run automatically
        + invalid LLM output: retried once, then deterministic fallback (marked in the report)
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
  Deterministic evidence tools                Controlled rerun (verifier)
  • check_fit_before_split  (AST)             • patcher rewrites YOUR code (AST-based)
  • check_split_overlap     (row hashes of    • original vs corrected, 5 seeds,
                             the real split)    separate processes with timeouts
  • trace_test_usage        (AST)             • one-sided paired t-test on per-seed diffs
              └────────────────────┬────────────────────┘
                                   ▼
          report.md / report.html / terminal / Streamlit dashboard
```

---

## 🔬 Scope

| Leak type | Detected by | Automatic fix + rerun |
| :--- | :--- | :--- |
| **Type 1:** scaler / imputer / encoder / PCA / feature selector fitted before `train_test_split` | `check_fit_before_split` (AST) | ✅ split first, re-fit each transformer on the training split only (`fit_transform`, `fit().transform()` and `fit()` + `transform()` forms) |
| **Type 2:** identical rows in train and test | `check_split_overlap`: runs the experiment once and hashes the rows of the actual split | ✅ drop exact duplicate rows before splitting |
| **Type 3:** test set used for training, early stopping (`eval_set=`), or choosing models / hyperparameters in a loop | `trace_test_usage` (AST) | ❌ reported as `SUSPECTED` with evidence (no automatic fix yet) |
| Image folders: near-duplicate images across train/test | `check_image_folders_overlap` (perceptual hash) | ❌ evidence only |
| Target leakage (feature that encodes the label) | not implemented | roadmap |

---

## 📊 Benchmark (small, written by the team)

Four planted-leak notebooks plus two "stress" scripts with unfamiliar variable names. Measured with `reprove benchmark --no-replay` (5 seeds). Numbers come from real runs.

| Case | What it contains | Expected | REPROVE | Rerun result |
| :--- | :--- | :---: | :---: | :--- |
| BM-01 | `StandardScaler` + `SelectKBest` fitted before split (1000 noise features) | CONFIRMED | ✅ CONFIRMED | AUC 0.7705 → 0.4688 (−0.3017), worse on 5/5 seeds, p < 0.0001 |
| BM-02 | Split first, sklearn `Pipeline` | CLEAN | ✅ CLEAN | – |
| BM-03 | Rows duplicated before split (~35% of test rows are copies of training rows) | CONFIRMED | ✅ CONFIRMED | AUC 0.8875 → 0.7730 (−0.1145), worse on 5/5 seeds, p = 0.0098 |
| BM-04 | `max_depth` chosen by test accuracy in a loop | SUSPECTED | ✅ SUSPECTED | no automatic fix (by design) |
| BM-05 | Real but harmless `MinMaxScaler` leak, different variable names | NOT_CONFIRMED | ✅ NOT_CONFIRMED | accuracy 0.9367 → 0.9267 (−0.0100), p = 0.10 |
| BM-06 | Clean code that uses `pd.concat` to join columns | CLEAN | ✅ CLEAN | – |

6/6 statuses match. This is a sanity check on cases we wrote ourselves, not a claim about accuracy on arbitrary code. Exact numbers can differ slightly between scikit-learn versions.

---

## 🚀 Quick start

```bash
git clone https://github.com/anakincodex/projectLumina.git reprove
cd reprove
pip install -e ".[dev]"

# Open-weight model (runs locally)
ollama pull qwen2.5-coder:7b

# Audit with the LLM agent (falls back to the deterministic plan if Ollama is not running)
reprove audit examples/01_clear_leak_scaler_before_split.ipynb

# Write report.md + report.html
reprove audit examples/01_clear_leak_scaler_before_split.ipynb -o audit_reports --no-fail-on-leak

# Dashboard
reprove ui

# Benchmark
reprove benchmark --no-replay
```

Useful flags: `--seeds 0,1,2,3,4` (at least 3), `--timeout 60` (seconds per run), `--model <ollama tag>`, `--verbose`.

### Replay for live demos

```bash
# Before the demo, with Ollama running: record real agent runs
reprove audit examples/01_clear_leak_scaler_before_split.ipynb --record --no-fail-on-leak
# On stage: instant replay of exactly that recorded run
reprove audit examples/01_clear_leak_scaler_before_split.ipynb --replay
```

A replay is keyed by a hash of the exact code, so an edited notebook never gets an old answer. The report says it is a replay and which mode produced it (LLM or fallback).

---

## ⚠️ Limitations (honest list)

- **It executes the code it audits** (separate process, timeout). That is process isolation, not a security sandbox. Only audit code you trust.
- Splits are recognised through `train_test_split`. Automatic fixes need the common `X_train, X_test, y_train, y_test = train_test_split(X, y, ...)` form. Anything else is reported as `SUSPECTED` with the reason.
- The metric is read from variables such as `test_auc` / `test_acc`, or from `sklearn.metrics` calls. Without one, verification is reported as not possible (no numbers are invented).
- Type 3 and image duplicates have detection only. Target leakage is not implemented.
- The benchmark is small and was written by the team.

---

## 🛠️ CI usage

```yaml
# .github/workflows/reprove-audit.yml - CI has no Ollama, so this uses the deterministic fallback
- run: pip install -e .
- run: reprove audit notebooks/train.ipynb   # exits 1 when a leak is CONFIRMED
```

---

## 📂 Project structure

```
src/reprove/
├── parser/        # ast + nbformat -> experiment graph with notebook cell/line mapping
├── agent/         # Ollama agent, shared audit engine (coverage guard), fallback, replay cache
├── tools/         # deterministic evidence tools (no LLM)
├── verifier/      # AST patcher, subprocess runner with metric harness, paired t-test
├── report/        # Markdown + self-contained HTML reports
├── cli/           # Typer CLI (audit, benchmark, ui) + Streamlit app
└── evaluation.py  # benchmark runner
examples/          # 4 planted-leak notebooks + examples/stress/ scripts
tests/             # pytest suite (uses a temporary cache; never calls a real LLM)
docs/              # demo script, project plan
```

## ⚖️ License

Apache License 2.0, see [LICENSE](LICENSE).
