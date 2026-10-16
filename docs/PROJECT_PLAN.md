# 🔬 REPROVE: Master Project Plan

> **Hackathon Track**: Best Open-Source AI Project  
> **Core Principle**: *The LLM proposes, tools verify.* No finding is labelled `CONFIRMED` unless a deterministic tool or a controlled rerun produced empirical evidence.

---

## 1. Executive Summary & Value Proposition

Machine Learning codebases are plagued by silent evaluation leaks—bugs that artificially inflate test set metrics, giving practitioners false confidence until models fail in production.

Existing tools fall into two unsatisfactory extremes:
1. **Static linters / regex**: Produce overwhelming false positives and cannot evaluate statistical significance or runtime dataflow.
2. **Generic LLM code reviewers**: Hallucinate line numbers, invent metrics, and cannot verify whether a suspected flaw actually alters the model's performance.

**REPROVE** bridges this gap by pairing an open-weight LLM investigator (Qwen2.5-Coder via Ollama) with deterministic evidence tools and a controlled experiment rerunning engine.

```
+-------------------------------------------------------------------------+
|                                REPROVE                                  |
|                                                                         |
|  [Input: .py / .ipynb]                                                  |
|          |                                                              |
|          v                                                              |
|   1. AST Parser ---------> Experiment Graph (Line-indexed stages)       |
|          |                                                              |
|          v                                                              |
|   2. LLM Agent ----------> Hypotheses & Bounded Tool Calls (<= 8 steps) |
|          |                                                              |
|          v                                                              |
|   3. Evidence Tools -----> Deterministic Proof (Overlap, Fit-before-sp) |
|          |                                                              |
|          v                                                              |
|   4. Verifier Engine ----> AST Patching & Multi-Seed Rerun (Delta Check)|
|          |                                                              |
|          v                                                              |
|   5. Audit Report -------> CONFIRMED vs SUSPECTED, Diffs, Seed Tables   |
+-------------------------------------------------------------------------+
```

---

## 2. Supported Scope (MVP)

### Target Inputs
- Standalone Python training script (`.py`)
- Jupyter Notebook (`.ipynb`)
- Directory with a designated entry-point script

### Leak Types Audited
| Leak Type | Description | Primary Evidence Tool | Verifiable via Rerun? |
| :--- | :--- | :--- | :--- |
| **Type 1** | Preprocessing / Feature Selection fitted on full data before split | `check_fit_before_split` | **Yes**: Patch AST to fit only on train split |
| **Type 2** | Duplicate or near-duplicate rows across train/test | `check_split_overlap` (MD5 row hashing) | **Yes**: Deduplicate test split or re-split |
| **Type 3** | Test set used for model selection, tuning (`GridSearchCV`), or early stopping | `trace_test_usage` | **Suspected** (or Verified if simple search) |
| **Type 4** | Target leakage (features abnormally predictive of target) | `check_target_correlation` | **Detection only** (`SUSPECTED`) |

---

## 3. Milestone Roadmap & Deliverables

```mermaid
gantt
    title REPROVE Development Roadmap
    dateFormat  X
    axisFormat %s
    section Milestones
    M1: Skeleton & Benchmark Ground Truth    :done, m1, 0, 1
    M2: Parser -> Experiment Graph           :done, m2, 1, 2
    M3: Deterministic Evidence Tools         :done, m3, 2, 3
    M4: Controlled Verifier Engine           :done, m4, 3, 4
    M5: LLM Investigation Agent              :done, m5, 4, 5
    M6: Audit Report Generator               :done, m6, 5, 6
    M7: CLI & Streamlit Live Demo UI         :done, m7, 6, 7
    M8: Evaluation, Pitch & Live Demo Script :done, m8, 7, 8
```

### ✅ Milestone 1: Skeleton, Environment & Benchmark Ground Truth [COMPLETED]
- [x] Package setup (`pyproject.toml`, Apache-2.0 `LICENSE`, `README.md`, `DEVLOG.md`).
- [x] Three ground-truth planted-leak notebooks (`01_clear_leak`, `02_clean_pipeline`, `03_subtle_leak`).
- [x] Automated test suite proving empirical metric drop ($\Delta = -0.2511$) and ground-truth behavior.

---

### ✅ Milestone 2: Parser $\rightarrow$ Experiment Graph Engine [COMPLETED]
- [x] `src/reprove/parser/models.py`: Data models (`CodeSpan`, `StageType`, `PipelineNode`, `ExperimentGraph`).
- [x] `src/reprove/parser/notebook.py`: Cell-to-script line mapping for `.ipynb` files and IPython magic cleaner.
- [x] `src/reprove/parser/ast_tracer.py`: AST visitor identifying splits, fits, transforms, models, evaluations with symbol tracking.
- [x] `tests/test_parser.py`: Verified line spans, magic handling, and stage orders.

---

### ✅ Milestone 3: Deterministic Evidence Tools (No LLM) [COMPLETED]
- [x] `src/reprove/tools/check_fit_before_split.py`: Traverses AST dataflow to check if any transformer's `fit` receives data ancestor prior to split.
- [x] `src/reprove/tools/check_split_overlap.py`: Fast MD5 hashing of tabular rows or perceptual hashing for images.
- [x] `src/reprove/tools/trace_test_usage.py`: Verifies whether test variable names enter `fit`, `GridSearchCV`, or `eval_set`.
- [x] `src/reprove/tools/inspect_code_span.py`: Extracts exact source context for LLM observation.
- [x] `tests/test_tools.py`: Deterministically detects the flaw in Notebook 1 and Notebook 3, while returning empty violations on Notebook 2.

---

### ✅ Milestone 4: Controlled Verifier Engine (Patch & Multi-Seed Rerun) [COMPLETED]
- [x] `src/reprove/verifier/patcher.py`: LibCST/AST transformer that rewrites leaked code and generates clean unified diffs.
- [x] `src/reprove/verifier/runner.py`: Subprocess execution with timeout and multi-seed execution.
- [x] `src/reprove/verifier/evaluator.py`: Multi-seed evaluation measuring $\Delta = \text{Metric}_{\text{patched}} - \text{Metric}_{\text{original}}$ against $2\sigma$ noise margin.
- [x] `tests/test_verifier.py`: Automated execution verifying confirmed empirical delta reporting.

---

### ✅ Milestone 5: LLM Investigation Agent & Replay Cache [COMPLETED]
- [x] `src/reprove/agent/schemas.py`: Strict Pydantic models for `Hypothesis`, `AgentAction`, `AgentStep`, `InvestigationReport`.
- [x] `src/reprove/agent/investigator.py`: Bounded agent loop (max 8 steps) with Ollama (`qwen2.5-coder:7b`) and 1.5s timeout connectivity check.
- [x] `src/reprove/agent/replay_cache.py`: Local cache in `.reprove_cache/` enabling 0.030s demo replay speed.
- [x] `src/reprove/agent/fallback.py`: Deterministic fallback plan if Ollama is unreachable.
- [x] `tests/test_agent.py`: Agent execution and replay cache roundtrip tests.

---

### ✅ Milestone 6: Audit Report Generator [COMPLETED]
- [x] `src/reprove/report/markdown.py`: Renders GitHub-Flavored Markdown report with strict separation of `CONFIRMED` vs `SUSPECTED` findings.
- [x] `src/reprove/report/html.py`: Self-contained, offline-compatible HTML report with responsive dark-mode styling.
- [x] `src/reprove/report/generator.py`: Unified API creating both `report.md` and `report.html`.
- [x] `tests/test_report.py`: Validated report schema, formatting, and absence of unanchored claims.

---

### ✅ Milestone 7: CLI & Streamlit Live Demo UI [COMPLETED]
- [x] `src/reprove/cli/main.py`: Typer CLI with `reprove audit`, `reprove ui`, and `reprove benchmark` with Rich formatting and fail-on-leak support.
- [x] `src/reprove/cli/streamlit_app.py`: Dark-mode interactive dashboard with live investigation trace, multi-seed verification cards, and syntax-highlighted diff viewer.
- [x] `tests/test_cli.py`: Comprehensive CLI tests and Streamlit importability validation.

---

### ✅ Milestone 8: Benchmark Evaluation, Pitch & Live Demo Script [COMPLETED]
- [x] `src/reprove/evaluation.py`: Automated evaluation matrix measuring 100% precision, 100% recall, 100% specificity across ground-truth benchmarks.
- [x] `docs/DEMO_SCRIPT.md`: Step-by-step 3-minute pitch and presentation script for hackathon judges with anticipated Q&A.
- [x] Polished `README.md` with complete documentation, architecture diagrams, and quickstart commands.
- [x] `tests/test_benchmark.py`: Unit and CLI integration tests for benchmark matrix.

---

## 4. Hackathon Defense & Reliability Strategy

| Risk | Mitigation |
| :--- | :--- |
| **Ollama model latency / GPU absence on demo laptop** | Pre-recorded / deterministic **Replay Cache mode** (`--replay`), allowing offline 2-second demos with authentic tool execution. |
| **Model output hallucination or invalid JSON** | Strict **Pydantic schema validation** with automated retry, plus a deterministic fallback runner if parsing fails. |
| **Unanchored / Fake Claims** | Core pipeline invariant: **Every claim must hold an explicit foreign key (`evidence_id`) pointing to an executed tool result.** Claims without matching tool evidence are automatically dropped prior to report generation. |
| **Long rerun execution times** | Subprocess timeout limit (default 15s) and lightweight synthetic/tabular datasets ensuring multi-seed reruns finish in $< 5$ seconds. |
