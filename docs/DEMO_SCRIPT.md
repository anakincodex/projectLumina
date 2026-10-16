# 🎤 REPROVE: 3-Minute Live Hackathon Demo Script

> **Hackathon Track**: Best Open-Source AI Project  
> **Core Principle**: *The LLM proposes, deterministic tools verify.* Zero fabricated numbers.  
> **Presenter Roles**: Team of 3-4 (Speaker / Live Driver / Tech Lead / Slide & UI Presenter).

---

## ⏱️ Timeline Overview

| Time | Section | Key Visual / Action | Speaker Goal |
| :---: | :--- | :--- | :--- |
| **0:00 - 0:40** | The Silent Crisis in ML | Slide: Kaggle & Academic Papers with Leaks | Hook judges with the billion-dollar evaluation leak problem |
| **0:40 - 1:10** | REPROVE Core Principle | Architecture Diagram | Explain why open-weight LLMs + deterministic verification beat naive chat prompts |
| **1:10 - 2:00** | Live Demo (Terminal & UI) | `reprove audit` & Streamlit Dashboard | Show live audit on textbook Hastie-Tibshirani benchmark ($\Delta = -0.2026$) |
| **2:00 - 2:30** | Counterfactual Proof & Control | Code Diff & Clean Benchmark Rerun | Prove empirical rerun drop and zero false-positive negative control |
| **2:30 - 3:00** | Benchmark Matrix & Conclusion | `reprove benchmark` (100% Precision/Recall) | Close with open-source impact, reproducibility, and Apache-2.0 license |

---

## 🎬 Section-by-Section Script & Commands

### 1. The Hook: The Silent Crisis in ML (0:00 – 0:40)
**Speaker**:  
> *"Judges, imagine training a medical diagnosis model that boasts a 95% test accuracy. You deploy it to a hospital, and patients start getting misdiagnosed. What went wrong?*  
> *It wasn’t the model architecture. It was **evaluation leakage**—a subtle bug where test set statistics leaked into feature selection before the train-test split.*  
> *Today, data scientists rely on two flawed extremes: static regex linters that spit out dozens of false alarms, or cloud LLMs that hallucinate line numbers and invent fake numbers. Neither can tell you: **does this bug actually break my model?**"*

---

### 2. Enter REPROVE (0:40 – 1:10)
**Speaker**:  
> *"We built **REPROVE**—an open-source autonomous agent that audits ML experiments and **empirically proves** whether evaluation leakage occurred.*  
> *Our core architectural principle is non-negotiable: **The LLM proposes, deterministic tools verify.***  
> *An open-weight model—Qwen2.5-Coder running locally via Ollama—investigates the code graph and formulates hypotheses. But the model is never allowed to confirm a finding. Confirmation requires a deterministic tool check and a controlled, multi-seed counterfactual rerun."*

---

### 3. Live Demonstration: Terminal Audit (1:10 – 1:40)
**Driver Action**:  
In the terminal, run:
```bash
reprove audit examples/01_clear_leak_scaler_before_split.ipynb --replay
```

**Speaker**:  
> *"Watch REPROVE audit a textbook benchmark from Hastie & Tibshirani’s Elements of Statistical Learning.*  
> *Look at the terminal output:*  
> *1. **Investigation Trajectory**: The open-weight agent inspected the experiment graph, identified a suspicious `SelectKBest` and `StandardScaler` call, and invoked our deterministic `check_fit_before_split` tool.*  
> *2. **Deterministic Evidence**: The tool traced the AST dataflow and proved that features were fitted on all 200 samples before splitting.*  
> *3. **Controlled Counterfactual Verification**: REPROVE automatically patched the AST to isolate fitting to the training split, and reran the experiment across random seeds.*  
> *Look at the numbers on screen: The original reported score was **0.7410**. The clean patched score dropped to **0.5384**—an empirical inflation of **Δ = -0.2026** against a random noise margin of ±0.07. That is empirical mathematical proof of leakage."*

---

### 4. Interactive UI & Zero False-Positive Control (1:40 – 2:20)
**Driver Action**:  
Switch to the browser showing the Streamlit interface (or run `reprove ui`):
```bash
reprove ui
```
- Click into **01 - Type 1: Clear Preprocessing Leak**.
- Show the **🚨 CONFIRMED LEAK** banner.
- Click the **🛠️ Counterfactual Patch** tab to show the syntax-highlighted AST diff.
- Switch the dropdown to **02 - Negative Control: Clean Encapsulated Pipeline** and click **Run Empirical Audit**.

**Speaker**:  
> *"In our Streamlit live dashboard, you can see the complete investigation audit trail.*  
> *Under the Counterfactual Patch tab, REPROVE generates the exact unified diff that engineers can merge to fix their code.*  
> *Now watch what happens when we audit a properly encapsulated `Pipeline` in Benchmark 02.*  
> *REPROVE runs all deterministic checks, finds zero leakage, and issues an immediate **✅ CLEAN EXPERIMENT PIPELINE** verdict.*  
> *No false alarms. No noisy warnings. 100% specificity."*

---

### 5. Benchmark Matrix & Conclusion (2:20 – 3:00)
**Driver Action**:  
In terminal, run:
```bash
reprove benchmark --replay
```

**Speaker**:  
> *"Across our ground-truth benchmark suite, REPROVE delivers **100% precision, 100% recall**, and runs in **30 milliseconds** using our replay cache.*  
> *Every single report is published as reproducible Markdown and self-contained HTML that data science teams can integrate into GitHub Actions to fail PRs with `--fail-on-leak`.*  
> *REPROVE is 100% open-source under the Apache-2.0 license, powered by open-weight AI, and ready for production.*  
> *Thank you, and we welcome your questions!"*

---

## 🎯 Anticipated Judge Q&A & Bulletproof Responses

### Q1: *"Why can't I just paste my code into ChatGPT or Claude?"*
**Response**:  
> *"Three reasons: First, cloud LLMs hallucinate metrics and line numbers. Second, they have zero execution environment—they cannot execute the code to tell you whether the bug actually changed the test score by 1% or 25%. Third, enterprise ML code often contains proprietary IP that cannot be sent to third-party cloud APIs. REPROVE runs locally with open-weight models (Qwen2.5-Coder) and grounds every claim in real, executed counterfactual reruns."*

### Q2: *"How is this different from flake8, Ruff, or SonarQube?"*
**Response**:  
> *"Linters rely on static AST pattern matching. They cannot trace dynamic dataflow across notebook cells, cannot hash multi-table row overlaps across splits, and cannot perform statistical significance tests on metric shifts. A linter flags syntax; REPROVE measures empirical evaluation inflation."*

### Q3: *"What happens if Ollama is not installed or the user is offline on a laptop?"*
**Response**:  
> *"REPROVE has a dual fallback guarantee: First, we have an instant Replay Cache that replays verified audits in under 40 milliseconds with zero network calls. Second, if an unknown script is audited without Ollama, REPROVE automatically fails over to a deterministic heuristic plan that executes the evidence tools and verifier engine directly without dropping a single verification check."*

### Q4: *"How do you guarantee the agent doesn't confirm a fake leak?"*
**Response**:  
> *"Through our architectural invariant: In REPROVE's data model, `FindingStatus.CONFIRMED` is strictly unreachable by the LLM. It can only be set by the verifier engine when both conditions are satisfied: (1) a deterministic evidence tool produced a verified `EvidenceObject`, and (2) the empirical delta $|Δ|$ strictly exceeds twice the multi-seed noise standard deviation ($2σ$)."*
