# 🎤 REPROVE: 3-Minute Demo Script

> **Track**: Best Open-Source AI Project
> **Core principle**: *The LLM proposes, tools verify.* Every number on screen comes from an executed run.

---

## ✅ Before you go on stage (do this 30+ minutes early)

1. `ollama serve` is running and `ollama pull qwen2.5-coder:7b` is done. Warm it up: `ollama run qwen2.5-coder:7b "hi"`.
2. Record **real agent runs** for the demo notebooks (each takes about a minute):
   ```bash
   reprove audit examples/01_clear_leak_scaler_before_split.ipynb --record --no-fail-on-leak
   reprove audit examples/02_clean_pipeline_no_leak.ipynb --record
   reprove audit examples/04_test_set_model_selection.ipynb --record
   ```
   Check that each output says `Mode: LLM_OLLAMA`. If it says `FALLBACK_DETERMINISTIC`, read the
   "Fallback reason" line, fix it (usually Ollama not running or model not pulled), and record again.
3. **Write down the numbers from YOUR recorded run** (original mean, corrected mean, mean difference, p-value)
   and use them in the script below. The values in brackets are from our test machine and may differ slightly.
4. Open `reprove ui` in a browser tab, with "Replay recorded run" ticked.
5. Have the backup screen recording ready.

---

## ⏱️ Timeline

| Time | Section | On screen |
| :---: | :--- | :--- |
| 0:00-0:30 | The problem | Notebook 01 claiming AUC ≈ 0.77 |
| 0:30-1:00 | How REPROVE works | Architecture diagram (README) |
| 1:00-2:00 | Live audit of notebook 01 | Terminal or Streamlit, replay of the recorded Qwen run |
| 2:00-2:30 | No false alarms + honesty | Notebook 02 (clean), notebook 04 (SUSPECTED) |
| 2:30-3:00 | Close | Benchmark table, open-source stack |

---

## 🎬 Script

### 1. The problem (0:00-0:30)
> "This notebook reports a test AUC of about 0.77. The features are actually pure random noise. The score is fake,
> because feature selection was fitted on all the data **before** the train/test split. This is called evaluation
> leakage, and it is common in tutorials, Kaggle notebooks and papers. Linters don't catch it, and asking a chatbot
> gives you an opinion, not proof."

### 2. How it works (0:30-1:00)
> "REPROVE is an open-source agent. An open-weight model, Qwen2.5-Coder running locally through Ollama, reads the code
> and decides which checks to run. The checks themselves are deterministic code: AST analysis and hashing the
> rows of the real split. Then REPROVE rewrites the user's own code to fix the flaw, re-runs both versions on five
> seeds, and runs a paired t-test. The LLM never decides whether something is confirmed. The measurement does."

### 3. Live audit (1:00-2:00)
**Driver**: `reprove audit examples/01_clear_leak_scaler_before_split.ipynb --replay --no-fail-on-leak`
(or in the UI: example 01 → Run audit)

> "First line: mode is a replay of a recorded **Qwen** run, so you see exactly what the agent did. In the trace, the
> rows marked `agent` are the model's own choices: it suspected the scaler and SelectKBest in cell 3 and called the
> fit-before-split check. Rows marked `coverage` are checks it didn't call, which REPROVE runs anyway, so an early
> 'finish' can never hide a leak."
>
> "Now the proof. Original AUC **[0.77]**, corrected AUC **[0.47]**, which is chance level. That's a drop of **[0.30]**, worse on
> **[5 of 5]** seeds, p **[< 0.0001]**. So this is CONFIRMED. And here is the fix as a diff of their own code: split first,
> fit on the training split only."

### 4. No false alarms, and honesty (2:00-2:30)
**Driver**: notebook 02, then notebook 04.

> "On a correctly built pipeline: clean, no findings. And here's a case where the test set is used to pick
> max_depth. REPROVE finds it and shows the exact loop, but says SUSPECTED, because it has no automatic fix for this
> type yet and won't claim a number it didn't measure."

### 5. Close (2:30-3:00)
**Driver**: show the benchmark table from the README (or run `reprove benchmark` if the recorded runs are present).

> "On our six benchmark cases, including two scripts with unfamiliar variable names, every status matches the
> planted truth, and a real-but-harmless scaler leak is correctly reported as *not* inflating the score.
> It's Apache-2.0, it runs fully offline on an open-weight model, and every number in the report comes from code
> that actually ran. Thank you!"

---

## 🎯 Likely judge questions

**"Why not just paste the code into ChatGPT?"**
> "A chatbot gives an opinion. We measure the effect by running the code with and without the flaw on five seeds. It
> also runs locally on an open-weight model, so private code never leaves the machine."

**"What does the LLM actually do, then?"**
> "It plans the investigation: it reads the code, forms a hypothesis, picks which tool to run, and decides what to
> check next based on the result. Its output is schema-constrained JSON and is validated. If it's invalid twice, we fall back to a fixed plan and
> the report says so."

**"How do you decide CONFIRMED?"**
> "Code decides, not the model: the corrected version must score worse on average, and a one-sided paired t-test over
> the per-seed differences must give p < 0.05, with at least 3 seeds. Otherwise it's NOT_CONFIRMED or SUSPECTED, with the reason."

**"Is replay cheating?"**
> "Replay re-shows a run we recorded earlier with the real model, keyed by a hash of the exact code. The report
> labels it as a replay and shows which mode produced it. Running it live gives the same tool results. Only the
> model's wording can differ."

**"What doesn't it handle?"**
> "Automatic fixes cover the common sklearn pattern of fitting before `train_test_split`, and duplicate rows. Test-set
> model selection is detected but not auto-fixed, and target leakage is on the roadmap. It also executes the audited
> code, so you should only audit code you trust."

**"Your benchmark is tiny."**
> "Yes, six cases we wrote ourselves. It's a sanity check, not an accuracy claim. That's why we included cases where
> the right answer is 'clean' or 'not confirmed'."
