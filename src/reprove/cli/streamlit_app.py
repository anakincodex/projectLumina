"""Streamlit demo UI for REPROVE.  Run with:  reprove ui   (or: streamlit run src/reprove/cli/streamlit_app.py)"""
import os
import tempfile

import streamlit as st

from reprove.agent.investigator import DEFAULT_MODEL, run_investigation
from reprove.agent.schemas import InvestigationReport
from reprove.report.generator import generate_report
from reprove.report.markdown import location, mode_description
from reprove.verifier import DEFAULT_SEEDS
from reprove.verifier.models import NOISE_RULE, format_p

st.set_page_config(page_title="REPROVE | ML Leakage Auditor", page_icon="🛡️", layout="wide")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _example(rel: str) -> str:
    """Resolve example paths whether Streamlit was started from the repo root or elsewhere."""
    return rel if os.path.exists(rel) else os.path.join(REPO_ROOT, rel)


BENCHMARKS = {
    "01 - Type 1: Clear Preprocessing Leak (Hastie-Tibshirani)": "examples/01_clear_leak_scaler_before_split.ipynb",
    "02 - Negative Control: Clean Encapsulated Pipeline": "examples/02_clean_pipeline_no_leak.ipynb",
    "03 - Type 2: Cross-Split Sample Duplication": "examples/03_subtle_leak_duplicate_split.ipynb",
    "04 - Type 3: Hyperparameter Chosen on Test Set": "examples/04_test_set_model_selection.ipynb",
    "05 - Stress: Real but Harmless Scaler Leak": "examples/stress/05_minmax_scaler_leak.py",
    "06 - Stress: Clean Code Using pd.concat": "examples/stress/06_concat_columns_clean.py",
}

st.title("🛡️ REPROVE")
st.subheader("Open-source agent that audits ML experiments for evaluation leakage")
st.caption("**The LLM proposes, tools verify.** A finding is CONFIRMED only when a controlled rerun with the flaw fixed scores significantly worse.")

# ---------------------------------------------------------------- sidebar
st.sidebar.header("🎯 Experiment to audit")
source = st.sidebar.radio("Source", ["Benchmark examples", "Upload a script / notebook"])
target_file = None
if source == "Benchmark examples":
    target_file = _example(BENCHMARKS[st.sidebar.selectbox("Example", list(BENCHMARKS.keys()))])
else:
    uploaded = st.sidebar.file_uploader("Upload .py or .ipynb", type=["py", "ipynb"])
    if uploaded is not None:
        suffix = ".ipynb" if uploaded.name.endswith(".ipynb") else ".py"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            target_file = tmp.name
        st.sidebar.caption("Uploaded code runs on this machine during the controlled rerun - only upload code you trust.")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Agent & verifier")
model_name = st.sidebar.text_input("Ollama model", value=DEFAULT_MODEL)
use_replay = st.sidebar.checkbox(
    "⚡ Replay recorded run (if one exists)",
    value=True,
    help="Shows a previously recorded audit of this exact code instantly. Runs live if none was recorded.",
)
record = st.sidebar.checkbox("💾 Record this run for replay", value=False)
seeds_input = st.sidebar.text_input("Seeds (>= 3)", value=", ".join(str(s) for s in DEFAULT_SEEDS))
try:
    seeds = [int(s.strip()) for s in seeds_input.split(",") if s.strip()]
except ValueError:
    st.sidebar.error("Seeds must be integers; using the defaults.")
    seeds = list(DEFAULT_SEEDS)

if st.sidebar.button("🔍 Run audit", type="primary", use_container_width=True) and target_file:
    with st.spinner("Auditing: the agent picks checks, then the experiment is re-run with each flaw fixed..."):
        st.session_state["report"] = run_investigation(
            file_path=target_file, model_name=model_name, use_replay=use_replay, seeds=seeds, record=record
        )

# ---------------------------------------------------------------- results
ORIGIN_LABEL = {"agent": "🤖 chosen by the LLM", "coverage": "🛡️ coverage (agent skipped it)", "fallback": "📋 fixed plan"}


def render_report(report: InvestigationReport) -> None:
    md_text, html_text = generate_report(report)

    st.markdown(f"**Mode:** `{report.execution_mode}` - {mode_description(report)}")
    if report.fallback_reason:
        st.warning(f"Fallback plan used: {report.fallback_reason}")

    verdict = report.verdict
    first_ran = next((v for v in report.verification_results if v.verification_ran), None)
    if verdict == "CONFIRMED":
        delta = f" (mean difference {first_ran.mean_delta:+.4f} in `{first_ran.primary_metric_name}`)" if first_ran else ""
        st.error(f"### 🚨 VERDICT: {len(report.confirmed_evidence)} CONFIRMED LEAK(S){delta}\nFixing the flaw made the score significantly worse - the reported metric is inflated.")
    elif verdict == "SUSPECTED":
        st.warning(f"### ⚠️ VERDICT: {len(report.suspected_evidence)} SUSPECTED LEAK(S) - NOT VERIFIED\nA tool found the pattern, but an automatic rerun was not possible (see the reason below).")
    elif verdict == "NOT_CONFIRMED":
        st.info("### 🔵 VERDICT: LEAKAGE PATTERN PRESENT, NO MEASURABLE INFLATION\nThe pattern exists, but fixing it did not change the score beyond seed noise.")
    elif verdict == "INCONCLUSIVE":
        st.info("### ⚪ VERDICT: INCONCLUSIVE\nSome checks could not run on this code.")
    else:
        st.success("### ✅ VERDICT: CLEAN\nNo leakage found by the fit-before-split, split-overlap and test-usage checks.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Confirmed", len(report.confirmed_evidence))
    c2.metric("Suspected", len(report.suspected_evidence))
    c3.metric("No measurable effect", len(report.not_confirmed_evidence))
    c4.metric("Clean checks", len(report.clean_checks))
    c5.metric("Steps", len(report.steps))
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["🕵️ Investigation trace", "📊 Controlled rerun", "🛠️ Suggested fix", "📄 Full report"])

    with tab1:
        for step in report.steps:
            with st.expander(f"Step {step.step_number}: `{step.tool_name}` -> {step.result or '-'}  ({ORIGIN_LABEL.get(step.origin, step.origin)})",
                             expanded=step.step_number == 1):
                st.markdown(f"**Reasoning:** {step.thought}")
                if step.tool_args:
                    st.markdown(f"**Arguments:** `{step.tool_args}`")
                st.markdown("**Tool output:**")
                st.code(step.observation, language="text")

    with tab2:
        findings = (
            [("CONFIRMED", ev) for ev in report.confirmed_evidence]
            + [("SUSPECTED", ev) for ev in report.suspected_evidence]
            + [("NOT_CONFIRMED", ev) for ev in report.not_confirmed_evidence]
        )
        if not findings:
            st.info("No leakage pattern was found, so no rerun was needed.")
        for label, ev in findings:
            vr = report.verification_for(ev.evidence_id)
            st.markdown(f"#### {ev.metrics.get('leak_type', 'LEAK')} - `{label}`")
            st.caption(f"Location: {location(ev)}")
            st.markdown(ev.summary)
            if vr and vr.verification_ran:
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("Original (as written)", f"{vr.original_mean:.4f}", f"± {vr.original_std:.4f}", delta_color="off")
                v2.metric("Corrected", f"{vr.patched_mean:.4f}", f"± {vr.patched_std:.4f}", delta_color="off")
                v3.metric("Mean difference", f"{vr.mean_delta:+.4f}")
                v4.metric("p-value (paired t-test)", format_p(vr.p_value))
                st.table([
                    {"Seed": s.seed, "Original": f"{s.original_metric:.4f}", "Corrected": f"{s.patched_metric:.4f}", "Difference": f"{s.delta:+.4f}"}
                    for s in vr.seed_runs
                ])
                st.caption(NOISE_RULE)
            if vr:
                st.markdown(f"**Verdict:** {vr.reason}")
            st.markdown("---")

    with tab3:
        diffs = [v for v in report.verification_results if v.diff]
        if diffs:
            for vr in diffs:
                st.markdown(f"Fix for evidence `{vr.evidence_id}` (generated from your own code):")
                st.code(vr.diff, language="diff")
        else:
            st.info("No automatic fix was generated for this audit.")
        for ev in report.suspected_evidence + report.not_confirmed_evidence:
            if ev.suggested_fix:
                st.markdown(f"**How to fix `{ev.metrics.get('leak_type', '')}`:** {ev.suggested_fix}")

    with tab4:
        st.download_button("📥 Download report.md", data=md_text, file_name="report.md", mime="text/markdown")
        st.download_button("📥 Download report.html", data=html_text, file_name="report.html", mime="text/html")
        st.markdown(md_text)


if "report" in st.session_state:
    render_report(st.session_state["report"])
else:
    st.info("Pick an example (or upload your own) and press **Run audit**.")
