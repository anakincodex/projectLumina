"""Streamlit Live Demo UI for REPROVE."""
import os
import tempfile
from typing import List

import streamlit as st

from reprove.agent.fallback import DEFAULT_SEEDS
from reprove.agent.investigator import run_investigation
from reprove.agent.schemas import InvestigationReport
from reprove.report.generator import generate_report
from reprove.verifier import FindingStatus

# Page Config
st.set_page_config(
    page_title="REPROVE | ML Evaluation Leakage Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark-mode modern dashboard)
st.markdown(
    """
    <style>
    .main {
        background-color: #0d1117;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .leak-badge {
        background-color: #ffebe9;
        color: #cf222e;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .clean-badge {
        background-color: #dafbe1;
        color: #1a7f37;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title & Header
st.title("🛡️ REPROVE")
st.subheader("Autonomous Open-Source Agent for Auditing ML Evaluation Leakage")
st.caption(
    "Principle: **The LLM proposes, tools verify.** No finding is confirmed without empirical counterfactual measurement."
)

# Benchmark paths
BENCHMARKS = {
    "01 - Type 1: Clear Preprocessing Leak (Hastie-Tibshirani)": "examples/01_clear_leak_scaler_before_split.ipynb",
    "02 - Negative Control: Clean Encapsulated Pipeline": "examples/02_clean_pipeline_no_leak.ipynb",
    "03 - Type 2: Cross-Split Sample Duplication": "examples/03_subtle_leak_duplicate_split.ipynb",
}

# Sidebar
st.sidebar.header("🎯 Target ML Experiment")
source_choice = st.sidebar.radio(
    "Select Source",
    ["Benchmark Notebooks", "Upload Custom Script/Notebook"],
)

target_file = None
temp_file_path = None

if source_choice == "Benchmark Notebooks":
    benchmark_name = st.sidebar.selectbox("Choose Benchmark", list(BENCHMARKS.keys()))
    target_file = BENCHMARKS[benchmark_name]
else:
    uploaded = st.sidebar.file_uploader("Upload .py or .ipynb", type=["py", "ipynb"])
    if uploaded is not None:
        suffix = ".ipynb" if uploaded.name.endswith(".ipynb") else ".py"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            temp_file_path = tmp.name
            target_file = temp_file_path

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Agent & Verifier Configuration")
model_name = st.sidebar.selectbox(
    "Open-Weight LLM (Ollama)",
    ["qwen2.5-coder:7b", "llama3.1:8b", "mistral:7b", "custom"],
    index=0,
)
if model_name == "custom":
    model_name = st.sidebar.text_input("Custom Model Tag", value="qwen2.5-coder:7b")

use_replay = st.sidebar.checkbox(
    "⚡ Fast Demo Replay Cache",
    value=True,
    help="Uses cached deterministic agent results for instantaneous (~0.05s) live presentation speed.",
)

seeds_input = st.sidebar.text_input("Evaluation Seeds", value="42, 101, 7")
try:
    seeds = [int(s.strip()) for s in seeds_input.split(",") if s.strip()]
except ValueError:
    seeds = DEFAULT_SEEDS

run_btn = st.sidebar.button("🔍 Run Empirical Audit", type="primary", use_container_width=True)

# Main Execution Flow
if run_btn and target_file:
    if not os.path.exists(target_file):
        st.error(f"Target file not found: {target_file}")
    else:
        with st.spinner("🕵️ REPROVE agent is auditing experiment & running counterfactual verifications..."):
            report = run_investigation(
                file_path=target_file,
                model_name=model_name,
                use_replay=use_replay,
                seeds=seeds,
            )
            st.session_state["report"] = report
            st.session_state["target_file"] = target_file

if "report" in st.session_state:
    report: InvestigationReport = st.session_state["report"]
    md_text, html_text = generate_report(report)

    # Top Verdict Banner
    num_confirmed = len(report.confirmed_evidence)
    num_suspected = len(report.suspected_evidence)
    num_clean = len(report.clean_checks)

    if num_confirmed > 0:
        vr = report.verification_results[0] if report.verification_results else None
        delta_str = f" (Measured Delta Δ = {vr.mean_delta:+.4f})" if vr else ""
        st.error(
            f"### 🚨 AUDIT VERDICT: {num_confirmed} CONFIRMED EVALUATION LEAK(S) DETECTED{delta_str}\n"
            f"The reported evaluation metric is empirically inflated. The model's true generalization performance is lower than reported."
        )
    elif num_suspected > 0:
        st.warning(
            f"### ⚠️ AUDIT VERDICT: {num_suspected} SUSPECTED LEAK(S) FOUND (UNCONFIRMED)\n"
            f"Deterministic checks flagged anomalies but empirical verification was inconclusive."
        )
    else:
        st.success(
            "### ✅ AUDIT VERDICT: CLEAN EXPERIMENT PIPELINE\n"
            "All deterministic evidence checks passed. No data leakage or test contamination detected."
        )

    # Metric Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confirmed Leaks", num_confirmed, delta=f"-{num_confirmed}" if num_confirmed > 0 else "0", delta_color="inverse")
    c2.metric("Suspected Leaks", num_suspected)
    c3.metric("Verified Clean Checks", num_clean)
    c4.metric("Agent Steps", len(report.steps))

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🕵️ Investigation Trace",
        "📊 Empirical Verification",
        "🛠️ Counterfactual Patch",
        "📄 Full Audit Report",
    ])

    with tab1:
        st.subheader("Agent Thought Process & Tool Invocations")
        for step in report.steps:
            with st.expander(f"Step {step.step_number}: `{step.tool_name}`", expanded=(step.step_number == 1)):
                st.markdown(f"**Agent Thought:**\n> {step.thought}")
                st.markdown(f"**Tool Invocations:** `{step.tool_name}({step.tool_args or ''})`")
                st.markdown(f"**Deterministic Observation:**\n```text\n{step.observation}\n```")

    with tab2:
        st.subheader("Empirical Verification Results (Multi-Seed Reruns)")
        if report.verification_results:
            for vr in report.verification_results:
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("Original Reported Metric", f"{vr.original_mean:.4f}", f"±{vr.original_std:.4f}")
                v2.metric("Patched Honest Metric", f"{vr.patched_mean:.4f}", f"±{vr.patched_std:.4f}")
                v3.metric("Measured Delta (Δ)", f"{vr.mean_delta:+.4f}", delta=f"{vr.mean_delta:+.4f}", delta_color="inverse")
                v4.metric("Noise Margin (2σ)", f"±{2 * vr.seed_noise_std:.4f}")

                st.markdown(f"**Verification Status:** `{vr.status.value}`")
                st.markdown(f"**Verdict Rationale:** {vr.reason}")

                # Multi-seed table
                st.markdown("##### Per-Seed Execution Measurements")
                seed_data = []
                for s in vr.seed_runs:
                    seed_data.append({
                        "Seed": s.seed,
                        "Original Score": f"{s.original_metric:.4f}",
                        "Patched Score": f"{s.patched_metric:.4f}",
                        "Delta": f"{s.delta:+.4f}",
                    })
                st.table(seed_data)
        else:
            st.info("No confirmed leaks required counterfactual verification rerun. The experiment is clean.")

    with tab3:
        st.subheader("AST-Generated Counterfactual Patch")
        if report.verification_results and report.verification_results[0].diff:
            diff_text = report.verification_results[0].diff
            st.code(diff_text, language="diff")
        else:
            st.info("No patch needed. The experiment pipeline is already leak-free.")

    with tab4:
        st.subheader("Audit Report Export")
        st.download_button(
            "📥 Download Markdown Report (report.md)",
            data=md_text,
            file_name="report.md",
            mime="text/markdown",
        )
        st.download_button(
            "📥 Download HTML Report (report.html)",
            data=html_text,
            file_name="report.html",
            mime="text/html",
        )
        st.markdown(md_text)
