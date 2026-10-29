"""Streamlit web application for DataSense AI Text-to-SQL System."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path for Streamlit execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import yaml

from src.config import settings
from src.database import (
    get_connection,
    init_tpch_db,
    get_schema_info,
    execute_query,
    TPCH_TABLES,
)
from src.validator import validate_sql
from src.generator import generate_sql
from src.retriever import SchemaMetricRetriever
from src.pipeline import SelfCorrectingPipeline, PipelineResult
from src.ambiguity import AmbiguityResult
from src.evaluator import BenchmarkRunner, BenchmarkReport, TaskResult

METRICS_FILE = Path(__file__).resolve().parent.parent / "data" / "metrics.yaml"
BENCHMARK_FILE = Path(__file__).resolve().parent.parent / "data" / "benchmark_queries.json"

# Page configuration
st.set_page_config(
    page_title="DataSense AI | Text-to-SQL",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished aesthetic
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }
    .badge-metric {
        background-color: #cfe2ff;
        color: #084298;
    }
    .badge-filter {
        background-color: #d1e7dd;
        color: #0f5132;
    }
    .badge-clarified {
        background-color: #e2d9f3;
        color: #432874;
    }
    .badge-semantic {
        background-color: #f3e8ff;
        color: #6b21a8;
    }
    .attempt-card {
        background-color: #f8f9fa;
        border-left: 4px solid #fd7e14;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.8rem;
    }
    .attempt-card-semantic {
        background-color: #faf5ff;
        border-left: 4px solid #9333ea;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.8rem;
    }
    .clarification-card {
        background-color: #fff9db;
        border-left: 5px solid #f59f00;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing DuckDB TPC-H Benchmark...")
def load_database():
    """Cache DuckDB connection and initialize TPC-H tables."""
    conn = get_connection()
    init_tpch_db(conn, scale_factor=settings.tpch_scale_factor)
    return conn


@st.cache_data(show_spinner="Extracting database schema...")
def load_schema():
    """Cache extracted schema dictionary and formatted string."""
    conn = load_database()
    schema_dict = get_schema_info(conn, as_string=False)
    schema_str = get_schema_info(conn, as_string=True)
    return schema_dict, schema_str


@st.cache_resource(show_spinner="Indexing business metrics in ChromaDB...")
def load_retriever(schema_dict_data):
    """Cache ChromaDB vector retriever and index metrics catalog."""
    retriever = SchemaMetricRetriever()
    retriever.index_catalog(str(METRICS_FILE), schema_dict_data)
    return retriever


# Initialize DB, Schema, and Vector Retriever
db_conn = load_database()
schema_dict, schema_str = load_schema()
retriever = load_retriever(schema_dict)

# Initialize Self-Correction Pipeline
pipeline = SelfCorrectingPipeline(
    db_conn=db_conn,
    retriever=retriever,
)

# Sidebar: Schema Inspector, Metrics Catalog, and Database Info
with st.sidebar:
    st.header("🗄️ Database & Schema")
    st.caption(f"Backend: **DuckDB** | Scale Factor: **{settings.tpch_scale_factor}**")

    provider = settings.get_effective_provider()
    if provider == "gemini":
        st.success(f"🤖 LLM: **Gemini** (`{settings.gemini_model}`)")
    elif provider == "openai":
        st.success(f"🤖 LLM: **OpenAI** (`{settings.openai_model}`)")
    else:
        st.warning("⚠️ No API Key Configured in `.env`")
        st.info("Set `GEMINI_API_KEY` or `OPENAI_API_KEY` in `.env` to enable live LLM generation.")

    st.divider()
    st.subheader("📚 Business Metric Dictionary")
    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            raw_metrics_yaml = yaml.safe_load(f) or {}

        with st.expander("📊 Defined Metrics & Formulas", expanded=False):
            for m in raw_metrics_yaml.get("metrics", []):
                st.markdown(f"**{m['name']}** (`{m['id']}`)")
                st.code(m["formula"], language="sql")
                st.caption(f"_{m['description']}_")

        with st.expander("🎯 Defined Business Filters", expanded=False):
            for flt in raw_metrics_yaml.get("filters", []):
                st.markdown(f"**{flt['name']}** (`{flt['id']}`)")
                st.code(flt["condition"], language="sql")
                if flt.get("join_condition"):
                    st.caption(f"Join: `{flt['join_condition']}`")
                st.caption(f"_{flt['description']}_")
    except Exception as exc:
        st.caption(f"Could not load metrics dictionary: {exc}")

    st.divider()
    st.subheader("📋 TPC-H Tables")
    if isinstance(schema_dict, dict):
        for table_name, columns in sorted(schema_dict.items()):
            with st.expander(f"📦 `{table_name}` ({len(columns)} cols)"):
                col_df = pd.DataFrame(columns)
                st.dataframe(
                    col_df,
                    width="stretch",
                    hide_index=True,
                )

    st.divider()
    st.subheader("💡 Sample Questions")
    sample_queries = [
        "Show top customers",  # Triggers HARD_AMBIGUITY (Revenue vs. Orders vs. Balance)
        "Show orders",  # Triggers SOFT_DEFAULT (limit and sort order)
        "What is the net revenue for customers in Germany?",
        "Calculate the item margin for urgent orders.",
        "What is the total discount loss on lineitems?",
        "What are the top 5 customers by total account balance?",
        "What is the total revenue grouped by order priority?",
        "SELECT * FROM read_csv('/etc/passwd');",  # Test AST & engine file-read protection
        "DROP TABLE customer;",  # Test validator safety alert on DDL
    ]

    selected_sample = st.selectbox(
        "Choose a sample to populate:",
        ["-- Select a question --"] + sample_queries,
    )

# Main Application Layout
st.markdown('<div class="main-title">⚡ DataSense AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Phase 5: Semantic Validation Engine · Ambiguity Classifier · Self-Correction · ChromaDB RAG.</div>',
    unsafe_allow_html=True,
)

# Initialize Session State
st.session_state.setdefault("active_question", "")
st.session_state.setdefault("ambiguity_result", None)
st.session_state.setdefault("clarified_context", "")
st.session_state.setdefault("pipeline_result", None)
# Benchmark tab — isolated from query assistant state
st.session_state.setdefault("benchmark_report", None)
st.session_state.setdefault("benchmark_running", False)

# ─── Top-Level Tab Navigation ────────────────────────────────────────────────
tab_qa, tab_bench = st.tabs(["💬 Query Assistant", "📊 Evaluation Benchmark"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — QUERY ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_qa:
    # Text Input for Business Question
    default_val = selected_sample if selected_sample != "-- Select a question --" else ""
    user_question = st.text_input(
        "Ask a business question about your data:",
        value=default_val,
        placeholder="e.g. Show top customers",
    )
    
    # Reset Guardrail: Invalidate clarification state if user changes query text
    if user_question != st.session_state.active_question:
        st.session_state.active_question = user_question
        st.session_state.ambiguity_result = None
        st.session_state.clarified_context = ""
        st.session_state.pipeline_result = None
    
    col_run, col_clear = st.columns([1, 5])
    with col_run:
        run_btn = st.button("🚀 Generate & Run SQL", type="primary", width="stretch")
    
    if run_btn:
        if not user_question.strip():
            st.warning("Please enter a question or select a sample question above.")
        else:
            st.session_state.pipeline_result = None
            st.session_state.clarified_context = ""
            st.session_state.ambiguity_result = None
    
            cleaned_input = user_question.strip()
            sql_keywords = ("SELECT ", "WITH ", "DROP ", "DELETE ", "UPDATE ", "INSERT ", "ALTER ", "CREATE ")
            is_direct_sql = cleaned_input.upper().startswith(sql_keywords)
    
            if is_direct_sql:
                # Bypass ambiguity check for direct SQL test statements
                with st.spinner("🤖 Processing direct SQL query..."):
                    result = pipeline.execute_with_repair(user_question, max_retries=3)
                st.session_state.pipeline_result = result
            else:
                # Step 1: Pre-retrieve RAG Context to inform Ambiguity Classifier
                rag_context = ""
                if retriever is not None:
                    try:
                        rag_context, _ = retriever.retrieve(user_question, top_k_metrics=2, top_k_filters=2, top_k_tables=3)
                    except Exception:
                        rag_context = ""
    
                with st.spinner("🔍 Analyzing query intent & detecting ambiguity..."):
                    amb_res = pipeline.check_ambiguity(
                        user_question=user_question,
                        schema_info=schema_str,
                        rag_context=rag_context,
                    )
    
                if amb_res.is_ambiguous:
                    st.session_state.ambiguity_result = amb_res
                    st.session_state.pipeline_result = None
                else:
                    default_ctx = amb_res.default_interpretation if amb_res.ambiguity_type == "SOFT_DEFAULT" else ""
                    st.session_state.clarified_context = default_ctx
                    with st.spinner("🤖 Processing with Self-Correcting Execution Pipeline..."):
                        result = pipeline.execute_with_repair(
                            user_question,
                            max_retries=3,
                            clarified_context=default_ctx,
                        )
                    st.session_state.pipeline_result = result
    
    # Step 2: Interactive Clarification Modal Card (Triggered on HARD_AMBIGUITY)
    if st.session_state.ambiguity_result is not None and st.session_state.ambiguity_result.is_ambiguous:
        amb_res: AmbiguityResult = st.session_state.ambiguity_result
        st.write("---")
        st.markdown(
            f"""
            <div class="clarification-card">
                <h4 style="margin: 0 0 0.4rem 0; color: #d9480f;">⚠️ Ambiguous Request Detected</h4>
                <p style="margin: 0 0 0.5rem 0; color: #495057;">{amb_res.reasoning}</p>
                <strong>{amb_res.clarification_prompt or "How would you like to interpret this request?"}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
        if amb_res.options:
            options_labels = [f"**{opt['label']}** — {opt['details']}" for opt in amb_res.options]
            selected_option_idx = st.radio(
                "Select an intended business definition:",
                range(len(options_labels)),
                format_func=lambda i: options_labels[i],
                key="ambiguity_radio_selection",
            )
    
            col_opt1, col_opt2 = st.columns([2, 2])
            with col_opt1:
                if st.button("✅ Proceed with Selected Option", type="primary", use_container_width=True):
                    chosen = amb_res.options[selected_option_idx]
                    clarified = f"{chosen['label']}: {chosen['details']}"
                    st.session_state.clarified_context = clarified
                    st.session_state.ambiguity_result = None
                    with st.spinner("🤖 Processing query with selected clarification..."):
                        res = pipeline.execute_with_repair(
                            user_question=user_question,
                            max_retries=3,
                            clarified_context=clarified,
                        )
                    st.session_state.pipeline_result = res
                    st.rerun()
    
            with col_opt2:
                fallback_label = amb_res.default_interpretation or "Standard system default"
                button_title = f"⚙️ Use Default ({fallback_label[:30]}...)" if len(fallback_label) > 30 else f"⚙️ Use Default ({fallback_label})"
                if st.button(button_title, type="secondary", use_container_width=True):
                    st.session_state.clarified_context = amb_res.default_interpretation
                    st.session_state.ambiguity_result = None
                    with st.spinner("🤖 Processing query with default interpretation..."):
                        res = pipeline.execute_with_repair(
                            user_question=user_question,
                            max_retries=3,
                            clarified_context=amb_res.default_interpretation,
                        )
                    st.session_state.pipeline_result = res
                    st.rerun()
    
    # Step 3: Render Execution Results (if available)
    if st.session_state.pipeline_result is not None:
        result: PipelineResult = st.session_state.pipeline_result
        st.write("---")
    
        # Display Clarification Badge if intent was clarified
        if st.session_state.clarified_context:
            st.markdown(
                f'<span class="metric-badge badge-clarified">💡 CLARIFIED INTENT</span> '
                f'<strong>{st.session_state.clarified_context}</strong>',
                unsafe_allow_html=True,
            )
            st.write("")
    
        # 1. RAG Context Inspection Card
        if result.rag_metadata:
            with st.expander("🔍 RAG Context & Retrieved Business Metrics", expanded=True):
                st.markdown("**Semantically Matched Definitions:**")
                for item in result.rag_metadata:
                    badge_class = "badge-metric" if item.get("kind") == "metric" else "badge-filter"
                    tag_type = "METRIC" if item.get("kind") == "metric" else "FILTER"
                    st.markdown(
                        f'<span class="metric-badge {badge_class}">{tag_type}</span> '
                        f'**{item.get("name")}** (`{item.get("id")}`)',
                        unsafe_allow_html=True,
                    )
                    if item.get("formula"):
                        st.code(item.get("formula"), language="sql")
                    if item.get("condition"):
                        st.code(item.get("condition"), language="sql")
                    if item.get("join_condition"):
                        st.caption(f"Join path: `{item.get('join_condition')}`")
    
                st.markdown("---")
                st.markdown("**Prompt Context Injected into LLM:**")
                st.markdown(f"```markdown\n{result.rag_context}\n```")
    
        # 2. Security Violation Rejection
        if result.is_security_violation:
            st.error(f"🛡️ **Security Policy Violation Blocked**: {result.error_message}")
            st.markdown("**Rejected SQL Statement:**")
            st.code(result.final_sql, language="sql")
            st.caption("🔒 Security violations are strictly non-retriable and terminated immediately on Attempt 1.")
    
        # 3. Successful Execution Outcome
        elif result.success:
            if result.total_attempts == 1:
                st.success("✅ **Query Succeeded on First Attempt**")
            elif result.was_semantically_repaired:
                st.info(
                    f"🧠 **Semantically Auto-Repaired on Attempt {result.total_attempts}** "
                    f"— semantic anomaly detected & corrected across {result.total_attempts - 1} reflection(s)"
                )
            else:
                st.info(
                    f"⚡ **Syntax Auto-Repaired on Attempt {result.total_attempts}** "
                    f"({result.total_attempts - 1} error reflection(s) resolved)"
                )
    
            # Repair History Card (if retries occurred)
            if result.attempt_history:
                with st.expander(
                    f"🔄 Self-Correction & Repair History ({len(result.attempt_history)} previous failure(s) resolved)",
                    expanded=True,
                ):
                    for record in result.attempt_history:
                        is_semantic = record.get("error_type") == "SEMANTIC_VALIDATION"
                        card_class = "attempt-card-semantic" if is_semantic else "attempt-card"
                        icon = "🧠" if is_semantic else "⚠️"
                        issue_type_html = (
                            f'<br/><span class="metric-badge badge-semantic">' 
                            f'{record.get("issue_type", "SEMANTIC")}</span>'
                            if is_semantic else ""
                        )
                        st.markdown(
                            f'<div class="{card_class}">' 
                            f'<strong>{icon} Attempt #{record.get("attempt")} · {record.get("error_type")}' 
                            f'</strong>{issue_type_html}<br/>' 
                            f'<code style="font-size:0.82rem;">{record.get("error")}</code>' 
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        st.code(record.get("sql"), language="sql")
    
            # Semantic Validation Summary (Stage 1/2 clean pass indicator)
            if result.semantic_validation is not None:
                sem = result.semantic_validation
                if sem.issue_type == "NONE":
                    judge_note = {
                        "PASSED": "The LLM judge also passed.",
                        "DISABLED": "The LLM judge is disabled.",
                        "UNAVAILABLE": "The LLM judge was unavailable.",
                        "ERROR": "The LLM judge could not verify this result.",
                    }.get(sem.judge_status, "The LLM judge did not run.")
                    st.markdown(
                        '<span class="metric-badge badge-semantic">🧠 SEMANTIC PASS</span> '
                        f'Output passed deterministic checks. {judge_note}',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<span class="metric-badge badge-semantic">🧠 SEMANTICALLY REPAIRED · {sem.issue_type}</span> '
                        f'{sem.reasoning}',
                        unsafe_allow_html=True,
                    )
                st.write("")
    
            # Final SQL & Results
            st.subheader("1. Executed DuckDB SQL")
            st.code(result.final_sql, language="sql")
    
            st.subheader("2. Query Execution & Results")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Total Latency", f"{result.execution_time_ms:.2f} ms")
            with col_m2:
                st.metric("Attempts", f"{result.total_attempts}")
            with col_m3:
                rows_count = len(result.df) if result.df is not None else 0
                st.metric("Rows Returned", f"{rows_count:,}")
            with col_m4:
                cols_count = len(result.df.columns) if result.df is not None else 0
                st.metric("Columns", f"{cols_count}")
    
            if result.df is not None:
                st.dataframe(result.df, width="stretch")
    
        # 4. Failed Execution (Exhausted Retries or Unrecoverable Error)
        else:
            st.error(f"❌ **Execution & Repair Failed**: {result.error_message}")
    
            if result.attempt_history:
                with st.expander("🔄 Attempt History & Error Log", expanded=True):
                    for record in result.attempt_history:
                        st.markdown(
                            f'<div class="attempt-card">'
                            f'<strong>Attempt #{record.get("attempt")} Error ({record.get("error_type")}):</strong><br/>'
                            f'<code>{record.get("error")}</code>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        st.code(record.get("sql"), language="sql")
    
            if result.final_sql:
                st.markdown("**Last Attempted SQL Query:**")
                st.code(result.final_sql, language="sql")
    

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATION BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════
with tab_bench:
    st.markdown("### 📊 TPC-H Execution Accuracy Benchmark")
    st.caption(
        "Evaluates the full DataSense AI pipeline (RAG + Ambiguity Classifier + Self-Correction + Semantic Validator) "
        "against 20 diverse TPC-H business queries. Ground truth SQL is executed **live** against the same DuckDB instance."
    )
    st.caption("Pass@1 and Pass@N use the same initial SQL. Result comparison ignores aliases, preserves string case and duplicate rows, and uses absolute numeric tolerance 0.01.")

    bench_col1, bench_col2 = st.columns([2, 4])
    with bench_col1:
        bench_provider = st.selectbox(
            "LLM Provider",
            ["auto", "gemini", "openai"],
            key="bench_provider_select",
        )
        bench_retries = st.slider("Max Self-Repair Retries", 1, 5, 3, key="bench_retries_slider")

    with bench_col2:
        st.markdown("")
        st.markdown("")
        run_bench_btn = st.button(
            "🚀 Run Full TPC-H Benchmark",
            type="primary",
            use_container_width=True,
            key="run_benchmark_btn",
            disabled=st.session_state.benchmark_running,
        )

    if run_bench_btn:
        st.session_state.benchmark_report = None
        st.session_state.benchmark_running = True
        provider_arg = None if bench_provider == "auto" else bench_provider

        if not BENCHMARK_FILE.exists():
            st.error(f"Benchmark file not found: {BENCHMARK_FILE}")
            st.session_state.benchmark_running = False
        else:
            import json as _json
            with open(BENCHMARK_FILE, "r", encoding="utf-8") as _f:
                _tasks = _json.load(_f)
            total_tasks = len(_tasks)

            progress_bar = st.progress(0, text="Initialising benchmark runner…")
            status_text = st.empty()

            def _streamlit_progress(current, total, task_id):
                pct = current / total
                progress_bar.progress(pct, text=f"Task {current}/{total} — `{task_id}`")
                status_text.caption(f"⏳ Running `{task_id}`…")

            try:
                bench_runner = BenchmarkRunner(
                    db_conn=db_conn,
                    retriever=retriever,
                    max_retries=bench_retries,
                    provider=provider_arg,
                )
                report: BenchmarkReport = bench_runner.run_benchmark(
                    benchmark_file=str(BENCHMARK_FILE),
                    progress_callback=_streamlit_progress,
                )
                st.session_state.benchmark_report = report
                progress_bar.progress(1.0, text="✅ Benchmark complete!")
                status_text.empty()
            except Exception as _exc:
                st.error(f"Benchmark failed: {_exc}")
            finally:
                st.session_state.benchmark_running = False

    # ── Results Display ───────────────────────────────────────────────────────
    report: BenchmarkReport = st.session_state.benchmark_report

    if report is not None:
        st.write("---")
        if report.evaluation_error_count:
            st.warning(f"This run had {report.evaluation_error_count} evaluation/provider error(s). Review the exported diagnostics before interpreting accuracy.")

        # Headline metric cards
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Total Tasks", report.total_tasks)
        with m2:
            st.metric(
                "Pass@1 Accuracy",
                f"{report.pass_at_1_accuracy:.1f}%",
                delta=None,
            )
        with m3:
            st.metric(
                "Pass@N Accuracy",
                f"{report.pass_at_n_accuracy:.1f}%",
            )
        with m4:
            delta_val = report.self_correction_delta
            st.metric(
                "Self-Correction Δ EX",
                f"+{delta_val:.1f}%" if delta_val >= 0 else f"{delta_val:.1f}%",
                delta=f"{delta_val:.1f}%",
                delta_color="normal",
            )
        with m5:
            st.metric(
                "Avg Latency",
                f"{report.avg_latency_ms / 1000:.2f}s",
            )

        # Secondary metrics row
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Semantic Repairs", report.semantic_repair_count)
        with s2:
            st.metric(
                "Ambiguity Classifier Accuracy",
                f"{report.ambiguity_accuracy:.1f}%",
            )
        with s3:
            st.metric("EX-Eligible Tasks", report.ex_eligible_tasks)

        # Per-task results table
        st.write("---")
        st.markdown("#### 📋 Per-Task Results")

        rows = []
        for t in report.task_results:
            if t.ex_eligible:
                p1 = "✅ Pass" if t.pass_at_1 else "❌ Fail"
                pn = "✅ Pass" if t.pass_at_n else "❌ Fail"
            else:
                p1 = "—"
                pn = "—"

            sem_badge = "🧠" if t.was_semantically_repaired else ""
            amb_badge = "✅" if t.ambiguity_correct else "❌"

            rows.append({
                "ID": t.task_id,
                "Category": t.category,
                "Question": t.question[:55] + ("…" if len(t.question) > 55 else ""),
                "Expected Amb.": t.expected_ambiguity,
                "Amb. Correct": amb_badge,
                "Pass@1": p1,
                "Pass@N": pn,
                "Attempts": t.total_attempts if t.ex_eligible else "—",
                "Sem.Repair": sem_badge,
                "Latency (s)": f"{t.latency_ms / 1000:.2f}",
            })

        results_df = pd.DataFrame(rows)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Per-task detail expander for failures
        failures = [t for t in report.task_results if t.ex_eligible and t.pass_at_n is False]
        if failures:
            st.write("---")
            with st.expander(f"🔍 Failed Task Details ({len(failures)} failures)", expanded=False):
                for t in failures:
                    st.markdown(f"**`{t.task_id}`** — {t.question}")
                    fcol1, fcol2 = st.columns(2)
                    with fcol1:
                        st.caption("Generated SQL:")
                        st.code(t.generated_sql or "N/A", language="sql")
                    with fcol2:
                        st.caption("Ground Truth SQL:")
                        st.code(t.ground_truth_sql or "N/A", language="sql")
                    if t.error_message:
                        st.caption(f"Error: {t.error_message}")
                    st.divider()

        # Export report
        st.write("---")
        import json as _json2
        report_json = _json2.dumps(report.to_dict(), indent=2, default=str)
        st.download_button(
            label="⬇️ Download Full Report (JSON)",
            data=report_json,
            file_name="benchmark_results.json",
            mime="application/json",
            key="download_report_btn",
        )
