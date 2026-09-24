"""Streamlit web application for DataSense AI Text-to-SQL System."""

from __future__ import annotations

import time
import streamlit as st
import pandas as pd
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
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-success {
        background-color: #d1e7dd;
        color: #0f5132;
    }
    .badge-warning {
        background-color: #fff3cd;
        color: #664d03;
    }
    .badge-danger {
        background-color: #f8d7da;
        color: #842029;
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


# Initialize DB and Schema
db_conn = load_database()
schema_dict, schema_str = load_schema()

# Sidebar: Schema Inspector and Database Info
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
        "What are the top 5 customers by total account balance?",
        "What is the total revenue grouped by order priority?",
        "List the top 5 nations by customer count.",
        "Which 5 suppliers have the highest available parts quantity?",
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
    '<div class="sub-title">Natural Language Text-to-SQL system running on DuckDB with TPC-H benchmark analytics.</div>',
    unsafe_allow_html=True,
)

# Text Input for Business Question
default_val = selected_sample if selected_sample != "-- Select a question --" else ""
user_question = st.text_input(
    "Ask a business question about your data:",
    value=default_val,
    placeholder="e.g. What are the top 5 customers by total account balance?",
)

col_run, col_clear = st.columns([1, 5])
with col_run:
    run_btn = st.button("🚀 Generate & Run SQL", type="primary", width="stretch")

if run_btn:
    if not user_question.strip():
        st.warning("Please enter a question or select a sample question above.")
    else:
        st.write("---")
        # Step 1: SQL Generation
        with st.spinner("🤖 Generating DuckDB SQL query..."):
            try:
                # If user tests direct SQL or safety rejection in the prompt:
                cleaned_input = user_question.strip()
                sql_keywords = ("SELECT ", "WITH ", "DROP ", "DELETE ", "UPDATE ", "INSERT ", "ALTER ", "CREATE ")
                if cleaned_input.upper().startswith(sql_keywords):
                    # User directly supplied SQL to test
                    generated_sql = cleaned_input
                else:
                    generated_sql = generate_sql(user_question, schema_str)
                sql_generation_error = None
            except Exception as exc:
                generated_sql = None
                sql_generation_error = str(exc)

        if sql_generation_error:
            st.error(f"❌ **Generation Error**: {sql_generation_error}")
        else:
            # Step 2: AST Safety Validation
            st.subheader("1. Generated SQL & Safety Validation")
            is_valid, validation_msg = validate_sql(generated_sql, allowed_tables=TPCH_TABLES)

            if not is_valid:
                st.error(f"🛡️ **Safety Validator Rejection**: {validation_msg}")
                st.markdown("**Rejected SQL Query:**")
                st.code(generated_sql, language="sql")
            else:
                st.success(f"✅ **Safety Check Passed**: {validation_msg}")
                st.code(generated_sql, language="sql")

                # Step 3: Execution against DuckDB
                st.subheader("2. Query Execution & Results")
                start_time = time.perf_counter()
                results_df, query_err = execute_query(generated_sql, conn=db_conn)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                if query_err:
                    st.error(f"⚠️ **Database Execution Error**: {query_err}")
                elif results_df is not None:
                    # Metrics row
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Execution Latency", f"{elapsed_ms:.2f} ms")
                    with col_m2:
                        st.metric("Rows Returned", f"{len(results_df):,}")
                    with col_m3:
                        st.metric("Columns", f"{len(results_df.columns)}")

                    # Interactive Dataframe
                    st.dataframe(results_df, width="stretch")
