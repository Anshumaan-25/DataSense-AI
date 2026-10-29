# DataSense AI — Comprehensive System Summary, Technical Audit & Future Roadmap

> **Document Version**: 1.2.0<br>
> **Last Updated**: October 29, 2026<br>
> **Status**: Phases 1–6, Tier 1 enhancements, and correctness/evaluation implementation verified (225/225 tests passing); live benchmark incomplete due to provider deadline failure<br>
> **Repository**: [https://github.com/Anshumaan-25/DataSense-AI.git](https://github.com/Anshumaan-25/DataSense-AI.git)

---

## 1. Executive Summary & Project Purpose

**DataSense AI** is an enterprise-grade, self-correcting Text-to-SQL analytical platform designed for complex analytical data warehouses. Operating on **DuckDB** using the standardized **TPC-H benchmark dataset** (default scale factor 0.1), the system bridges natural language business questions to highly optimized, strictly read-only SQL queries.

Unlike naive LLM Text-to-SQL prototypes that hallucinate column names, fail on business formulas, silently execute arbitrary interpretations, or expose databases to prompt injection and Local File Inclusion (LFI), DataSense AI integrates:
1. **Multi-Tier Zero-Trust AST Security Sandboxing** (`sqlglot` + DuckDB C++ lockdown).
2. **Domain-Specific Schema & Metric Vector RAG** (`ChromaDB` persistent storage with incremental content-hash caching).
3. **3-Tier Business Ambiguity Classification & Interactive Human-in-the-Loop Clarification** (`CLEAR`, `SOFT_DEFAULT`, `HARD_AMBIGUITY`).
4. **Bounded Self-Correction State Machine** with compiler error reflection, sample rows for conversion/type errors, loop stagnation detection, and actual per-attempt SQL/output telemetry.
5. **2-Stage Semantic Validator** (connection-specific casing profiles, AST catalog-formula checks for requested metrics, and an optional LLM judge with explicit execution status).
6. **Execution Accuracy (EX) Benchmarking Suite** observing Pass@1 and Pass@N from the same repair trajectory. Value-based comparison supports aliases, a global column permutation, duplicate-preserving row matching, and explicit numeric/string/order policies. Gold outputs never drive repairs. Token usage is not currently recorded.

---

## 2. Architecture & Pipeline Flow

```
                     [ Natural Language Question ]
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │ 1. ChromaDB Semantic RAG Retriever (retriever.py)    │
        │    • metrics_catalog (net_revenue, item_margin, etc.)│
        │    • filters_catalog (german_customers, urgent)      │
        │    • schema_catalog (table DDLs & column specs)      │
        │    • Persistent collections + content-hash reuse    │
        └──────────────────────────────────────────────────────┘
                                   │
                                   ▼ (Retrieved Formulas & Business Rules)
        ┌──────────────────────────────────────────────────────┐
        │ 2. Ambiguity Classifier & Clarification Agent        │
        │    (src/ambiguity.py)                                │
        │    • 3-Tier Taxonomy: CLEAR | SOFT_DEFAULT | HARD    │
        │    • ⚠️ HARD_AMBIGUITY ──► Streamlit Clarification   │
        │    • CLEAR / SOFT_DEFAULT ──► Auto-proceed           │
        └──────────────────────────────────────────────────────┘
                                   │
                                   ▼ (Grounded Intent & Formulas)
        ┌──────────────────────────────────────────────────────┐
        │ 3. Bounded Self-Correction (default: 3 attempts)      │
        │    (src/pipeline.py)                                 │
        │                                                      │
        │   [Attempt Loop (1 .. 3)]                            │
        │   ├── 3.1 LLM SQL Generator (generator.py)           │
        │   │       • Prompt compilation with schema + RAG     │
        │   │       • Error feedback reflection on retry       │
        │   │       • Type/conversion errors: 3 sample rows    │
        │   │                                                  │
        │   ├── 3.2 AST Safety Validator (validator.py)        │
        │   │       • Prohibited DDL/DML & file access check   │
        │   │       • Fatal security abort (zero retry)        │
        │   │       • Syntax error → Feedback loop             │
        │   │                                                  │
        │   ├── 3.3 DuckDB OLAP Execution (database.py)        │
        │   │       • In-process analytical query run          │
        │   │       • Runtime binder/catalog error → Loop      │
        │   │                                                  │
        │   └── 3.4 Fast-Path 2-Stage Semantic Validator       │
        │       (src/semantic.py)                              │
        │       • Stage 1: Deterministic Heuristics            │
        │         - Casing from profiled database values      │
        │         - Cartesian join / row explosion detection   │
        │         - Metric formula omission detection          │
        │       • Stage 2: Optional LLM Conformance Judge      │
        │       • Issue detected → Feedback loop to repair     │
        └──────────────────────────────────────────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │ 4. Interactive Presentation & Evaluation (app.py)    │
        │    • Tab 1: 💬 Query Assistant (Interactive UI)       │
        │    • Tab 2: 📊 Evaluation Benchmark (Live EX Suite)   │
        └──────────────────────────────────────────────────────┘
```

---

## 3. Current Project Status & Completed Milestones

| Phase | Module | Description | Test Coverage | Status |
|---|---|---|---|---|
| **Phase 1** | `src/database.py`<br>`src/validator.py` | DuckDB TPC-H auto-init, AST safety, engine lockdown, cached low-cardinality text profiles, and authorized table sampling. | 49 tests (`test_database.py`, `test_validator.py`) | ✅ Complete |
| **Phase 2** | `src/retriever.py`<br>`src/generator.py`<br>`data/metrics.yaml` | Persistent ChromaDB, content-hash cache invalidation and stale-entry pruning, semantic retrieval, Gemini/OpenAI generation and sample-aware repair prompts. | 21 tests (`test_retriever.py`, `test_generator.py`) | ✅ Complete |
| **Phase 3** | `src/pipeline.py` | Bounded repair with actual per-attempt outcomes, preserved intent, selective samples, positive attempt-budget validation, stagnation detection, and immediate security abort. | 20 tests (`test_repair.py`) | ✅ Complete |
| **Phase 4** | `src/ambiguity.py` | 3-tier taxonomy, retrieval-grounded classification, explicit fallback status, deterministic browse defaults, structured clarification UI. | 17 tests (`test_ambiguity.py`) | ✅ Complete |
| **Phase 5** | `src/semantic.py` | Dynamic casing; requested-metric AST checks with alias/CTE/derived-column support; explicit judge status; remaining deterministic heuristics. | 45 tests (`test_semantic.py`) | ✅ Complete |
| **Phase 6** | `src/evaluator.py`<br>`scripts/run_benchmark.py`<br>`data/benchmark_queries.json` | One-trajectory EX measurement, alias-independent value comparison, tolerance and duplicate handling, safe reference preflight, error-aware reports/CLI, deterministic top-N references. | 73 tests (`test_evaluator.py`, `test_benchmark_cli.py`) | ✅ Implementation verified; live baseline pending |
| **Total** | **All Modules** | **Automated unit and integration suite** | **225 / 225 Passing** | ✅ **Verified offline** |

**Latest verification:** 225 tests passed in 6.13 seconds immediately before committing, using the existing Conda Python 3.13.13 environment: 79 additional tests since the 146-test Tier 1 milestone, and 121 since the original 104-test baseline. Automated provider calls are mocked; persistent storage is also tested across a separate Python process. A live 20-question Gemini run was attempted but stopped during the first task after a separate generation probe returned `504 DEADLINE_EXCEEDED`. No live accuracy score is available. See `reports/correctness_sprint_benchmark_status.json`.

---

## 4. Problems & Technical Issues Encountered & Resolved

### 4.1. Remote Git History Divergence
- **Symptom**: `git push origin main` rejected with non-fast-forward error due to remote repository having 15 independent commits (documentation adjustments).
- **Resolution**: Analyzed git diff to confirm no conflicting source changes, executed `git pull --rebase origin main` to maintain linear history, and pushed all Phase 1–6 features cleanly.

### 4.2. Streamlit Entrypoint `ModuleNotFoundError: No module named 'src'`
- **Symptom**: When launching `streamlit run src/app.py`, Streamlit set Python's `sys.path[0]` to `/src/` instead of project root, causing `from src.config import settings` to crash.
- **Resolution**: Injected dynamic root resolution at the very top of `src/app.py`:
  ```python
  import sys
  from pathlib import Path
  PROJECT_ROOT = Path(__file__).resolve().parent.parent
  if str(PROJECT_ROOT) not in sys.path:
      sys.path.insert(0, str(PROJECT_ROOT))
  ```

### 4.3. Google Gemini Model `503 UNAVAILABLE` (Demand Spikes)
- **Symptom**: Experimental `gemini-2.5-flash` endpoint returned `503 UNAVAILABLE: This model is currently experiencing high demand`.
- **Resolution**:
  1. Updated default model to stable `gemini-3.8-flash` in `src/config.py`, `.env`, and `.env.example`.
  2. Implemented exponential backoff retry loops (up to 3 attempts with progressive sleep) across `src/generator.py`, `src/ambiguity.py`, and `src/semantic.py` to transparently absorb transient API spikes.

### 4.4. Test Suite Isolation with Active `.env` Keys
- **Symptom**: `test_generate_sql_missing_keys_raises` failed once a valid `.env` key was populated because `generate_sql()` succeeded instead of raising `ValueError`.
- **Resolution**: Since `Settings` is a `@dataclass(frozen=True)`, updated `tests/test_generator.py` to use `monkeypatch.setattr(generator, "settings", Settings(gemini_api_key="", openai_api_key=""))` ensuring deterministic test hermeticity.

### 4.5. DuckDB Process-Level Write Locking
- **Symptom**: While Streamlit holds an active connection to `data/tpch.duckdb` in read-write mode, external concurrent terminal processes (e.g. ad-hoc CLI scripts) receive `IO Error: Could not set lock on file ... Conflicting lock is held`.
- **Resolution**: Streamlit caches the connection via `@st.cache_resource`. Documented that CLI benchmarking and parallel scripts should execute with `read_only=True` or on an in-memory instance (`:memory:`) to prevent locking conflicts.

### 4.6. Hardcoded Casing Values and Cross-Database Isolation
- **Previous limitation**: The semantic validator assumed a fixed TPC-H uppercase vocabulary, which could suggest incorrect values for other databases.
- **Resolution**: `init_tpch_db()` now profiles low-cardinality text columns after engine lockdown. A weak-reference cache isolates profiles by connection. Caller-supplied databases are profiled lazily; `profile_distinct_values(conn)` explicitly refreshes a profile. SQL AST scopes associate equality/IN literals with physical table columns; existing exact matches, unrelated literals, negated predicates, and ambiguous casing variants are not guessed. Empty scalar aggregates containing NULL also receive casing checks.

### 4.7. Persistent Index Reuse and Catalog Invalidation
- **Previous limitation**: Application restarts rebuilt an ephemeral vector index; repeated upserts alone did not remove deleted catalog entries.
- **Resolution**: Default clients persist to `data/chroma_storage` (configurable via `CHROMA_STORAGE_PATH`). Versioned SHA-256 hashes cover each document and its metadata. Only changed/missing entries are upserted; removed metrics, filters, and schema tables are deleted. Corpus hashes are recorded after successful collection synchronization, allowing interrupted updates to recover. Tests cover process restarts, changed formulas/schema, removals, empty catalogs, reordering, and missing-entry recovery. Generated storage is git-ignored.

### 4.8. Type Repair Context and Injected Callable Compatibility
- **Previous limitation**: Repair prompts had schema and error text but no examples of actual value formats. The broad `TypeError` compatibility fallback could also invoke a failing model function twice.
- **Resolution**: Conversion/type errors now collect up to three rows from each referenced, authorized physical table before a repair attempt. Strings are bounded to 256 characters plus a truncation marker. CTE aliases are resolved to base tables; external reads and unauthorized queries are rejected. Samples are serialized into both provider repair prompts while preserving RAG, clarification, and error history. Sampling failures do not block repair. Signature inspection preserves legacy injected functions without retrying internal `TypeError`s.

### 4.9. Local Python Interpreter Selection
- **Observed issue**: `/usr/bin/python3` is Python 3.9.6 and lacks `pytest` on the audited machine.
- **Resolution**: Used the already-installed `/opt/homebrew/Caskroom/miniconda/base/bin/python3` (Python 3.13.13), without installing dependencies or changing the shell configuration. The full suite passes under this interpreter.

### 4.10. Metric Formula False Positives and Retrieval Alternatives
- **Reproduced defect**: Correct net-revenue SQL using `l.l_discount` was rejected by the old substring check. Retrieved alternatives could also impose net revenue on an explicit gross-revenue question.
- **Resolution**: Match requested metrics using catalog names/aliases with specificity (e.g. "gross revenue" takes precedence over the contained generic alias "revenue"). Clarified intent takes priority. AST comparison removes table aliases and redundant parentheses, normalizes numeric literals and commutative addition/multiplication, and follows projected CTE/derived columns. Presentation ROUND/CAST wrappers are supported. Comments, strings, unused CTEs, and WHERE subqueries do not satisfy an output formula. All five catalog formulas and custom retrieved formulas have regression coverage.

### 4.11. Repair Measurement Was Confounded by Regeneration
- **Previous defect**: Pass@1 and Pass@N came from separate initial generations, so reported lift included sampling variability and duplicate model cost.
- **Resolution**: `PipelineAttempt` retains each actual SQL, validation outcome, output frame, and error type. One pipeline execution supplies both first-attempt and final scores. Wrong executable results do not trigger a gold-informed repair. Reports identify `single_trajectory_v1`, retain first SQL/error diagnostics, and publish the attempt budget, model/provider, dataset SHA-256, and comparison policy.

### 4.12. EX Comparison Aliases, Precision, and Duplicate Rows
- **Previous defect**: Equivalent aliases or numeric dtypes failed comparison; decimal rounding hid differences and introduced boundary artifacts; lowercasing result strings could hide incorrect data.
- **Resolution**: Compare original values with default absolute tolerance 0.01 and relative tolerance zero, preserving string case, nulls, and duplicate multiplicity. Ignore output aliases and permit one consistent column permutation while verifying complete rows. Bipartite matching handles ambiguous numeric matches without greedy false negatives. Strict normalized column-name and case-insensitive-string policies remain explicit options. Per-task `order_sensitive: true` enables ordered rows; unordered comparison is the default. `DataFrameNormalizer` remains a legacy display utility and is not used for EX scoring.

### 4.13. Classifier/Reference Failures and Nondeterministic Top-N Gold
- **Previous defects**: Benchmark classification lacked RAG context and dropped default interpretations; classifier failures could disappear from the denominator; broken gold SQL reduced EX eligibility; tied LIMIT results depended on unspecified order.
- **Resolution**: Use retrieval-grounded classification and preserve SOFT_DEFAULT through the entire trajectory. Predicted HARD_AMBIGUITY pauses execution and counts as an EX miss for an expected-clear task. Invalid classifier output is flagged and remains in classifier accuracy. Validate dataset IDs/schema and all EX gold SQL safely before LLM work; invalid references abort rather than shrink the denominator. Explicit top-N tie rules and deterministic order-browse defaults were added. The local audit found four parts tied at the top-10 price boundary and 66 orders on the latest date.
- **Reporting**: Completed reports distinguish `COMPLETED` from `COMPLETED_WITH_ERRORS`; CLI exits are 0 (completed), 1 (setup/reference failure), and 2 (completed with evaluation/provider errors). Requested RAG failures are fatal instead of silently disabling retrieval. The CLI accepts `--read-only` for an existing database.

### 4.14. Live Benchmark Provider Deadline Failure — Open
- **Observed on October 29, 2026**: The configured `gemini-3.8-flash` metadata request succeeded. A bounded generation probe returned `504 DEADLINE_EXCEEDED`; the full CLI run stalled generating SQL for its first task and was interrupted (exit 130).
- **Current status**: Zero completed task results and no live EX score. The failure status artifact records null accuracy fields and the retry command. Provider request deadlines/SDK retry limits and durable partial-run persistence remain operational improvements to implement; existing SDK defaults can stall a run.

---

## 5. Current Technical Limitations & Known Edge Cases

1. **Distinct-Value Profiling Bounds and Refresh**:
   - Profiling covers `main` base-table VARCHAR/CHAR/TEXT columns, at most 100 distinct non-null values per column, and values up to 256 characters. High-cardinality columns are skipped. Result size is bounded; the underlying DISTINCT queries can still scan large tables at initialization.
   - Profiles refresh during `init_tpch_db()` or explicit `profile_distinct_values(conn)`. Refresh after external data changes. Standalone `SemanticValidator` callers must supply `db_conn` or `distinct_value_profile` for casing checks. Direct base-table equality/IN predicates are supported; expression/derived-column lineage and ambiguous case variants are skipped.
2. **DuckDB Concurrency Model**:
   - DuckDB is an embedded OLAP engine. While fast, simultaneous writes across multiple processes require server-mode deployment (e.g., DuckDB via MotherDuck or a lightweight database daemon) rather than direct local file mounts.
3. **Local Persistent ChromaDB Scope**:
   - Unchanged entries reuse stored embeddings. Initial indexing still needs the embedding model; hash comparison still loads catalog metadata. Use a separate storage directory for independent catalogs/databases. Collection updates are recoverable but are not an atomic transaction across all three collections; concurrent catalog writers are not coordinated.
4. **LLM Structured Output Variations**:
   - While Gemini 3.8 and OpenAI support JSON schema constraints, prompt drift can occasionally yield extra text outside JSON fences. Fallback regex parsers are active in `ambiguity.py` and `semantic.py`, but native Pydantic schema validation should be reinforced.
5. **Semantic Judge and Benchmark Measurement Scope**:
   - The LLM judge remains disabled by default. UI messages now distinguish a passed judge from disabled, unavailable, and failed verification. Enabled-judge exceptions retain the existing permissive behavior but are labeled `ERROR`, never a successful judge verdict. AST formula checks establish structural agreement, not general SQL algebra equivalence or join correctness. Token telemetry remains future work.
6. **Repair Samples**:
   - Samples are the first up to three rows returned by DuckDB without an explicit ordering guarantee; they may not include the value that caused a conversion failure. Current pipeline authorization remains the TPC-H table whitelist. Authorized sample values are sent to the configured LLM provider during type/conversion repair.
7. **Evaluation and Provider Limits**:
   - EX supports scalar cells; nested/list-valued outputs raise an evaluation error. Ambiguous column permutations and tolerant row matching can be expensive for large result sets; current coverage targets the TPC-H suite. One run over 20 questions is not a generalization study. Provider timeouts and interrupted runs can prevent a completed report; the live baseline remains pending after the observed 504.

---

## 6. Future Improvements & Strategic Roadmap

### Tier 1: Immediate Enhancements — Completed October 29, 2026
- [x] **Dynamic Distinct Value Profiling**:
  Automatically profile low-cardinality text columns during initialization and reuse connection-specific, column-aware casing values in the semantic validator. Explicit refresh and lazy profiling support caller-supplied databases.
- [x] **Persistent ChromaDB Storage**:
  Default to `data/chroma_storage`, reuse unchanged embeddings via versioned content hashes, reindex changed entries, and prune deleted catalog/schema entries. `CHROMA_STORAGE_PATH` overrides the storage directory.
- [x] **Enhanced SQL Repair Context**:
  Inject up to three bounded sample rows from referenced authorized tables for conversion errors and type mismatches, preserving business rules, clarified intent, and prior errors. Both Gemini and OpenAI prompt paths are tested.

### Correctness & Trustworthy Evaluation Sprint — Implementation Completed October 29, 2026
- [x] Requested-metric AST validation with aliases, wrappers, output lineage, and decoy regressions.
- [x] Pass@1 and Pass@N from one observed trajectory without gold feedback into repair.
- [x] Explicit EX policy for aliases, column permutations, numeric tolerance, string case, duplicates, nulls, and optional row ordering.
- [x] Retrieval-grounded ambiguity classification, preserved defaults, and visible classifier failures.
- [x] Safe gold-query preflight, stable eligibility, and deterministic top-N tie rules.
- [x] Accurate judge UI status, auditable reports, and CLI error exits/read-only support.
- [x] Full offline regression suite: 225/225 passing.
- [ ] Complete the live 20-question baseline — attempted; provider deadline failure recorded in `reports/correctness_sprint_benchmark_status.json`.

### Tier 2: Mid-Term Architectural Scaling
- [ ] **Dynamic Few-Shot In-Context Learning (RAG for SQL Exemplars)**:
  Index successful `(question, gold_sql)` pairs into ChromaDB. During query generation, retrieve top-2 semantically similar past queries as few-shot demonstrations to boost Pass@1 accuracy.
- [ ] **Multi-Dialect Database Adapters**:
  Decouple `src/database.py` into an abstract `BaseDatabaseAdapter` interface to allow seamless plug-and-play with PostgreSQL, Snowflake, ClickHouse, and BigQuery.
- [ ] **Interactive Visualizations (Auto-Plot Engine)**:
  Add an automatic chart generator in Streamlit (using Plotly / Vega-Lite) that analyzes the shape and data types of `PipelineResult.result_df` (e.g. time series line charts, category bar charts).

### Tier 3: Enterprise & Governance Capabilities
- [ ] **Role-Based Access Control & Column Masking**:
  Integrate AST-level row-level filtering and column redaction (e.g., masking customer account balances or PII unless authenticated with administrative privileges).
- [ ] **Active Learning & Metric Catalog Evolution**:
  Allow users to mark a query result as "Verified Gold". Add an admin button in Streamlit to export verified queries directly into `data/metrics.yaml` or `data/benchmark_queries.json`.
- [ ] **Asynchronous Parallel Benchmark Harness**:
  Implement a worker-pool asynchronous runner in `scripts/run_benchmark.py` using `asyncio` or `ThreadPoolExecutor` with token rate-limiting to execute the 20 benchmark queries in parallel.

---

## 7. Verification & Run Commands Cheat Sheet

### Environment Setup
```bash
# 1. Activate Virtual Environment
source venv/bin/activate

# 2. Verify / Edit API Configuration
cat .env

# 3. Run Complete Automated Test Suite (225 Tests)
python3 -m pytest tests/ -v
```

Audited-machine fallback when system Python lacks pytest:
```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python3 -m pytest tests/ -v
```

Optional persistent-index setting (relative paths resolve from the project root):
```env
CHROMA_STORAGE_PATH=data/chroma_storage
```

### Web Application Launch
```bash
# Launch Streamlit Interactive UI
streamlit run src/app.py
```

### Standalone Benchmark Execution
```bash
# Run CLI Evaluation Benchmark Suite
python3 scripts/run_benchmark.py --output reports/benchmark_run.json

# Read-only live baseline against an existing database
python3 scripts/run_benchmark.py --read-only --output reports/correctness_sprint_benchmark.json
```

---

## 8. Mandatory Agent Protocol & Maintenance Directive

> ### ⚠️ CRITICAL INSTRUCTION FOR ANY AI MODEL / AGENT READING THIS FILE:
> 
> **You are strictly required to preserve and maintain this document as the single source of truth for the DataSense AI codebase.**
> 
> Whenever you make modifications, bug fixes, refactorings, or implement any phase/feature from the roadmap:
> 1. **Do NOT allow this document to become obsolete or stale.**
> 2. **Immediately before concluding your response and committing your changes, you MUST read this document, update the status table, record any new problems encountered/resolved, refresh the test counts, and check off completed items in the Roadmap.**
> 3. **Verify that `python3 -m pytest tests/` has 100% pass rate before committing.**
> 4. **Ensure all changes to this document are staged and committed alongside code updates.**
