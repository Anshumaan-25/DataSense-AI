# DataSense AI ⚡

Production-grade Text-to-SQL system powered by **DuckDB**, **ChromaDB Schema-Aware RAG**, **TPC-H analytical benchmark data**, **AST query safety validation via `sqlglot`**, and multi-provider LLM support (**Groq, Google Gemini & OpenAI**) wrapped in an interactive **Streamlit** user interface.

---

## 🏗️ Architecture

```
[ User Question ]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ ChromaDB Semantic RAG Retriever (retriever.py)         │
│  • metrics_catalog (net_revenue, item_margin, etc.)     │
│  • filters_catalog (german_customers, urgent_orders)   │
│  • schema_catalog (table definitions & columns)        │
└────────────────────────────────────────────────────────┘
       │
       ▼  (Retrieved Formulas & Business Rules)
┌────────────────────────────────────────────────────────┐
│ Ambiguity Classifier & Clarification Agent             │
│ (src/ambiguity.py)                                     │
│  • 3-Tier Taxonomy: CLEAR | SOFT_DEFAULT | HARD_AMBIG  │
│  • Domain-grounded options from metrics catalog        │
│  • ⚠️ HARD_AMBIGUITY ──► Streamlit Interactive Choice  │
│  • CLEAR / SOFT_DEFAULT ──► Auto-proceed with default  │
└────────────────────────────────────────────────────────┘
       │
       ▼  (Preserved Clarified Intent & Formulas)
┌────────────────────────────────────────────────────────┐
│ Bounded Self-Correction State Machine (pipeline.py)    │
│  Attempt Loop (N ≤ 3):                                 │
│                                                        │
│  1. LLM Generator (generator.py)                       │
│     • Initial generation / Reflection on compiler hints │
│     • Strict adherence to clarified user intent        │
│                                                        │
│  2. AST Safety Validator (validator.py)                │
│     • Prohibited DDL/DML & file-read checks            │
│     • ⚠️ Security violation → Immediate Fatal Abort    │
│     • Benign syntax error → Loop to Repair             │
│                                                        │
│  3. DuckDB Analytics Engine (database.py)              │
│     • In-process analytical execution                  │
│     • Runtime Binder/Catalog error → Loop to Repair    │
│                                                        │
│  4. Fast-Path 2-Stage Semantic Validator (semantic.py) │
│     • Stage 1: Deterministic Heuristics (<1ms)         │
│       - Zero-row & casing mismatch check               │
│       - Cartesian join & row explosion check           │
│       - Metric formula omission detection              │
│     • Stage 2: LLM Semantic Judge                      │
│       - Output shape & column alignment verification   │
│       - Non-conforming results → Loop to Repair        │
│     • Execution Success → Return PipelineResult        │
└────────────────────────────────────────────────────────┘
       │
       ▼
 ┌───────────────────────────────────────────────────────┐
 │ Streamlit UI & Evaluation Dashboard (app.py)          │
 │  • Tab 1: 💬 Query Assistant                          │
 │    - Interactive Clarification Card                   │
 │    - Self-Correction & Repair History Telemetry       │
 │    - Interactive DataFrame & Export                   │
 │  • Tab 2: 📊 Evaluation Benchmark                    │
 │    - Execution Accuracy (EX) Benchmark Suite         │
 │    - Pass@1 vs. Repaired Pass@N Comparison            │
 │    - Latency & Per-Attempt Outcome Breakdown          │
 └───────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
DataSense AI/
├── data/
│   ├── tpch.duckdb              # Persisted DuckDB instance with TPC-H benchmark tables
│   ├── metrics.yaml             # Business metric dictionary (aggregation formulas & filter rules)
│   ├── benchmark_queries.json   # 20 diverse TPC-H evaluation benchmark queries
│   └── chroma_storage/          # Generated persistent vector index (git-ignored)
├── scripts/
│   ├── run_benchmark.py         # Standalone CLI evaluation & benchmark suite
│   ├── diagnose_gemini.py       # Bounded Gemini inference health check
│   └── diagnose_groq.py         # One-request Groq JSON inference health check
├── src/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Environment variables, provider settings, and paths
│   ├── database.py              # DuckDB connection manager, TPC-H generator & schema extractor
│   ├── validator.py             # sqlglot-based AST query safety validator & file-read blocker
│   ├── retriever.py             # ChromaDB semantic vector search for metrics & schema
│   ├── ambiguity.py             # 3-Tier intent ambiguity classifier & clarification agent
│   ├── generator.py             # LLM prompt construction with RAG context & repair reflection
│   ├── llm.py                   # Shared Gemini deadlines, retries, thinking, and request pacing
│   ├── groq_client.py           # Groq deadlines, retries, request/token pacing, and redacted errors
│   ├── semantic.py              # 2-Stage silent execution & semantic validation engine
│   ├── pipeline.py              # Bounded self-correction repair state machine (N ≤ 3)
│   ├── evaluator.py             # DataFrame normalizer, EX comparator & benchmark harness
│   └── app.py                   # Streamlit app: Query Assistant + Evaluation Benchmark tabs
├── tests/
│   ├── __init__.py
│   ├── test_database.py         # Tests for TPC-H setup, schema extractor, and query execution
│   ├── test_generator.py        # Tests for SQL extraction and error handling
│   ├── test_retriever.py        # Tests for ChromaDB indexing, semantic search & thresholding
│   ├── test_validator.py        # Tests for AST safety validation, file blocking & whitelisting
│   ├── test_repair.py           # Tests for self-correction loop, compiler hints & security abort
│   ├── test_ambiguity.py        # Tests for 3-tier taxonomy, fallback parser & clarification flow
│   ├── test_semantic.py         # Tests for zero-row, casing, cartesian & metric omission checks
│   └── test_evaluator.py        # Tests for dataframe normalization, EX metrics & benchmark runner
├── .env.example                 # Template for API keys and configuration
├── requirements.txt             # Core dependencies (duckdb, chromadb, sqlglot, streamlit, pyyaml)
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone & Set Up Virtual Environment

Use Python 3.10 or newer; the current SDK and test suite were verified on Python 3.13.13.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to configure your preferred LLM provider:

```env
# Provider choice: "auto", "groq", "gemini", or "openai"
LLM_PROVIDER=groq

# Option A: Groq (recommended for free development)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
GROQ_TIMEOUT_SECONDS=120
GROQ_MAX_ATTEMPTS=3
GROQ_MAX_OUTPUT_TOKENS=2048
GROQ_REQUESTS_PER_MINUTE=6
GROQ_TOKENS_PER_MINUTE=7000

# Option B: Google Gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash
GEMINI_TIMEOUT_SECONDS=60
GEMINI_MAX_ATTEMPTS=3
GEMINI_THINKING_LEVEL=low
GEMINI_MAX_OUTPUT_TOKENS=4096
GEMINI_REQUESTS_PER_MINUTE=4

# Option C: OpenAI
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# DuckDB Settings
DUCKDB_PATH=data/tpch.duckdb
TPCH_SCALE_FACTOR=0.1

# Persistent vector index (relative to project root, or an absolute path)
CHROMA_STORAGE_PATH=data/chroma_storage
```

### 3. Launch Streamlit Application

```bash
streamlit run src/app.py
```

The database (`data/tpch.duckdb`) is automatically initialized with TPC-H tables (`sf=0.1`) on first run. ChromaDB persists metrics, filters, and schema in `data/chroma_storage`. Content hashes reuse unchanged embeddings across restarts; changed entries are reindexed and removed entries are pruned. Set `CHROMA_STORAGE_PATH` to use another directory. Relative paths resolve from the project root.

Initialization also profiles low-cardinality text columns after DuckDB external access is disabled. The semantic validator uses actual column values for casing corrections, including mixed-case values outside TPC-H. Profiles contain at most 100 distinct non-null values per column and omit strings longer than 256 characters. Refresh them with `profile_distinct_values(conn)` after external data changes. Standalone validators accept `db_conn` or an explicit `distinct_value_profile`; the pipeline supplies its active connection automatically.

When execution fails with a conversion or type error, the next repair prompt includes up to three sample rows from each referenced authorized table. Long text values are truncated to 256 characters plus a marker. Sampling preserves the SQL safety checks and does not run for ordinary syntax errors or security rejections. Sample values are provided to the configured LLM as data examples along with business rules, clarified intent, and error history.

---

## 📚 Business Metric Catalog (`data/metrics.yaml`)

Domain knowledge is formally cataloged to ensure exact calculations:

| Metric / Rule | Type | Formula / Condition | Target Tables & Join |
|---|---|---|---|
| **Net Revenue** | Aggregation | `SUM(l_extendedprice * (1 - l_discount))` | `lineitem` |
| **Gross Revenue** | Aggregation | `SUM(l_extendedprice)` | `lineitem` |
| **Discount Loss** | Aggregation | `SUM(l_extendedprice * l_discount)` | `lineitem` |
| **Tax Amount** | Aggregation | `SUM(l_extendedprice * (1 - l_discount) * l_tax)` | `lineitem` |
| **Item Margin** | Aggregation | `SUM(l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity)` | `lineitem -> partsupp` (composite join on partkey & suppkey) |
| **Urgent Orders** | Filter | `o_orderpriority = '1-URGENT'` | `orders` |
| **German Customers**| Filter | `n_name = 'GERMANY'` | `customer -> nation` (`c_nationkey = n_nationkey`) |

---

## 🛡️ Multi-Tier Security Architecture

DataSense AI implements a zero-trust, two-tier defense against SQL injection, data corruption, and **Local File Inclusion (LFI) / Arbitrary File Read**:

### Layer 1: AST Safety Validator (`sqlglot`)
All queries are analyzed before reaching the database:
1. **Dialect-Aware Parsing**: Queries are parsed into abstract syntax trees using the `duckdb` dialect.
2. **Single-Statement Rule**: Multi-statement chains (e.g. `SELECT 1; DROP TABLE customer;`) are immediately rejected.
3. **Read-Only Root**: The root AST expression must be a `Select`, `Union`, or CTE (`WITH ... SELECT`).
4. **Deep Prohibited Token Traversal**: Any occurrence of `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `CREATE`, `PRAGMA`, `ATTACH`, `COPY`, or `TRUNCATE` is blocked.
5. **File-Reading Prevention**: Blocks native read operations (`exp.ReadCSV`, `exp.ReadParquet`) and functions (`read_csv_auto`, `read_json`, `read_text`, `read_blob`, `glob`, etc.).
6. **Replacement Scan & Path Detection**: Rejects any table identifier containing file path separators (`/`, `\`) or extensions (`.parquet`, `.csv`, `.json`, etc.).
7. **Table Whitelist**: Ensures queries only reference authorized database tables or declared CTE aliases.

### Layer 2: DuckDB Engine-Level Sandboxing
As a fail-safe against parser bypasses, the DuckDB connection is locked down:
- `SET enable_external_access = false;` is enforced immediately after TPC-H benchmark initialization.
- DuckDB's C++ core physically aborts all local filesystem operations and network socket calls.
- Once disabled during a session, external access **cannot be re-enabled** via SQL (`Cannot enable external access while database is running`).

---

## 🧪 Running Automated Tests

Run the full pytest suite:

```bash
pytest tests/ -v
```

All 325 tests cover:
- TPC-H table generation and idempotency.
- Schema extraction (dictionary and prompt-ready DDL string).
- Valid query execution and query error handling.
- DuckDB C++ engine-level file access denial (`enable_external_access=false`).
- AST validator acceptance of valid SELECTs, CTEs, and UNIONs.
- AST validator rejection of destructive operations and chained multi-statement queries.
- AST validator rejection of file-reading functions (`read_csv`, `read_parquet`, `read_text`, `glob`).
- AST validator rejection of direct file path replacement scans (`'data/*.parquet'`, `'/etc/passwd'`).
- AST validator table whitelisting and CTE resolution.
- ChromaDB indexing counts for metrics, filters, and schema collections.
- Persistent embedding reuse across Python process restarts, incremental formula/schema updates, deleted entries, interrupted updates, and missing-entry recovery.
- Connection-specific distinct-value profiling, high-cardinality exclusion, refresh behavior, quoted identifiers, and read-only connections.
- Semantic RAG retrieval for compound business questions ("What is the net revenue for customers in Germany?").
- Composite join retrieval for item margin and urgent priority.
- Semantic distance thresholding on irrelevant queries.
- SQL code fence extraction from LLM outputs.
- Self-correction state machine first-attempt success path.
- Self-correction on DuckDB runtime typos (`c_namee` -> `c_name` via DuckDB compiler hints).
- Self-correction on sqlglot AST parse errors.
- Strict security violation immediate fatal abort (zero retries on `DROP`, `DELETE`, `read_csv`).
- Configured attempt-budget enforcement (three attempts by default) on persistent errors.
- Loop stagnation & identical query cycle detection.
- 3-Tier intent ambiguity taxonomy classification (`CLEAR`, `SOFT_DEFAULT`, `HARD_AMBIGUITY`).
- Resilient JSON parsing and graceful fallback on malformed model responses.
- Interactive user clarification persistence across the self-correction repair loop.
- Semantic validation: column-aware casing repair from profiled values, including aliases, IN predicates, escaped literals, case collisions, and NULL scalar aggregates.
- Conversion/type repair samples: bounded rows, physical-table/CTE resolution, security boundaries, sampling failure fallback, and Groq/Gemini/OpenAI prompt integration.
- Semantic validation: multi-table row explosion detection (more than 1,000 rows without grouping or a limit).
- Semantic validation: requested catalog formulas compared as ASTs, including aliases, ROUND/CAST wrappers, commutative operands, CTE/derived output columns, and rejection of formula decoys outside the output.
- Actual first-attempt telemetry, preserved ambiguity defaults, safe gold-query preflight, classifier error accounting, and CLI integration/error exits.
- Shared Gemini transport: shrinking deadlines, capped transient retries, permanent-error handling, retry-delay hints, quota pacing, immediate daily-quota failure, client cleanup, secret redaction, and generation/diagnostic CLI integration.
- Groq integration: all model stages, native JSON mode, provider selection, request and rolling token pacing, retry deadlines, daily quotas, cleanup, secret redaction, and the diagnostic/benchmark CLI.
- Provider-aware evaluation: failed classification prevents dependent generation; fatal Gemini/Groq errors abort the run and export null accuracy fields rather than a misleading quality score.
- EX comparison: output aliases and column order are ignored using one global mapping; complete rows and duplicate multiplicity must match. Original numeric values use absolute tolerance 0.01 and zero relative tolerance. Nulls match nulls; string case is preserved. A task can set `order_sensitive: true` to require row order. Strict column-name or case-insensitive string comparison are explicit comparator options.
- Benchmark suite: Pass@1 and Pass@N observe the same initial SQL and repair trajectory. Gold outputs never drive repairs. Exports include the protocol, first SQL/error, provider/model, dataset hash, comparison policy, latency, and classifier/error status. Token telemetry is not currently recorded.

The legacy `DataFrameNormalizer` display utility is not used for EX scoring. The Streamlit semantic indicator states whether the optional LLM judge actually ran and passed.

Run a live benchmark against an existing database with:

```bash
python3 scripts/run_benchmark.py --provider groq --read-only --output reports/groq_free_tier_benchmark.json
```

CLI exit codes are 0 for completion, 1 for setup/reference errors, and 2 for evaluation/provider errors, including an aborted run. Top-N references include deterministic tie rules; SOFT_DEFAULT order browsing uses order date descending, then order key descending, with LIMIT 10.

For free development, configure **Groq** with `qwen/qwen3.8-27b`. The key was verified with real JSON generation and live classification, SQL execution, repair, and semantic judging. All stages share bounded retries and request/token pacing. The current local `.env` explicitly selects Groq. In `auto` mode, a configured Groq key takes precedence, followed by Gemini and OpenAI. A request never silently falls back to another provider/model. See [Groq setup and verification](docs/GROQ_SETUP.md).

The October 29 Groq baseline completed **20/20 tasks with zero evaluation/provider errors**: **94.44% Pass@1 and Pass@N (17/18 SQL tasks)** and **100% ambiguity classification accuracy**. The RAG-enabled run took 601 seconds, including free-tier pacing. One supplier lookup selected names only while the reference also requires supplier key and account balance. No repair occurred in this baseline; repair and optional judging were separately verified live. See `reports/groq_free_tier_benchmark.json` and `reports/groq_ui_smoke.json`. These results describe one small benchmark, not general SQL accuracy.

The original October 29 attempts on Gemini 3.8 timed out; the same key worked with Gemini 3.6, which then exhausted its 20-call daily allowance. This model-specific Gemini quota does not prevent using Groq's free tier. The full benchmark needs at least 38 calls before retries. Historical Gemini reports with provider errors remain diagnostic evidence rather than clean quality measurements; see `reports/gemini_failure_diagnosis.json`.

For a bounded, synthetic generation health check:

```bash
python3 scripts/diagnose_groq.py --output reports/groq_health_check.json
```

Restart Streamlit after changing `.env` to reload the frozen settings. Groq pacing is per process and key/model; other applications or keys in the same organization share the provider limits. The [Gemini troubleshooting guide](docs/GEMINI_TROUBLESHOOTING.md) remains available for explicitly selected Gemini runs.

---

## 💡 Example Analytical Queries

Try asking the assistant:
- *"What is the net revenue for customers in Germany?"* *(Triggers `net_revenue` formula and `german_customers` filter)*
- *"Calculate the item margin for urgent orders."* *(Triggers `item_margin` composite join and `urgent_orders` filter)*
- *"What is the total discount loss on lineitems?"* *(Triggers `discount_loss` formula)*
- *"What are the top 5 customers by total account balance?"*
- *"SELECT * FROM read_csv('/etc/passwd');"* *(Trigger AST file-read rejection)*
- *"DROP TABLE customer;"* *(Trigger AST DDL rejection)*

:)
