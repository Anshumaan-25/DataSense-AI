# DataSense AI ⚡

Production-grade Text-to-SQL system powered by **DuckDB**, **ChromaDB Schema-Aware RAG**, **TPC-H analytical benchmark data**, **AST query safety validation via `sqlglot`**, and multi-provider LLM support (**Google Gemini & OpenAI**) wrapped in an interactive **Streamlit** user interface.

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
│     • Execution Success → Return PipelineResult        │
└────────────────────────────────────────────────────────┘
       │
       ▼
 [ Interactive Streamlit UI (app.py) ]
  • Interactive Clarification Card (Radio + Default)
  • Clarified Intent Badges & Formula cards
  • Attempt Status Badges (First-Try vs. Auto-Repaired)
  • Collapsible Self-Correction & Repair History
  • Interactive DataFrame & Execution Telemetry
```

---

## 📁 Project Structure

```
DataSense AI/
├── data/
│   ├── tpch.duckdb          # Persisted DuckDB instance with TPC-H benchmark tables
│   └── metrics.yaml         # Business metric dictionary (aggregation formulas & filter rules)
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Environment variables, provider settings, and paths
│   ├── database.py          # DuckDB connection manager, TPC-H generator & schema extractor
│   ├── validator.py         # sqlglot-based AST query safety validator & file-read blocker
│   ├── retriever.py         # ChromaDB semantic vector search for metrics & schema
│   ├── ambiguity.py         # 3-Tier intent ambiguity classifier & clarification agent
│   ├── generator.py         # LLM prompt construction with RAG context & repair reflection
│   ├── pipeline.py          # Bounded self-correction repair state machine (N ≤ 3)
│   └── app.py               # Modern Streamlit web app with interactive clarification modal
├── tests/
│   ├── __init__.py
│   ├── test_database.py     # Tests for TPC-H setup, schema extractor, and query execution
│   ├── test_generator.py    # Tests for SQL extraction and error handling
│   ├── test_retriever.py    # Tests for ChromaDB indexing, semantic search & thresholding
│   ├── test_validator.py    # Tests for AST safety validation, file blocking & whitelisting
│   ├── test_repair.py       # Tests for self-correction loop, compiler hints & security abort
│   └── test_ambiguity.py    # Tests for 3-tier taxonomy, fallback parser & clarification flow
├── .env.example             # Template for API keys and configuration
├── requirements.txt         # Core dependencies (duckdb, chromadb, sqlglot, streamlit, pyyaml)
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone & Set Up Virtual Environment

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
# Provider choice: "auto", "gemini", or "openai"
LLM_PROVIDER=auto

# Option A: Google Gemini (Recommended)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Option B: OpenAI
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# DuckDB Settings
DUCKDB_PATH=data/tpch.duckdb
TPCH_SCALE_FACTOR=0.1
```

### 3. Launch Streamlit Application

```bash
streamlit run src/app.py
```

The database (`data/tpch.duckdb`) is automatically initialized with TPC-H tables (`sf=0.1`) on first run, and ChromaDB indexes the metric catalog into memory.

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

All 63 tests cover:
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
- Semantic RAG retrieval for compound business questions ("What is the net revenue for customers in Germany?").
- Composite join retrieval for item margin and urgent priority.
- Semantic distance thresholding on irrelevant queries.
- SQL code fence extraction from LLM outputs.
- Self-correction state machine first-attempt success path.
- Self-correction on DuckDB runtime typos (`c_namee` -> `c_name` via DuckDB compiler hints).
- Self-correction on sqlglot AST parse errors.
- Strict security violation immediate fatal abort (zero retries on `DROP`, `DELETE`, `read_csv`).
- Hard loop cap enforcement ($N \le 3$) on persistent errors.
- Loop stagnation & identical query cycle detection.
- 3-Tier intent ambiguity taxonomy classification (`CLEAR`, `SOFT_DEFAULT`, `HARD_AMBIGUITY`).
- Resilient JSON parsing and graceful fallback on malformed model responses.
- Interactive user clarification persistence across the self-correction repair loop.

---

## 💡 Example Analytical Queries

Try asking the assistant:
- *"What is the net revenue for customers in Germany?"* *(Triggers `net_revenue` formula and `german_customers` filter)*
- *"Calculate the item margin for urgent orders."* *(Triggers `item_margin` composite join and `urgent_orders` filter)*
- *"What is the total discount loss on lineitems?"* *(Triggers `discount_loss` formula)*
- *"What are the top 5 customers by total account balance?"*
- *"SELECT * FROM read_csv('/etc/passwd');"* *(Trigger AST file-read rejection)*
- *"DROP TABLE customer;"* *(Trigger AST DDL rejection)*
