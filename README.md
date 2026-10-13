# DataSense AI ⚡

**Phase 1**: Production-grade Text-to-SQL system powered by DuckDB, TPC-H analytical benchmark data, AST query safety validation via `sqlglot`, and multi-provider LLM support (Google Gemini & OpenAI) wrapped in an interactive Streamlit UI.

---


## 🏗️ Architecture

```
[ User Question ] + [ DuckDB TPC-H Schema ]
                   │
                   ▼
       [ LLM Generator (Gemini / OpenAI) ]
                   │
                   ▼  (Raw SQL)
       [ AST Safety Validator (sqlglot) ]
          │                           │
   (Valid SELECT)               (Prohibited / DDL / DML)
          │                           │
          ▼                           ▼
  [ DuckDB Execution ]         [ Safety Alert ]
          │                    (DROP/DELETE/UPDATE blocked)
          ▼
 [ Interactive Results ]
  (DataFrame + Latency)
```

---

## 📁 Project Structure

```
DataSense AI/
├── data/
│   └── tpch.duckdb          # Persisted DuckDB instance with TPC-H benchmark tables
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Environment variables, provider settings, and paths
│   ├── database.py          # DuckDB connection manager, TPC-H generator & schema extractor
│   ├── validator.py         # sqlglot-based AST query safety validator
│   ├── generator.py         # LLM prompt construction & SQL extraction (Gemini / OpenAI)
│   └── app.py               # Modern Streamlit web application
├── tests/
│   ├── __init__.py
│   ├── test_database.py     # Tests for TPC-H setup, schema extractor, and query execution
│   ├── test_generator.py    # Tests for SQL extraction and error handling
│   └── test_validator.py    # Tests for AST safety validation and injection blocking
├── .env.example             # Template for API keys and configuration
├── requirements.txt         # Core dependencies
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

The database (`data/tpch.duckdb`) will be automatically initialized with TPC-H tables (`sf=0.1`) on first run.

---

## 🛡️ Multi-Tier Security Architecture

To prevent SQL injection, data tampering, accidental database mutation, and **Local File Inclusion (LFI) / Arbitrary File Read**, DataSense AI implements a zero-trust, two-tier defense:

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

All 41 tests cover:
- TPC-H table generation and idempotency.
- Schema extraction (dictionary and prompt-ready DDL string).
- Valid query execution and query error handling.
- DuckDB C++ engine-level file access denial (`enable_external_access=false`).
- AST validator acceptance of valid SELECTs, CTEs, and UNIONs.
- AST validator rejection of destructive operations and chained multi-statement queries.
- AST validator rejection of file-reading functions (`read_csv`, `read_parquet`, `read_text`, `glob`).
- AST validator rejection of direct file path replacement scans (`'data/*.parquet'`, `'/etc/passwd'`).
- AST validator table whitelisting and CTE resolution.
- SQL code fence extraction from LLM outputs.

---

## 💡 Example Analytical Queries

Try asking the assistant:
- *"What are the top 5 customers by total account balance?"*
- *"What is the total revenue grouped by order priority?"*
- *"List the top 5 nations by customer count."*
- *"Which 5 suppliers have the highest available parts quantity?"*
- *"SELECT * FROM read_csv('/etc/passwd');"* *(Trigger AST file-read rejection)*
- *"DROP TABLE customer;"* *(Trigger AST DDL rejection)*
