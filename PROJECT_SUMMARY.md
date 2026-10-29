# DataSense AI — Comprehensive System Summary, Technical Audit & Future Roadmap

> **Document Version**: 1.4.0<br>
> **Last Updated**: October 29, 2026<br>
> **Status**: Phases 1–6, Tier 1, correctness/evaluation, and free Groq integration verified (325/325 tests passing); live benchmark completed with zero provider/evaluation errors, 94.44% EX, 100% ambiguity accuracy; Streamlit startup passed<br>
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
7. **Shared Gemini Request Policy** (`src/llm.py`): verified `gemini-3.6-flash` default, one request deadline, bounded transient retries, provider retry-delay hints, and configurable request pacing across generation, classification, and semantic judging.
8. **Free Groq Integration** (`src/groq_client.py`): `qwen/qwen3.8-27b` across generation, repair, classification, and optional judging; shared request/token pacing, explicit provider selection, redacted bounded failures, and a one-request inference diagnostic. The local `.env` selects Groq without changing credentials or requiring Gemini billing.

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
| **Phase 2** | `src/retriever.py`<br>`src/generator.py`<br>`src/llm.py`<br>`src/groq_client.py`<br>`data/metrics.yaml` | Persistent retrieval, sample-aware generation/repair, bounded Gemini/Groq requests, and Groq request/token pacing across model stages. | 101 tests (`test_retriever.py`, `test_generator.py`, `test_llm.py`, `test_groq.py`) | ✅ Complete |
| **Phase 3** | `src/pipeline.py` | Bounded repair with actual per-attempt outcomes, preserved intent, selective samples, positive attempt-budget validation, stagnation detection, and immediate security abort. | 20 tests (`test_repair.py`) | ✅ Complete |
| **Phase 4** | `src/ambiguity.py` | 3-tier taxonomy, retrieval-grounded classification, explicit fallback status, deterministic browse defaults, structured clarification UI. | 17 tests (`test_ambiguity.py`) | ✅ Complete |
| **Phase 5** | `src/semantic.py` | Dynamic casing; requested-metric AST checks with alias/CTE/derived-column support; explicit judge status; remaining deterministic heuristics. | 45 tests (`test_semantic.py`) | ✅ Complete |
| **Phase 6** | `src/evaluator.py`<br>`scripts/run_benchmark.py`<br>`scripts/diagnose_gemini.py`<br>`scripts/diagnose_groq.py`<br>`data/benchmark_queries.json` | One-trajectory EX, explicit comparison policy, safe gold preflight, deterministic top-N references, provider diagnostics, and abort-aware reports/CLI. | 93 tests (`test_evaluator.py`, `test_benchmark_cli.py`, `test_diagnose_gemini.py`, `test_diagnose_groq.py`) | ✅ Verified offline and with a completed Groq live baseline |
| **Total** | **All Modules** | **Automated unit and integration suite** | **325 / 325 Passing** | ✅ **Verified offline** |

**Latest offline verification:** The final pre-commit run passed all 325 tests in 11.70 seconds using the existing Conda Python 3.13.13 environment. The Groq integration adds 49 tests to the 276-test Gemini recovery milestone. Automated provider calls are mocked; persistent storage is also tested across a separate Python process. Live Groq stage checks are in `reports/groq_health_check.json` and `reports/groq_pipeline_smoke.json`. Streamlit AppTest startup passed with zero exceptions and confirmed the Groq sidebar/model and provider selector (`reports/groq_ui_smoke.json`). The original Gemini failure evidence remains preserved.

**Latest live baseline:** `reports/groq_free_tier_benchmark.json` completed all 20 tasks with RAG enabled on `qwen/qwen3.8-27b`, exit 0, and zero evaluation/provider errors. Pass@1 and Pass@N were both **94.44% (17/18 EX-eligible tasks)**; ambiguity classification was **100% (20/20)**. Wall time was 601.0s, including free-tier pacing; average task latency was 30.04s. No repairs occurred in that run, so its measured repair lift is 0; the separate live smoke check verifies repair and optional judging. The single miss (`tpch_17`) returned supplier names with the correct Germany filter, while the reference also selects supplier key and account balance. The benchmark and comparator were unchanged; this is one 20-question baseline, not a generalization claim.

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

### 4.14. Original Live Benchmark Provider Deadline Failure — Historical
- **Observed on October 29, 2026**: The configured `gemini-3.8-flash` metadata request succeeded. A bounded generation probe returned `504 DEADLINE_EXCEEDED`; the full CLI run stalled generating SQL for its first task and was interrupted (exit 130).
- **Original outcome**: Zero completed task results and no live EX score for that attempt. Its status artifact retains null accuracy fields and the retry command. Request deadlines and retry controls have since been implemented in Section 4.15. Durable partial-run persistence remains future work.

### 4.15. Gemini Model Serving Failure, Unlimited Timeout, and Free-Tier Quotas
- **Verified diagnosis**: The same key and native API endpoint could read model metadata, but `gemini-3.8-flash` generation returned 504 after 58.72s with low thinking and 58.66s with default thinking. Its Interactions API also timed out after 45.39s. The same tiny prompt succeeded on `gemini-3.6-flash` in 6.75s; actual SQL generation then returned executable `SELECT 1 AS ok;` in 2.03s. This isolates the observed failure to 3.8's generation serving path; Google's internal backend cause is not exposed to this client.
- **Additional provider evidence**: Google explicitly reported transient high demand (503) on a 3.6 request. The first fast bulk run also returned 429 with quota ID `GenerateRequestsPerMinutePerProjectPerModel-FreeTier` and **limit 5 requests/minute** for this project/model. That run completed with 14 evaluation/provider errors and is retained as `reports/gemini_recovery_benchmark.json`; its scores are not a clean model-quality baseline.
- **Gemini account limit**: The paced retry exposed `GenerateRequestsPerDayPerProjectPerModel-FreeTier` with **limit 20 requests/day**, now exhausted. That attempt is retained in `reports/gemini_recovery_paced_benchmark.json`; it finished all 20 tasks with provider errors before the new abort guard was added. Its zero scores are not a model-quality baseline. The 20-question workload requires at least 38 classification/generation calls before retries or repairs, so a daily reset alone cannot cover a full run. The next documented reset is October 30, 2026 at 12:30 PM IST (midnight Pacific). A full Gemini run needs sufficient project/model quota; the user instead chose the free Groq alternative in Section 4.16. No more live Gemini requests were made after identifying its exhausted daily quota.
- **Local defect**: The installed SDK's effective HTTP timeout was `None`. Its effective maximum SDK attempts was one, so nested SDK retries were investigated but were not observed here. The old application retry paths were duplicated and did not explicitly cover 504.
- **Fix**: Default and local `.env` model changed to the verified 3.6 endpoint. `src/llm.py` now handles generation, ambiguity, and semantic-judge requests with a 60s total logical budget; shrinking SDK timeouts; one SDK attempt; up to three application attempts; exponential backoff/jitter; HTTP Retry-After and Google RetryInfo delays; guaranteed client cleanup; explicit low thinking and 4096 output tokens; no sampling overrides or unused automatic function calling. Config validation rejects invalid limits, permanent 4xx errors fail immediately, and errors redact configured keys.
- **Quota control**: Default pacing is 4 requests/minute, below the observed 5 RPM allowance, and includes retries. Pacing shares state across stages/threads within one process and key/model; it does not coordinate other processes/projects. Request waits cannot exceed the remaining logical budget. `GEMINI_REQUESTS_PER_MINUTE=0` disables pacing for an appropriately provisioned project.
- **Failure propagation and measurement**: Structured daily-quota failures bypass transient retries. Failed classification stops the assistant before SQL generation. A fatal Gemini provider failure aborts a benchmark after its first affected task, retains the dataset's EX eligibility count, and exports `ABORTED_PROVIDER_ERROR`, completed-task count, and null accuracy/delta fields. The CLI exits 2 and shows accuracy as unmeasured; the UI also hides accuracy cards and preserves stopped progress. Offline tests cover both classification and generation failures, native SDK quota payloads, and the CLI export path.
- **Operator tools**: `scripts/diagnose_gemini.py` verifies actual generation with one synthetic bounded request, optional model/timeout overrides, JSON output, and a useful exit status. `docs/GEMINI_TROUBLESHOOTING.md` documents the diagnosis and exact configuration, restart, generation-check, quota, and benchmark steps. The declared SDK minimum is 2.25.0 (Python 3.10+); testing used SDK 2.25.0 on Python 3.13.13. Credentials were preserved in the ignored `.env` file.

### 4.16. Free Groq Provider Integration and Token-Aware Pacing
- **Constraint and access check**: The user requires free operation and supplied a Groq key. Authentication and model listing succeeded without exposing the key. The old Llama example is not in the key's available text-model list; the configured model is the available `qwen/qwen3.8-27b`.
- **Integration**: Native Groq requests use the installed OpenAI-compatible SDK, with Groq's endpoint/key and no OpenAI credentials. Generation, repair, ambiguity JSON, and optional semantic-judge JSON use the same transport. Explicit provider selection is respected; `auto` prefers a configured Groq key, then Gemini, then OpenAI. Neither requests nor benchmarks silently switch providers/models. CLI/UI provider choices and report metadata include Groq.
- **Free-tier pacing**: Groq's published Qwen limits are 30 RPM, 1,000 RPD, 8,000 TPM, and 200,000 TPD, subject to the organization's actual limits. Local defaults cap requests at 6 RPM and reserve up to 7,000 tokens in a rolling 60s window, using a character-based prompt estimate plus the output budget and replacing successful reservations with actual usage. This is per process/key/model and does not coordinate other clients in the same organization. Token estimates are approximate; server quota errors remain authoritative.
- **Failure policy**: A 120s logical request budget includes pacing and retries, with at most three application attempts and zero SDK retries. Transient HTTP/connection failures use bounded backoff and server retry hints. Explicit daily-quota and permanent errors fail immediately. Empty/truncated completions fail, clients close, and errors redact all configured keys. Typed provider failures retain the existing benchmark-abort/null-accuracy behavior for Groq as well as Gemini. Failed classification prevents dependent SQL generation.
- **Live evidence**: `scripts/diagnose_groq.py` returned valid JSON in 0.57s. Actual application classification returned CLEAR; generated SQL counted 25 nations; repair corrected a missing-table query to `SELECT COUNT(*) FROM nation`; the optional semantic judge passed. `reports/groq_pipeline_smoke.json` consolidates captured live observations and an independent local SQL recheck after correcting result-wrapper mistakes in the one-off reporting script. The full RAG-enabled 20-task run completed in 601.0s with no provider/evaluation errors, 94.44% Pass@1/Pass@N, and 100% classification accuracy. Streamlit startup also passed; its UI evidence is in `reports/groq_ui_smoke.json`.
- **Operator setup**: The ignored `.env` selects Groq and retains every API key. `docs/GROQ_SETUP.md` includes exact setup, restart, diagnostic, and benchmark commands. The SDK minimum is the tested `openai>=3.18.0`; no separate Groq SDK is needed. The full offline suite passes 325 tests.
- **Session completion**: The local Streamlit server was stopped at the user's request before committing; port 8501 was confirmed closed. The configuration remains ready for the next launch. API keys remain in the ignored `.env`; changed source, documentation, and diagnostic artifacts passed a configured-key scan before publication.

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
   - Gemini and OpenAI structured-output requests can occasionally yield extra text outside JSON fences. Fallback regex parsers are active in `ambiguity.py` and `semantic.py`, but native Pydantic schema validation should be reinforced.
5. **Semantic Judge and Benchmark Measurement Scope**:
   - The LLM judge remains disabled by default. UI messages now distinguish a passed judge from disabled, unavailable, and failed verification. Enabled-judge exceptions retain the existing permissive behavior but are labeled `ERROR`, never a successful judge verdict. AST formula checks establish structural agreement, not general SQL algebra equivalence or join correctness. Token telemetry remains future work.
6. **Repair Samples**:
   - Samples are the first up to three rows returned by DuckDB without an explicit ordering guarantee; they may not include the value that caused a conversion failure. Current pipeline authorization remains the TPC-H table whitelist. Authorized sample values are sent to the configured LLM provider during type/conversion repair.
7. **Evaluation and Provider Limits**:
   - EX supports scalar cells; nested/list-valued outputs raise an evaluation error. Ambiguous column permutations and tolerant row matching can be expensive for large result sets; current coverage targets the TPC-H suite. One run over 20 questions is not a generalization study. Interrupted benchmark runs are not checkpointed. Request policies cannot repair provider capacity or raise quotas. Groq's local token estimates and per-process pacing cannot coordinate other organization clients or guarantee remaining daily tokens; use the documented provider controls.

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
- [x] Complete the live 20-question baseline — Groq completed all 20 tasks with zero evaluation/provider errors, 94.44% Pass@1/Pass@N, and 100% ambiguity classification accuracy. Historical Gemini failure evidence is preserved.

### Gemini Runtime Recovery — October 29, 2026
- [x] Isolate model-specific 504s with same-key/model/endpoint probes and confirm a working model.
- [x] Unify all Gemini calls with validated deadlines, retry limits, reasoning/output controls, cleanup, and redacted errors.
- [x] Honor server-directed retry delays and pace below the observed free-tier request limit.
- [x] Fail immediately on daily-quota exhaustion; stop dependent generation and abort provider-failed benchmarks with unmeasured accuracy.
- [x] Add a reusable inference diagnostic and exact operator troubleshooting steps.
- [x] Full automated regression suite: 276/276 passing.
- [x] Restore live operation through a free Groq alternative; the Gemini-specific quota remains unchanged. Historical evidence remains in `reports/gemini_failure_diagnosis.json`.

### Free Groq Development — October 29, 2026
- [x] Authenticate the supplied key and select an available free-tier model.
- [x] Integrate generation, repair, classification, and optional judging with bounded request/token pacing.
- [x] Expose provider choice in configuration, CLI/UI, report metadata, and inference diagnostics.
- [x] Verify all live model stages and the 325-test offline suite.
- [x] Complete and inspect the RAG-enabled 20-question benchmark and verify Streamlit startup. The single EX mismatch is a supplier output-column difference; the dataset/scoring policy remain unchanged.

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

# 3. Run Complete Automated Test Suite (325 Tests)
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

Free Groq development settings (the current local provider; restart Streamlit after `.env` changes):
```env
LLM_PROVIDER=groq
GROQ_MODEL=qwen/qwen3.8-27b
GROQ_TIMEOUT_SECONDS=120
GROQ_MAX_ATTEMPTS=3
GROQ_MAX_OUTPUT_TOKENS=2048
GROQ_REQUESTS_PER_MINUTE=6
GROQ_TOKENS_PER_MINUTE=7000
```

Gemini recovery settings for an explicitly selected Gemini run:
```env
GEMINI_MODEL=gemini-3.6-flash
GEMINI_TIMEOUT_SECONDS=60
GEMINI_MAX_ATTEMPTS=3
GEMINI_THINKING_LEVEL=low
GEMINI_MAX_OUTPUT_TOKENS=4096
GEMINI_REQUESTS_PER_MINUTE=4
```

Actual generation health check (one synthetic request):
```bash
python3 scripts/diagnose_groq.py --output reports/groq_health_check.json
# For an explicitly selected Gemini model:
python3 scripts/diagnose_gemini.py --output reports/gemini_health_check.json
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
python3 scripts/run_benchmark.py --provider groq --read-only --output reports/groq_free_tier_benchmark.json
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
