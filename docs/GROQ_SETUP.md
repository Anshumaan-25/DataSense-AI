# Groq free-tier setup

Groq is connected to SQL generation, repair, ambiguity classification, and the optional semantic judge. The current key authenticated successfully, and `qwen/qwen3.8-27b` was listed for it. A synthetic JSON-generation check succeeded in 0.57 seconds. Live application checks classified a count question, executed the generated SQL with a result of 25 nations, repaired a missing-table query, and passed the semantic judge. Evidence is in `reports/groq_health_check.json` and `reports/groq_pipeline_smoke.json`.

The full RAG-enabled benchmark completed all 20 tasks with zero evaluation/provider errors in 601.0 seconds: 94.44% Pass@1/Pass@N (17/18 SQL tasks) and 100% ambiguity accuracy. The one miss selected supplier names while the reference also selects supplier key and account balance; the dataset/scoring policy was unchanged. See `reports/groq_free_tier_benchmark.json`. Streamlit startup passed without exceptions and confirmed Groq in the sidebar and provider selector (`reports/groq_ui_smoke.json`).

## Configuration

Keep the key in the ignored `.env` file, never in source control. The local configuration now selects:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_actual_key
GROQ_MODEL=qwen/qwen3.8-27b
GROQ_TIMEOUT_SECONDS=120
GROQ_MAX_ATTEMPTS=3
GROQ_MAX_OUTPUT_TOKENS=2048
GROQ_REQUESTS_PER_MINUTE=6
GROQ_TOKENS_PER_MINUTE=7000
```

The SDK is already installed in this machine's Conda environment; `requirements.txt` records the tested minimum `openai>=3.18.0`. Groq uses the SDK with its own endpoint and key. No OpenAI account or key is needed for Groq.

An explicit provider selection is respected even when another key is present. `auto` prefers a configured Groq key, then Gemini, then OpenAI. Requests do not silently switch models or providers. The sidebar and benchmark selector display Groq, and exports record the actual provider/model.

## Limits and request behavior

Groq's published free-tier table lists Qwen at 30 requests/minute, 1,000 requests/day, 8,000 tokens/minute, and 200,000 tokens/day. Confirm your actual organization limits in [Groq settings](https://console.groq.com/settings/limits); any token or request cap can stop a run. See [official rate-limit documentation](https://console.groq.com/docs/rate-limits). These published allowances are not a guarantee of capacity or SQL accuracy.

The application uses conservative local caps of 6 requests/minute and 7,000 tokens in a rolling 60-second window. Before a request, it reserves a prompt estimate (characters divided by three, with message overhead) plus the configured output allowance. Successful responses replace that reservation with actual total-token usage. Pauses of up to about a minute are expected during a benchmark. The estimate is approximate; server rate limits remain authoritative.

Request/token pacing is shared across stages and threads in one process, per key/model. It does not coordinate other keys or processes using the same Groq organization. Keep other model-using clients idle during a free-tier benchmark. Set either local limit to `0` to disable that specific limiter only when your actual quota permits it. If one prompt exceeds the configured reservation budget, shorten its context/output budget or configure a verified higher token allowance.

Each logical call has a 120-second budget including pacing and retries, up to three application attempts, one SDK attempt, and server-directed retry delays when they fit. Permanent request errors and explicit daily-quota errors stop immediately. Empty or truncated responses are failures. Clients are closed and configured keys are redacted from reported errors.

Classification failure prevents dependent SQL generation. A fatal Groq/Gemini provider error stops the benchmark and exports `ABORTED_PROVIDER_ERROR` with null accuracy fields. The optional judge retains its existing visible `ERROR` fallback behavior if judging is unavailable; a failed judge is never displayed as passed.

## Run on this machine

From the repository directory:

```bash
cd "/Users/anshu/Documents/PROJECTS/DataSense AI"
/opt/homebrew/Caskroom/miniconda/base/bin/python3 scripts/diagnose_groq.py --output reports/groq_health_check.json
/opt/homebrew/Caskroom/miniconda/base/bin/python3 -m streamlit run src/app.py
```

If Streamlit is already running, stop it with Ctrl+C before relaunching to reload `.env` and cached settings.

To run the full RAG-enabled benchmark against the existing database:

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python3 scripts/run_benchmark.py --provider groq --read-only --output reports/groq_free_tier_benchmark.json
```

The 20 tasks require at least 38 calls before repairs, and token pacing can make the run take several minutes. Report accuracy only after the final JSON confirms completion and shows its error count. Gold results never feed back into repairs.

For a 401, check the key in [Groq Console](https://console.groq.com/keys). For a 404, choose an available model. For a 429, inspect whether the exhausted limit is requests or tokens, per minute or per day; the app cannot increase that allowance. The free integration requires no Gemini billing setup.

References: [Groq OpenAI compatibility](https://console.groq.com/docs/openai), [models](https://console.groq.com/docs/models), [JSON mode](https://console.groq.com/docs/structured-outputs), [rate limits](https://console.groq.com/docs/rate-limits).
