# Gemini diagnosis and recovery

For free development, the app now supports Groq with your existing Groq key. Follow [Groq setup](GROQ_SETUP.md) to avoid the exhausted Gemini allowance. The billing steps below apply only if you choose to continue using Gemini at a higher quota; payment is not required to use the Groq integration.

## What was verified on October 29, 2026

- The configured API key reached Google's native `generativelanguage.googleapis.com` endpoint. No proxy override was active.
- `gemini-3.8-flash` metadata was available, but a tiny generation request returned HTTP 504 after about 59 seconds with both low and default thinking. Its Interactions endpoint also timed out. This isolates the observed failure to generation on that model's serving path; Google's internal reason is not visible to this client.
- With the same key and prompt, `gemini-3.6-flash` returned `OK` in 6.75 seconds. The application's actual SQL-generation path subsequently returned executable `SELECT 1 AS ok;` in 2.03 seconds.
- Google also returned HTTP 503 with an explicit high-demand message, and HTTP 429 identifying this project's free-tier limits as **5 requests per minute** and **20 requests per day** for Gemini 3.6 Flash. The daily allowance is now exhausted. Those are separate provider-capacity and quota failures, not SQL execution errors.
- The installed SDK had no HTTP timeout by default and one total SDK attempt. Retry multiplication was investigated but was **not** observed in this environment.

The application now defaults to the verified working model and a shared bounded request policy. It does not switch models silently during a benchmark.

## Required account action for the exhausted daily quota

The 20-question benchmark needs at least **38 model calls**: two for each of 18 SQL questions and one for each of two ambiguity-only questions, before retries or repairs. The observed 20-call daily allowance cannot cover a full run even after a reset. Pacing only addresses the per-minute limit.

For enough quota to complete the full benchmark:

1. Open [AI Studio Projects](https://aistudio.google.com/projects) and locate the project that owns the key in `.env`.
2. In its **Billing Tier** column, select **Set up billing**.
3. Link an existing billing account or complete Google's account setup. Follow the displayed Prepay or Postpay flow; Google's current Prepay minimum is $5 or its local equivalent.
4. Confirm the project shows a paid tier and, for Prepay, a positive credit balance. Resolve any **Set up Prepay** or **No credits** notice.
5. Open [the active project limits](https://aistudio.google.com/rate-limit?timeRange=last-28-days), select the same project/model, and verify the daily allowance can cover at least 38 calls plus retries.
6. Restart Streamlit and run the generation check below before another full benchmark.

These account/payment actions must be completed by the account owner. [Google's setup instructions](https://ai.google.dev/gemini-api/docs/billing) describe the flow.

For light free-tier use, the next documented daily reset is **October 30, 2026 at 12:30 PM IST** (midnight Pacific). It restores the small daily allowance; it does not make a 38-call run fit into 20 calls. Google applies limits per project, so another key in the same project does not raise them. See [rate-limit rules](https://ai.google.dev/gemini-api/docs/rate-limits).

Daily quota errors now fail immediately without transient retries. Classification failure stops the query assistant before SQL generation. A fatal Gemini provider error aborts a benchmark after the first affected task, exports `ABORTED_PROVIDER_ERROR` with null accuracy fields, and avoids submitting the rest of the dataset. Older recovery-attempt reports predate this fail-fast behavior; their provider-error scores are not valid clean baselines.

## Configuration

Keep the existing API key in `.env` and set:

```env
GEMINI_MODEL=gemini-3.6-flash
GEMINI_TIMEOUT_SECONDS=60
GEMINI_MAX_ATTEMPTS=3
GEMINI_THINKING_LEVEL=low
GEMINI_MAX_OUTPUT_TOKENS=4096
GEMINI_REQUESTS_PER_MINUTE=4
```

The timeout is a total budget per logical model call, including pacing and retries. SDK retries are disabled so only the shared application policy controls retries. The SDK timeout shrinks with the remaining budget. HTTP 408/429/500/502/503/504 and transport failures can retry; permanent 4xx errors fail immediately. Retry-After and Google `RetryInfo` delays are honored when they fit within the budget. Empty responses are reported as failures.

Reasoning tokens share the output-token limit. The output budget is 4096 rather than an artificially tiny probe limit. Sampling overrides such as temperature are omitted for current Gemini models.

Pacing is per process and key/model. It does not coordinate another application or process using the same Google project. Keep other Gemini clients idle during a free-tier benchmark. On a project with a higher verified limit, adjust the pacing to suit that limit; `0` disables local pacing.

## Exact verification steps on this machine

1. Open a terminal in the repository:

   ```bash
   cd "/Users/anshu/Documents/PROJECTS/DataSense AI"
   ```

2. Install the declared SDK version if using a different Python environment:

   ```bash
   /opt/homebrew/Caskroom/miniconda/base/bin/python3 -m pip install -r requirements.txt
   ```

3. Test actual generation, rather than only checking whether the model is listed:

   ```bash
   /opt/homebrew/Caskroom/miniconda/base/bin/python3 scripts/diagnose_gemini.py --output reports/gemini_health_check.json
   ```

   A working check returns `"status": "SUCCESS"` and `"response": "OK"`. It sends a fixed synthetic prompt, never business data, and uses one bounded request. A single 503 can be transient; allow the server's indicated delay and repeat the check.

4. Restart an existing Streamlit process with **Ctrl+C**, then relaunch it so the frozen settings and `.env` are reloaded:

   ```bash
   /opt/homebrew/Caskroom/miniconda/base/bin/python3 -m streamlit run src/app.py
   ```

5. After sufficient daily quota is available, stop other model-using clients and run the full benchmark:

   ```bash
   /opt/homebrew/Caskroom/miniconda/base/bin/python3 scripts/run_benchmark.py --read-only --output reports/gemini_recovery_paced_benchmark.json
   ```

   At 4 requests per minute, approximately 38 generation/classification calls take about ten minutes, plus any transient retries. The configured rate applies to all model-calling stages and retries.

## If a request still fails

| Result | Action |
| --- | --- |
| 504 on 3.8 while 3.6 works | Keep `GEMINI_MODEL=gemini-3.6-flash`; the client cannot repair Google's serving path. To check 3.8 later without changing settings, run `scripts/diagnose_gemini.py --model gemini-3.8-flash --timeout 60`. |
| 503 high demand | Wait and retry with bounded backoff; inspect the service status page. A larger timeout does not guarantee additional service capacity. |
| 429 requests per minute | Check the active project/model rate limit; keep pacing below it and stop competing clients. If the error gives a retry delay larger than the remaining budget, retry later or deliberately increase the request budget. |
| 429 tokens per minute or daily quota | Follow the specific quota metric. The observed daily limit is exhausted; use the account steps above or wait for a reset for a smaller workload. The full benchmark needs more than the free daily allowance. |
| 401/403 | Verify the selected key, its project and restrictions in AI Studio. |
| 404 | Select a model available to that project. |
| Empty output / MAX_TOKENS | Increase the output-token budget or reduce thinking effort within the model's supported settings. |

Provider references: [API errors](https://ai.google.dev/gemini-api/docs/api-errors), [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits), [active project limits](https://aistudio.google.com/rate-limit?timeRange=last-28-days), [thinking controls](https://ai.google.dev/gemini-api/docs/thinking), [service status](https://aistudio.google.com/status).
