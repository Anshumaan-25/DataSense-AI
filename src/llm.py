"""One bounded Gemini request policy shared by all model-calling stages."""

from __future__ import annotations

import logging
import math
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from threading import Lock
from typing import Callable, Optional

import httpx
from google import genai
from google.genai import types

from src.config import Settings, settings

logger = logging.getLogger(__name__)
TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}
_RATE_LOCK = Lock()
_LAST_REQUEST = {}


def _quota_kind(exc) -> Optional[str]:
    payload = getattr(exc, "details", {})
    error = payload.get("error", payload) if isinstance(payload, dict) else {}
    details = error.get("details", []) if isinstance(error, dict) else []
    kind = None
    for detail in details if isinstance(details, list) else []:
        if not isinstance(detail, dict) or not str(detail.get("@type", "")).endswith("google.rpc.QuotaFailure"):
            continue
        violations = detail.get("violations", [])
        for violation in violations if isinstance(violations, list) else []:
            identifier = str(violation.get("quotaId", "")).lower() if isinstance(violation, dict) else ""
            if "perday" in identifier:
                return "daily"
            if "perminute" in identifier:
                kind = "minute"
    return kind


def _retry_after(exc) -> Optional[float]:
    """Read HTTP Retry-After and google.rpc.RetryInfo without parsing error prose."""
    values = []
    headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
    header = headers.get("Retry-After") or headers.get("retry-after")
    if header is not None:
        try:
            values.append(float(header))
        except (ValueError, TypeError):
            try:
                timestamp = parsedate_to_datetime(str(header))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                values.append((timestamp - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
    payload = getattr(exc, "details", {})
    if isinstance(payload, dict):
        error = payload.get("error", payload)
        details = error.get("details", []) if isinstance(error, dict) else []
        for detail in details if isinstance(details, list) else []:
            if not isinstance(detail, dict) or not str(detail.get("@type", "")).endswith("google.rpc.RetryInfo"):
                continue
            duration = detail.get("retryDelay")
            try:
                if isinstance(duration, str) and duration.endswith("s"):
                    values.append(float(duration[:-1]))
                elif isinstance(duration, dict):
                    values.append(float(duration.get("seconds", 0)) + float(duration.get("nanos", 0)) / 1e9)
            except (ValueError, TypeError):
                pass
    valid = [value for value in values if math.isfinite(value) and value >= 0]
    return max(valid) if valid else None


def _pace_request(api_key, model_name, rpm, deadline, clock, sleep) -> bool:
    if rpm == 0:
        return True
    remaining = deadline - clock()
    if remaining <= 0 or not _RATE_LOCK.acquire(timeout=remaining):
        return False
    try:
        now = clock()
        previous = _LAST_REQUEST.get((api_key, model_name))
        wait = max(0, previous + 60 / rpm - now) if previous is not None else 0
        if wait >= deadline - now:
            return False
        if wait:
            sleep(wait)
        _LAST_REQUEST[(api_key, model_name)] = clock()
        return True
    finally:
        _RATE_LOCK.release()


class ProviderRequestError(RuntimeError):
    """A fatal model-provider failure, distinct from a SQL or parsing failure."""


class GeminiRequestError(ProviderRequestError):
    """Redacted provider failure with actionable diagnostics and request count."""

    def __init__(self, model: str, attempts: int, code: Optional[int], detail: str, quota_kind: Optional[str] = None):
        self.provider = "gemini"
        self.model, self.attempts, self.code = model, attempts, code
        self.quota_kind = quota_kind
        action = {
            400: "Check the model's supported request parameters.",
            401: "Check GEMINI_API_KEY in Google AI Studio.",
            403: "Check API-key restrictions, project access, region, and billing.",
            404: "Choose a model available to this key; run scripts/diagnose_gemini.py.",
            429: "Check the project's Gemini rate limits and quota in Google AI Studio.",
            504: "Retry later or select a working GEMINI_MODEL; increase GEMINI_TIMEOUT_SECONDS only if necessary.",
        }.get(code, "Check the Gemini service status and network, then retry.")
        if quota_kind == "daily":
            action = "Daily project/model quota exhausted. Wait for midnight Pacific or review the project's usage tier in Google AI Studio. Another key in the same project does not raise this quota."
        super().__init__(f"Gemini model {model!r} failed after {attempts} request(s)"
                         f" (HTTP {code if code is not None else 'network/timeout'}): {detail}. {action}")


def generate_gemini_text(
    user_prompt: str,
    model_name: str,
    api_key: str,
    system_prompt: str = "",
    response_mime_type: Optional[str] = None,
    config: Settings = settings,
    client_factory: Optional[Callable] = None,
    sleep_fn: Optional[Callable] = None,
    clock_fn: Optional[Callable] = None,
) -> str:
    """Use one total deadline and bounded jittered retries, never layered retries.

    Timeout milliseconds passed to each SDK call shrink with the remaining budget.
    Permanent 4xx errors are not retried. Reasoning effort and output length are
    explicit; sampling overrides are omitted for current Gemini models.
    """
    create_client = client_factory or genai.Client
    sleep = sleep_fn or time.sleep
    clock = clock_fn or time.monotonic
    deadline = clock() + config.gemini_timeout_seconds
    generation = {
        "system_instruction": system_prompt or None,
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        "max_output_tokens": config.gemini_max_output_tokens,
    }
    if config.gemini_thinking_level != "auto":
        generation["thinking_config"] = types.ThinkingConfig(thinking_level=config.gemini_thinking_level)
    if response_mime_type:
        generation["response_mime_type"] = response_mime_type
    generation_config = types.GenerateContentConfig(**generation)
    last_detail, last_code, attempts = "Request deadline exceeded", None, 0
    last_quota_kind = None

    for attempt in range(1, config.gemini_max_attempts + 1):
        if not _pace_request(api_key, model_name, config.gemini_requests_per_minute, deadline, clock, sleep):
            last_detail += "; request pacing would exceed the total deadline"
            break
        remaining = deadline - clock()
        if remaining <= 0:
            break
        attempts = attempt
        client = None
        try:
            client = create_client(api_key=api_key, http_options=types.HttpOptions(
                timeout=max(1, math.floor(remaining * 1000)),
                retry_options=types.HttpRetryOptions(attempts=1),
            ))
            response = client.models.generate_content(
                model=model_name, contents=user_prompt, config=generation_config,
            )
            text = response.text or ""
            if not text.strip():
                reason = str(response.candidates[0].finish_reason) if getattr(response, "candidates", None) else "no candidates"
                raise GeminiRequestError(model_name, attempts, None,
                    f"Empty response ({reason}); check output-token/thinking limits and safety feedback")
            return text
        except GeminiRequestError:
            raise
        except Exception as exc:
            code = getattr(exc, "code", None)
            last_code = int(code) if isinstance(code, (int, str)) and str(code).isdigit() else None
            last_detail = str(exc)
            for secret in (api_key, config.gemini_api_key, config.openai_api_key, config.groq_api_key):
                if secret:
                    last_detail = last_detail.replace(secret, "<redacted>")
            last_detail = last_detail[:1200]
            retry_hint = _retry_after(exc)
            last_quota_kind = _quota_kind(exc)
            if last_quota_kind == "daily":
                raise GeminiRequestError(model_name, attempts, last_code, last_detail, quota_kind="daily") from None
            transient = last_code in TRANSIENT_CODES or isinstance(exc, httpx.TransportError)
            if not transient:
                raise GeminiRequestError(model_name, attempts, last_code, last_detail) from None
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    logger.debug("Could not close Gemini client", exc_info=False)
        remaining = deadline - clock()
        if attempt >= config.gemini_max_attempts or remaining <= 0:
            break
        delay = 1.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
        if retry_hint is not None:
            delay = max(delay, retry_hint)
        if delay >= remaining:
            last_detail += f"; retry requires waiting {delay:.1f}s, beyond the remaining request budget"
            break
        logger.warning("Gemini %s returned %s; retry %s/%s in %.1fs", model_name, last_code, attempt + 1, config.gemini_max_attempts, delay)
        sleep(delay)

    raise GeminiRequestError(model_name, attempts, last_code, last_detail, quota_kind=last_quota_kind) from None
