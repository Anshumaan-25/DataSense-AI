"""Bounded Groq requests with shared request and token pacing for free-tier use."""

from __future__ import annotations

import math
import logging
import random
import time
from threading import Lock
from typing import Optional

import httpx
from openai import APIConnectionError, OpenAI

from src.config import Settings, settings
from src.llm import ProviderRequestError, TRANSIENT_CODES, _pace_request, _retry_after

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_TOKEN_LOCK = Lock()
_TOKEN_WINDOWS = {}
logger = logging.getLogger(__name__)


class GroqRequestError(ProviderRequestError):
    def __init__(self, model, attempts, code, detail, quota_kind=None):
        self.provider, self.model, self.attempts = "groq", model, attempts
        self.code, self.quota_kind = code, quota_kind
        action = {
            401: "Check GROQ_API_KEY in the Groq console.",
            403: "Check this key's project and model permissions in Groq.",
            404: "Choose a model listed for this key; run scripts/diagnose_groq.py.",
            429: "Check Groq request and token limits; wait for the reported reset.",
        }.get(code, "Check the Groq model status and network, then retry.")
        if quota_kind == "daily":
            action = "Groq daily quota exhausted. Wait for its reset before another run."
        super().__init__(f"Groq model {model!r} failed after {attempts} request(s) "
            f"(HTTP {code if code is not None else 'network/timeout'}): {detail}. {action}")


def _reserve_tokens(key, estimated, limit, deadline, clock, sleep):
    """Reserve a conservative prompt/output estimate in a rolling 60s window."""
    if not limit:
        return None
    if estimated > limit:
        raise ValueError("Prompt plus output reservation exceeds GROQ_TOKENS_PER_MINUTE; "
            "shorten the prompt/output budget or set a verified higher token limit.")
    while True:
        remaining = deadline - clock()
        if remaining <= 0 or not _TOKEN_LOCK.acquire(timeout=remaining):
            raise TimeoutError("Token pacing exceeded GROQ_TIMEOUT_SECONDS")
        try:
            now = clock()
            window = [entry for entry in _TOKEN_WINDOWS.get(key, []) if entry[0] > now - 60]
            _TOKEN_WINDOWS[key] = window
            if sum(entry[1] for entry in window) + estimated <= limit:
                reservation = [now, estimated]
                window.append(reservation)
                return reservation
            wait = min(entry[0] for entry in window) + 60 - now + 0.01
        finally:
            _TOKEN_LOCK.release()
        if wait >= deadline - clock():
            raise TimeoutError("Token pacing would exceed GROQ_TIMEOUT_SECONDS")
        sleep(wait)


def generate_groq_text(
    user_prompt: str, model_name: str, api_key: str, system_prompt: str = "",
    response_mime_type: Optional[str] = None, config: Settings = settings,
    client_factory=None, sleep_fn=None, clock_fn=None,
) -> str:
    """Call Groq without SDK retries, secret leakage, or silent provider fallback."""
    create = client_factory or OpenAI
    sleep, clock = sleep_fn or time.sleep, clock_fn or time.monotonic
    deadline = clock() + config.groq_timeout_seconds
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
    messages.append({"role": "user", "content": user_prompt})
    # This estimate includes JSON/message overhead; actual usage replaces it after success.
    estimated = math.ceil((len(user_prompt) + len(system_prompt)) / 3) + 64 + config.groq_max_output_tokens
    attempts, last_code, detail, quota_kind = 0, None, "Request deadline exceeded", None
    for attempt in range(1, config.groq_max_attempts + 1):
        try:
            reservation = _reserve_tokens((api_key, model_name), estimated,
                config.groq_tokens_per_minute, deadline, clock, sleep)
        except (ValueError, TimeoutError) as exc:
            raise GroqRequestError(model_name, attempts, last_code, f"{detail}; {exc}", quota_kind) from None
        if not _pace_request(api_key, model_name, config.groq_requests_per_minute, deadline, clock, sleep):
            detail += "; request pacing would exceed GROQ_TIMEOUT_SECONDS"
            break
        remaining = deadline - clock()
        if remaining <= 0:
            break
        if reservation is not None:
            with _TOKEN_LOCK:
                reservation[0] = clock()
        attempts = attempt
        client = None
        try:
            client = create(api_key=api_key, base_url=GROQ_BASE_URL, timeout=remaining, max_retries=0)
            options = dict(model=model_name, messages=messages, temperature=0,
                max_completion_tokens=config.groq_max_output_tokens)
            if response_mime_type == "application/json":
                options["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**options)
            usage = getattr(response, "usage", None)
            tokens = getattr(usage, "total_tokens", None)
            if reservation is not None and isinstance(tokens, int) and tokens > 0:
                with _TOKEN_LOCK:
                    reservation[1] = tokens
            choice = response.choices[0] if response.choices else None
            text = choice.message.content if choice is not None else ""
            if choice is not None and choice.finish_reason == "length":
                raise GroqRequestError(model_name, attempts, None,
                    "Output truncated; increase GROQ_MAX_OUTPUT_TOKENS within the token allowance")
            if not text or not text.strip():
                raise GroqRequestError(model_name, attempts, None, "Empty completion")
            return text
        except GroqRequestError:
            raise
        except Exception as exc:
            last_code = getattr(exc, "status_code", None)
            detail = str(exc)
            for secret in (api_key, config.groq_api_key, config.gemini_api_key, config.openai_api_key):
                if secret:
                    detail = detail.replace(secret, "<redacted>")
            detail = detail[:1200]
            quota_kind = "daily" if last_code == 429 and any(term in detail.lower()
                for term in ("requests per day", "tokens per day", "rpd", "tpd")) else None
            if quota_kind == "daily" or not (last_code in TRANSIENT_CODES or isinstance(exc, (APIConnectionError, httpx.TransportError))):
                raise GroqRequestError(model_name, attempts, last_code, detail, quota_kind) from None
            delay = max(1.5 * 2 ** (attempt - 1) + random.uniform(0, 0.5), _retry_after(exc) or 0)
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    logger.debug("Could not close Groq client", exc_info=False)
        if attempt == config.groq_max_attempts:
            break
        if delay >= deadline - clock():
            detail += f"; retry requires {delay:.1f}s, beyond the remaining request budget"
            break
        sleep(delay)
    raise GroqRequestError(model_name, attempts, last_code, detail, quota_kind) from None
