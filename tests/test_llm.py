"""Gemini request policy tests with deterministic clocks and injected clients."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pandas as pd
import pytest

from src import llm
from src.config import Settings


class ProviderFailure(Exception):
    def __init__(self, code, message="provider failure"):
        super().__init__(message)
        self.code = code


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.setattr(llm.random, "uniform", lambda *args: 0)
    return replace(Settings(), gemini_api_key="private-test-key", openai_api_key="private-other-key",
        gemini_model="gemini-3.6-flash", gemini_timeout_seconds=10,
        gemini_max_attempts=3, gemini_thinking_level="low", gemini_max_output_tokens=4096,
        gemini_requests_per_minute=0)


def clients_for(outcomes):
    iterator = iter(outcomes)
    clients, options = [], []

    def create(**kwargs):
        options.append(kwargs["http_options"])
        outcome = next(iterator)
        client = SimpleNamespace(models=SimpleNamespace(generate_content=MagicMock(side_effect=
            outcome if isinstance(outcome, Exception) else lambda **kw: outcome)), close=MagicMock())
        clients.append(client)
        return client

    return create, clients, options


def request(policy, outcomes, **kwargs):
    create, clients, options = clients_for(outcomes)
    clock = kwargs.pop("clock", Clock())
    result = llm.generate_gemini_text("prompt", model_name=policy.gemini_model, api_key=policy.gemini_api_key,
        config=policy, client_factory=create, sleep_fn=clock.sleep, clock_fn=clock, **kwargs)
    return result, clients, options, clock


def test_shared_request_disables_sdk_retries_and_sets_explicit_generation_options(policy):
    result, clients, options, _ = request(policy, [SimpleNamespace(text="OK")],
        system_prompt="system rules", response_mime_type="application/json")
    assert result == "OK"
    assert options[0].timeout == 10000
    assert options[0].retry_options.attempts == 1
    kwargs = clients[0].models.generate_content.call_args.kwargs
    assert kwargs["contents"] == "prompt"
    assert kwargs["config"].system_instruction == "system rules"
    assert kwargs["config"].automatic_function_calling.disable is True
    assert kwargs["config"].response_mime_type == "application/json"
    assert kwargs["config"].max_output_tokens == 4096
    assert str(kwargs["config"].thinking_config.thinking_level).lower().endswith("low")
    assert kwargs["config"].temperature is None
    clients[0].close.assert_called_once()


@pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504])
def test_transient_errors_retry_with_a_shrinking_total_deadline(policy, code):
    result, clients, options, clock = request(policy, [ProviderFailure(code), SimpleNamespace(text="OK")])
    assert result == "OK"
    assert clock.sleeps == [1.5]
    assert [option.timeout for option in options] == [10000, 8500]
    assert len(clients) == 2 and all(client.close.call_count == 1 for client in clients)


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_permanent_errors_are_not_retried_and_secrets_are_redacted(policy, code):
    create, clients, _ = clients_for([ProviderFailure(code, f"failure: {policy.gemini_api_key} {policy.openai_api_key}")])
    clock = Clock()
    with pytest.raises(llm.GeminiRequestError) as raised:
        llm.generate_gemini_text("prompt", model_name=policy.gemini_model, api_key=policy.gemini_api_key,
            config=policy, client_factory=create, sleep_fn=clock.sleep, clock_fn=clock)
    assert raised.value.code == code and raised.value.attempts == 1
    assert policy.gemini_api_key not in str(raised.value)
    assert policy.openai_api_key not in str(raised.value)
    assert "<redacted>" in str(raised.value)
    assert not clock.sleeps
    clients[0].close.assert_called_once()


def test_transport_timeouts_are_retryable(policy):
    result, clients, _, _ = request(policy, [httpx.ReadTimeout("read timed out"), SimpleNamespace(text="OK")])
    assert result == "OK" and len(clients) == 2


def test_retries_stop_at_the_configured_attempt_count(policy):
    create, clients, _ = clients_for([ProviderFailure(503)] * 3)
    clock = Clock()
    with pytest.raises(llm.GeminiRequestError) as raised:
        llm.generate_gemini_text("prompt", model_name=policy.gemini_model, api_key=policy.gemini_api_key,
            config=policy, client_factory=create, sleep_fn=clock.sleep, clock_fn=clock)
    assert raised.value.attempts == 3 and len(clients) == 3
    assert clock.sleeps == [1.5, 3.0]


def test_a_consumed_deadline_does_not_start_another_request(policy):
    clock = Clock()
    client = SimpleNamespace(models=SimpleNamespace(), close=MagicMock())

    def timeout(**kwargs):
        clock.now = 10
        raise ProviderFailure(504)

    client.models.generate_content = MagicMock(side_effect=timeout)
    factory = MagicMock(return_value=client)
    with pytest.raises(llm.GeminiRequestError) as raised:
        llm.generate_gemini_text("prompt", model_name=policy.gemini_model, api_key=policy.gemini_api_key,
            config=policy, client_factory=factory, sleep_fn=clock.sleep, clock_fn=clock)
    assert raised.value.attempts == 1
    assert not clock.sleeps
    factory.assert_called_once()
    assert "GEMINI_TIMEOUT_SECONDS" in str(raised.value)


def test_backoff_cannot_exceed_the_remaining_budget(policy):
    policy = replace(policy, gemini_timeout_seconds=1)
    create, clients, _ = clients_for([ProviderFailure(503)])
    clock = Clock()
    with pytest.raises(llm.GeminiRequestError):
        llm.generate_gemini_text("prompt", model_name=policy.gemini_model, api_key=policy.gemini_api_key,
            config=policy, client_factory=create, sleep_fn=clock.sleep, clock_fn=clock)
    assert len(clients) == 1 and not clock.sleeps


def test_empty_or_truncated_output_is_not_a_success(policy):
    response = SimpleNamespace(text="", candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")])
    create, clients, _ = clients_for([response])
    with pytest.raises(llm.GeminiRequestError, match="MAX_TOKENS"):
        llm.generate_gemini_text("prompt", model_name=policy.gemini_model, api_key=policy.gemini_api_key,
            config=policy, client_factory=create)
    clients[0].close.assert_called_once()


def test_auto_thinking_preserves_the_model_default(policy):
    _, clients, _, _ = request(replace(policy, gemini_thinking_level="auto"), [SimpleNamespace(text="OK")])
    assert clients[0].models.generate_content.call_args.kwargs["config"].thinking_config is None


@pytest.mark.parametrize("field,value", [
    ("gemini_timeout_seconds", 0), ("gemini_timeout_seconds", -1),
    ("gemini_timeout_seconds", float("nan")), ("gemini_timeout_seconds", float("inf")),
    ("gemini_max_attempts", 0), ("gemini_max_output_tokens", 0),
    ("gemini_thinking_level", "unsupported"),
    ("gemini_requests_per_minute", -1), ("gemini_requests_per_minute", float("nan")),
])
def test_invalid_request_configuration_is_rejected(policy, field, value):
    with pytest.raises(ValueError):
        replace(policy, **{field: value})


@pytest.mark.parametrize("stage,payload", [
    ("generation", "SELECT 1 AS ok"),
    ("ambiguity", '{"ambiguity_type":"CLEAR"}'),
    ("semantic", '{"is_valid":true}'),
])
def test_every_gemini_stage_uses_the_same_bounded_transport(policy, monkeypatch, stage, payload):
    from src import ambiguity, generator, semantic
    create, clients, options = clients_for([SimpleNamespace(text=payload)])
    monkeypatch.setattr(llm.genai, "Client", create)
    for module in (ambiguity, generator, semantic):
        monkeypatch.setattr(module, "settings", policy)
    if stage == "generation":
        assert generator._call_gemini("prompt", policy.gemini_model, policy.gemini_api_key) == payload
    elif stage == "ambiguity":
        assert ambiguity.AmbiguityClassifier().classify_intent("Show one", "schema", provider="gemini").classification_succeeded
    else:
        result = semantic.SemanticValidator(enable_llm_judge=True).validate_output(
            "How many?", "SELECT COUNT(*)", pd.DataFrame({"count": [1]}), provider="gemini")
        assert result.judge_status == "PASSED"
    assert 0 < options[0].timeout <= 10000
    assert options[0].retry_options.attempts == 1
    clients[0].close.assert_called_once()


@pytest.mark.parametrize("kind", ["http_header", "rpc_duration", "rpc_seconds_nanos"])
def test_provider_retry_delay_is_honored(policy, kind):
    failure = ProviderFailure(429)
    if kind == "http_header":
        failure.response = SimpleNamespace(headers={"Retry-After": "4"})
    else:
        duration = "4s" if kind == "rpc_duration" else {"seconds": 3, "nanos": 1_000_000_000}
        failure.details = {"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": duration}]}}
    result, _, options, clock = request(policy, [failure, SimpleNamespace(text="OK")])
    assert result == "OK" and clock.sleeps == [4]
    assert options[1].timeout == 6000


def test_long_provider_backoff_is_reported_instead_of_violating_the_deadline(policy):
    failure = ProviderFailure(429)
    failure.response = SimpleNamespace(headers={"Retry-After": "120"})
    create, clients, _ = clients_for([failure])
    clock = Clock()
    with pytest.raises(llm.GeminiRequestError, match="waiting 120.0s"):
        llm.generate_gemini_text("prompt", model_name=policy.gemini_model, api_key=policy.gemini_api_key,
            config=policy, client_factory=create, sleep_fn=clock.sleep, clock_fn=clock)
    assert len(clients) == 1 and not clock.sleeps


def test_malformed_retry_details_do_not_mask_the_provider_error(policy):
    failure = ProviderFailure(503)
    failure.details = {"error": {"details": None}}
    result, _, _, clock = request(policy, [failure, SimpleNamespace(text="OK")])
    assert result == "OK" and clock.sleeps == [1.5]


def test_optional_pacing_spaces_requests_and_uses_the_remaining_budget(policy, monkeypatch):
    monkeypatch.setattr(llm, "_LAST_REQUEST", {})
    policy = replace(policy, gemini_requests_per_minute=12)
    clock = Clock()
    request(policy, [SimpleNamespace(text="OK")], clock=clock)
    _, _, options, _ = request(policy, [SimpleNamespace(text="OK")], clock=clock)
    assert clock.sleeps == [5]
    assert options[0].timeout == 5000


def test_pacing_cannot_wait_beyond_the_total_deadline(policy, monkeypatch):
    monkeypatch.setattr(llm, "_LAST_REQUEST", {(policy.gemini_api_key, policy.gemini_model): 0})
    policy = replace(policy, gemini_requests_per_minute=1)
    create = MagicMock()
    clock = Clock()
    with pytest.raises(llm.GeminiRequestError, match="pacing would exceed"):
        llm.generate_gemini_text("prompt", model_name=policy.gemini_model, api_key=policy.gemini_api_key,
            config=policy, client_factory=create, sleep_fn=clock.sleep, clock_fn=clock)
    create.assert_not_called()
    assert not clock.sleeps


def test_daily_quota_is_detected_from_structured_google_errors_without_retrying(policy):
    from google.genai.errors import ClientError
    failure = ClientError(429, {"error": {"message": "Daily quota exhausted", "details": [{
        "@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{
            "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
        }],
    }]}})
    create, clients, _ = clients_for([failure])
    clock = Clock()
    with pytest.raises(llm.GeminiRequestError) as raised:
        llm.generate_gemini_text("prompt", model_name=policy.gemini_model, api_key=policy.gemini_api_key,
            config=policy, client_factory=create, sleep_fn=clock.sleep, clock_fn=clock)
    assert raised.value.quota_kind == "daily"
    assert raised.value.attempts == 1 and not clock.sleeps
    assert "midnight Pacific" in str(raised.value)
    clients[0].close.assert_called_once()


def test_daily_quota_takes_priority_when_a_minute_limit_is_also_exceeded():
    failure = ProviderFailure(429)
    failure.details = {"error": {"details": [{
        "@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
            {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"},
            {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"},
        ],
    }]}}
    assert llm._quota_kind(failure) == "daily"
