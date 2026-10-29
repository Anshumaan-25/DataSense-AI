"""Groq routing, bounded retries, quota pacing, and application integration."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pandas as pd
import pytest
from openai import APIConnectionError, RateLimitError

from src import ambiguity, generator, groq_client, semantic
from src.config import Settings
from src.llm import ProviderRequestError


class Clock:
    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def __call__(self):
        return self.now

    def sleep(self, value):
        self.sleeps.append(value)
        self.now += value


class Failure(Exception):
    def __init__(self, code, message="provider unavailable", headers=None):
        super().__init__(message)
        self.status_code = code
        self.response = SimpleNamespace(headers=headers or {})


def completion(text="OK", tokens=100, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
        usage=SimpleNamespace(total_tokens=tokens))


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.setattr(groq_client, "_TOKEN_WINDOWS", {})
    monkeypatch.setattr(groq_client.random, "uniform", lambda *a: 0)
    return Settings(llm_provider="groq", groq_api_key="private-groq-key",
        gemini_api_key="private-google-key", openai_api_key="private-openai-key",
        groq_model="qwen/qwen3.8-27b", groq_timeout_seconds=120,
        groq_requests_per_minute=0, groq_tokens_per_minute=0)


def clients_for(outcomes):
    clients = []
    iterator = iter(outcomes)

    def factory(**kwargs):
        outcome = next(iterator)
        call = MagicMock(side_effect=outcome) if isinstance(outcome, Exception) else MagicMock(return_value=outcome)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=call)), close=MagicMock())
        clients.append(client)
        return client

    return MagicMock(side_effect=factory), clients


def request(policy, outcomes, clock=None, **kwargs):
    factory, clients = clients_for(outcomes)
    clock = clock or Clock()
    result = groq_client.generate_groq_text("Return OK", policy.groq_model, policy.groq_api_key,
        config=policy, client_factory=factory, clock_fn=clock, sleep_fn=clock.sleep, **kwargs)
    return result, factory, clients, clock


def test_transport_targets_groq_without_sdk_retries_and_closes_client(policy):
    text, factory, clients, _ = request(policy, [completion()], system_prompt="rules", response_mime_type="application/json")
    assert text == "OK"
    assert factory.call_args.kwargs == {"api_key": policy.groq_api_key,
        "base_url": "https://api.groq.com/openai/v1", "timeout": 120, "max_retries": 0}
    options = clients[0].chat.completions.create.call_args.kwargs
    assert options["model"] == policy.groq_model and options["response_format"] == {"type": "json_object"}
    assert options["messages"][0] == {"role": "system", "content": "rules"}
    assert options["max_completion_tokens"] == 2048
    clients[0].close.assert_called_once()


@pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504])
def test_transient_errors_retry_with_shrinking_deadline(policy, code):
    text, factory, clients, clock = request(policy, [Failure(code), completion()])
    assert text == "OK" and clock.sleeps == [1.5]
    assert [call.kwargs["timeout"] for call in factory.call_args_list] == [120, 118.5]
    assert all(client.close.call_count == 1 for client in clients)


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_permanent_errors_do_not_retry_and_redact_all_keys(policy, code):
    factory, clients = clients_for([Failure(code, f"{policy.groq_api_key} {policy.gemini_api_key} {policy.openai_api_key}")])
    with pytest.raises(groq_client.GroqRequestError) as raised:
        groq_client.generate_groq_text("prompt", policy.groq_model, policy.groq_api_key, config=policy, client_factory=factory)
    assert isinstance(raised.value, ProviderRequestError)
    assert raised.value.code == code and raised.value.provider == "groq"
    assert all(key not in str(raised.value) for key in (policy.groq_api_key, policy.gemini_api_key, policy.openai_api_key))
    assert factory.call_count == 1 and clients[0].close.call_count == 1


def test_sdk_connection_errors_are_retryable(policy):
    failure = APIConnectionError(request=httpx.Request("POST", "https://api.groq.com"))
    assert request(policy, [failure, completion()])[0] == "OK"


def test_retry_after_is_respected(policy):
    _, _, _, clock = request(policy, [Failure(429, headers={"Retry-After": "4"}), completion()])
    assert clock.sleeps == [4]


def test_daily_quota_payload_fails_without_retry(policy):
    failure = RateLimitError("Rate limit reached on tokens per day (TPD)",
        response=httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com")),
        body={"error": {"code": "rate_limit_exceeded"}})
    factory, _ = clients_for([failure])
    with pytest.raises(groq_client.GroqRequestError) as raised:
        groq_client.generate_groq_text("prompt", policy.groq_model, policy.groq_api_key, config=policy, client_factory=factory)
    assert raised.value.quota_kind == "daily" and factory.call_count == 1


def test_long_retry_delay_does_not_exceed_total_budget(policy):
    with pytest.raises(groq_client.GroqRequestError, match="beyond the remaining"):
        request(policy, [Failure(429, headers={"Retry-After": "150"})])


def test_retry_attempt_limit_is_enforced(policy):
    factory, clients = clients_for([Failure(503)] * 3)
    clock = Clock()
    with pytest.raises(groq_client.GroqRequestError) as raised:
        groq_client.generate_groq_text("prompt", policy.groq_model, policy.groq_api_key, config=policy,
            client_factory=factory, clock_fn=clock, sleep_fn=clock.sleep)
    assert raised.value.attempts == 3 and len(clients) == 3


@pytest.mark.parametrize("response", [completion(""), completion("SELECT 1", finish="length"), SimpleNamespace(choices=[], usage=None)])
def test_empty_and_truncated_output_are_failures(policy, response):
    with pytest.raises(groq_client.GroqRequestError):
        request(policy, [response])


def test_rolling_token_pacing_uses_actual_response_usage(policy):
    policy = replace(policy, groq_tokens_per_minute=2500)
    clock = Clock()
    request(policy, [completion(tokens=2400)], clock=clock)
    _, factory, _, clock = request(policy, [completion()], clock=clock)
    assert clock.sleeps == [60.01]
    assert factory.call_args.kwargs["timeout"] == pytest.approx(59.99)


def test_token_wait_cannot_exceed_deadline(policy):
    policy = replace(policy, groq_tokens_per_minute=2500, groq_timeout_seconds=20)
    clock = Clock()
    request(policy, [completion(tokens=2400)], clock=clock)
    with pytest.raises(groq_client.GroqRequestError, match="pacing"):
        request(policy, [completion()], clock=clock)
    assert not clock.sleeps


def test_oversized_token_reservation_fails_before_api_call(policy):
    policy = replace(policy, groq_tokens_per_minute=100)
    factory = MagicMock()
    with pytest.raises(groq_client.GroqRequestError, match="shorten the prompt"):
        groq_client.generate_groq_text("prompt", policy.groq_model, policy.groq_api_key, config=policy, client_factory=factory)
    factory.assert_not_called()


def test_request_pacing_is_shared_between_calls(policy, monkeypatch):
    from src import llm
    monkeypatch.setattr(llm, "_LAST_REQUEST", {})
    policy = replace(policy, groq_requests_per_minute=6)
    clock = Clock()
    request(policy, [completion()], clock=clock)
    _, factory, _, clock = request(policy, [completion()], clock=clock)
    assert clock.sleeps == [10]
    assert factory.call_args.kwargs["timeout"] == 110


def test_cleanup_failure_preserves_the_provider_error(policy):
    factory, clients = clients_for([Failure(401)])
    def create(**kwargs):
        client = factory(**kwargs)
        client.close.side_effect = RuntimeError("cleanup failed")
        return client
    with pytest.raises(groq_client.GroqRequestError) as raised:
        groq_client.generate_groq_text("prompt", policy.groq_model, policy.groq_api_key,
            config=policy, client_factory=create)
    assert raised.value.code == 401 and clients[0].close.call_count == 1


@pytest.mark.parametrize("repair", [False, True])
def test_explicit_groq_without_key_never_uses_another_provider(policy, monkeypatch, repair):
    monkeypatch.setattr(generator, "settings", replace(policy, groq_api_key=""))
    call = MagicMock()
    monkeypatch.setattr(generator, "_call_gemini", call)
    monkeypatch.setattr(generator, "_call_openai", call)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        if repair:
            generator.generate_repaired_sql("question", "schema", provider="groq")
        else:
            generator.generate_sql("question", "schema", provider="groq")
    call.assert_not_called()


@pytest.mark.parametrize("stage,payload", [("generation", "SELECT 1"), ("repair", "SELECT 1"),
    ("classification", '{"ambiguity_type":"CLEAR"}'), ("judge", '{"is_valid":true}')])
def test_all_model_stages_use_groq_and_json_mode_where_needed(policy, monkeypatch, stage, payload):
    for module in (generator, ambiguity, semantic):
        monkeypatch.setattr(module, "settings", policy)
    factory, clients = clients_for([completion(payload)])
    monkeypatch.setattr(groq_client, "OpenAI", factory)
    if stage == "generation":
        assert generator.generate_sql("How many?", "schema") == "SELECT 1"
    elif stage == "repair":
        assert generator.generate_repaired_sql("How many?", "schema", failed_sql="bad SQL", error_message="syntax") == "SELECT 1"
    elif stage == "classification":
        assert ambiguity.AmbiguityClassifier().classify_intent("How many?", "schema").classification_succeeded
    else:
        result = semantic.SemanticValidator(enable_llm_judge=True).validate_output(
            "How many?", "SELECT COUNT(*)", pd.DataFrame({"count": [1]}))
        assert result.judge_status == "PASSED"
    options = clients[0].chat.completions.create.call_args.kwargs
    assert options["model"] == policy.groq_model
    assert ("response_format" in options) == (stage in {"classification", "judge"})


def test_missing_groq_key_stops_classification_cleanly(policy, monkeypatch):
    monkeypatch.setattr(ambiguity, "settings", replace(policy, groq_api_key=""))
    result = ambiguity.AmbiguityClassifier().classify_intent("question", "schema", provider="groq")
    assert not result.classification_succeeded and "GROQ_API_KEY" in result.reasoning


@pytest.mark.parametrize("field,value", [("groq_timeout_seconds", 0), ("groq_timeout_seconds", float("nan")),
    ("groq_max_attempts", 0), ("groq_max_output_tokens", 0), ("groq_requests_per_minute", -1),
    ("groq_requests_per_minute", float("inf")), ("groq_tokens_per_minute", -1)])
def test_invalid_groq_limits_are_rejected(policy, field, value):
    with pytest.raises(ValueError):
        replace(policy, **{field: value})


def test_explicit_provider_selection_and_auto_preference(policy):
    assert policy.get_effective_provider() == "groq"
    assert replace(policy, groq_api_key="").get_effective_provider() == "none"
    assert replace(policy, llm_provider="gemini").get_effective_provider() == "gemini"
    assert replace(policy, llm_provider="openai").get_effective_provider() == "openai"
    assert replace(policy, llm_provider="auto").get_effective_provider() == "groq"
