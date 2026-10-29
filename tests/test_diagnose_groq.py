"""The Groq diagnostic checks inference once and returns actionable status."""

import json
import sys
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from scripts import diagnose_groq as doctor
from src.groq_client import GroqRequestError


@pytest.mark.parametrize("response,status", [('{"ok":true}', "SUCCESS"), ('{"ok":false}', "UNEXPECTED_RESPONSE"), ("invalid", "FAILED")])
def test_probe_uses_one_synthetic_json_request(monkeypatch, response, status):
    monkeypatch.setattr(doctor, "settings", replace(doctor.settings, groq_api_key="test"))
    call = MagicMock(return_value=response)
    monkeypatch.setattr(doctor, "generate_groq_text", call)
    report = doctor.check_generation(model="probe", timeout=12)
    assert report["status"] == status and report["business_data_sent"] is False
    assert call.call_args.kwargs["config"].groq_max_attempts == 1
    assert call.call_args.kwargs["config"].groq_timeout_seconds == 12
    assert call.call_args.kwargs["response_mime_type"] == "application/json"


def test_missing_key_does_not_make_an_api_call(monkeypatch):
    monkeypatch.setattr(doctor, "settings", replace(doctor.settings, groq_api_key=""))
    call = MagicMock()
    monkeypatch.setattr(doctor, "generate_groq_text", call)
    assert doctor.check_generation()["status"] == "FAILED"
    call.assert_not_called()


def test_failed_probe_writes_diagnostic_and_exits_nonzero(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor, "settings", replace(doctor.settings, groq_api_key="test"))
    monkeypatch.setattr(doctor, "generate_groq_text", MagicMock(side_effect=GroqRequestError("model", 1, 429, "limited")))
    output = tmp_path / "health.json"
    monkeypatch.setattr(sys, "argv", ["doctor", "--output", str(output)])
    assert doctor.main() == 1
    assert json.loads(output.read_text())["code"] == 429
