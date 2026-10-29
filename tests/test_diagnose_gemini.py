"""Generation checks must exercise inference, remain bounded, and omit secrets."""

from dataclasses import replace
from unittest.mock import MagicMock
import json
import sys

import pytest

from scripts import diagnose_gemini as doctor
from src.llm import GeminiRequestError


@pytest.mark.parametrize("response,status", [("OK", "SUCCESS"), ("other", "UNEXPECTED_RESPONSE")])
def test_generation_probe_uses_one_request_without_changing_the_config(monkeypatch, response, status):
    monkeypatch.setattr(doctor, "settings", replace(doctor.settings, gemini_api_key="test"))
    generate = MagicMock(return_value=response)
    monkeypatch.setattr(doctor, "generate_gemini_text", generate)
    report = doctor.check_generation(model="alternate", timeout=12)
    assert report["status"] == status and report["business_data_sent"] is False
    options = generate.call_args.kwargs
    assert options["config"].gemini_max_attempts == 1
    assert options["config"].gemini_timeout_seconds == 12
    assert options["model_name"] == "alternate"
    assert doctor.settings.gemini_model != "alternate"


def test_probe_records_provider_failure_instead_of_claiming_healthy_metadata(monkeypatch):
    monkeypatch.setattr(doctor, "settings", replace(doctor.settings, gemini_api_key="test"))
    monkeypatch.setattr(doctor, "generate_gemini_text", MagicMock(side_effect=GeminiRequestError("model", 1, 504, "deadline")))
    report = doctor.check_generation()
    assert report["status"] == "FAILED" and report["code"] == 504
    assert "response" not in report


def test_probe_with_no_key_does_not_send_a_request(monkeypatch):
    monkeypatch.setattr(doctor, "settings", replace(doctor.settings, gemini_api_key=""))
    generate = MagicMock()
    monkeypatch.setattr(doctor, "generate_gemini_text", generate)
    assert doctor.check_generation()["status"] == "FAILED"
    generate.assert_not_called()


@pytest.mark.parametrize("status,exit_code", [("SUCCESS", 0), ("FAILED", 1)])
def test_diagnostic_cli_saves_json_and_returns_a_useful_exit_code(tmp_path, monkeypatch, status, exit_code):
    output = tmp_path / "nested" / "health.json"
    check = MagicMock(return_value={"status": status, "model": "probe-model"})
    monkeypatch.setattr(doctor, "check_generation", check)
    monkeypatch.setattr(sys, "argv", ["diagnose_gemini.py", "--model", "probe-model", "--timeout", "12", "--output", str(output)])
    assert doctor.main() == exit_code
    assert json.loads(output.read_text())["status"] == status
    check.assert_called_once_with(model="probe-model", timeout=12)


@pytest.mark.parametrize("timeout", ["0", "-1", "nan"])
def test_diagnostic_cli_rejects_invalid_deadlines_before_network_calls(monkeypatch, timeout):
    monkeypatch.setattr(sys, "argv", ["diagnose_gemini.py", "--timeout", timeout])
    generate = MagicMock()
    monkeypatch.setattr(doctor, "generate_gemini_text", generate)
    with pytest.raises(SystemExit) as raised:
        doctor.main()
    assert raised.value.code == 2
    generate.assert_not_called()
