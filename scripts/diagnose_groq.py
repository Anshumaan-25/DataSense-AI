#!/usr/bin/env python3
"""Verify Groq authentication, model access, and JSON generation with one request."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.groq_client import GroqRequestError, generate_groq_text


def check_generation(model=None, timeout=None):
    config = replace(settings, groq_max_attempts=1,
        groq_timeout_seconds=timeout if timeout is not None else settings.groq_timeout_seconds)
    model = model or config.groq_model
    report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "groq", "model": model, "business_data_sent": False,
        "timeout_seconds": config.groq_timeout_seconds, "max_attempts": 1}
    started = time.monotonic()
    try:
        if not config.groq_api_key:
            raise ValueError("GROQ_API_KEY is not configured.")
        response = generate_groq_text('Return exactly this JSON object: {"ok": true}',
            model_name=model, api_key=config.groq_api_key, response_mime_type="application/json", config=config)
        report["status"] = "SUCCESS" if json.loads(response) == {"ok": True} else "UNEXPECTED_RESPONSE"
        report["response"] = response[:200]
    except (GroqRequestError, ValueError) as exc:
        report.update(status="FAILED", code=getattr(exc, "code", None), error=str(exc))
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Check a model without changing .env.")
    parser.add_argument("--timeout", type=float, help="Total logical request budget in seconds.")
    parser.add_argument("--output", type=Path, help="Save redacted JSON diagnostics.")
    args = parser.parse_args()
    try:
        report = check_generation(args.model, args.timeout)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
