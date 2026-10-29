#!/usr/bin/env python3
"""Verify real generation, using one bounded request and no business data."""

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
from src.llm import GeminiRequestError, generate_gemini_text


def check_generation(model=None, timeout=None):
    config = replace(settings, gemini_max_attempts=1,
        gemini_timeout_seconds=timeout if timeout is not None else settings.gemini_timeout_seconds)
    model = model or config.gemini_model
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model, "timeout_seconds": config.gemini_timeout_seconds,
        "max_attempts": 1, "thinking_level": config.gemini_thinking_level,
        "endpoint": "generateContent", "business_data_sent": False,
    }
    started = time.monotonic()
    try:
        if not config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        response = generate_gemini_text(
            "Return only the text OK.", model_name=model, api_key=config.gemini_api_key, config=config,
        )
        report["status"] = "SUCCESS" if response.strip() == "OK" else "UNEXPECTED_RESPONSE"
        report["response"] = response[:200]
    except (GeminiRequestError, ValueError) as exc:
        report.update(status="FAILED", code=getattr(exc, "code", None), error=str(exc))
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Probe another model without modifying .env.")
    parser.add_argument("--timeout", type=float, help="Total request deadline in seconds.")
    parser.add_argument("--output", type=Path, help="Save the redacted diagnostic as JSON.")
    args = parser.parse_args()
    try:
        report = check_generation(model=args.model, timeout=args.timeout)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
