"""LLM wrapper to convert natural language business questions and schema into DuckDB SQL."""

from __future__ import annotations

import os
import re
from typing import Optional
from src.config import settings

SYSTEM_PROMPT = """You are a Senior Data Engineer specializing in DuckDB and TPC-H analytics.
Given a database schema and a natural language user question, generate a correct, optimal, and strictly read-only DuckDB SQL query.

Rules:
1. Output ONLY executable DuckDB SQL inside a ```sql ... ``` markdown code block.
2. Do NOT provide any explanatory text, commentary, or pleasantries before or after the code block.
3. The query MUST be read-only (SELECT or WITH ... SELECT). Never use DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, or PRAGMA.
4. Use standard TPC-H join keys:
   - customer.c_custkey = orders.o_custkey
   - orders.o_orderkey = lineitem.l_orderkey
   - part.p_partkey = lineitem.l_partkey
   - supplier.s_suppkey = lineitem.l_suppkey
   - customer.c_nationkey = nation.n_nationkey
   - supplier.s_nationkey = nation.n_nationkey
   - nation.n_regionkey = region.r_regionkey
5. Ensure column names and table names match the provided schema exactly.
6. Query ONLY the tables provided in the schema. Never use external file functions (read_csv, read_parquet, read_json, read_text, glob) or file path replacement scans.
"""


def extract_sql_from_response(raw_text: str) -> str:
    """Extract clean SQL string from LLM response, stripping markdown fences."""
    if not raw_text:
        return ""

    text = raw_text.strip()

    # Look for ```sql ... ``` block
    sql_match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if sql_match:
        sql = sql_match.group(1).strip()
    else:
        # If no code fence is present, remove lines that look like markdown headers or notes
        lines = [
            line for line in text.splitlines()
            if not line.strip().startswith(("#", "//", "Note:", "Here is"))
        ]
        sql = "\n".join(lines).strip()

    return sql


def _call_gemini(user_prompt: str, model_name: str, api_key: str) -> str:
    """Invoke Google Gemini model using the official google-genai SDK."""
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=[
            SYSTEM_PROMPT,
            user_prompt,
        ],
    )
    return response.text or ""


def _call_openai(user_prompt: str, model_name: str, api_key: str) -> str:
    """Invoke OpenAI model using the openai SDK."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def generate_sql(
    user_question: str,
    schema_info: str,
    provider: Optional[str] = None,
) -> str:
    """Convert a natural language question into executable DuckDB SQL using an LLM.

    Args:
        user_question: Natural language question asked by user.
        schema_info: Extracted DDL/schema text.
        provider: 'gemini', 'openai', or None (uses config auto-detection).

    Returns:
        Clean, executable SQL query string.

    Raises:
        ValueError: If no API key is configured or generation fails.
    """
    if not user_question or not user_question.strip():
        raise ValueError("User question cannot be empty.")

    active_provider = provider.lower() if provider else settings.get_effective_provider()

    user_prompt = f"""Database Schema:
{schema_info}

User Question:
{user_question}

Write the DuckDB SQL query to answer the question:"""

    raw_response = ""

    if active_provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError(
                "Gemini provider selected but GEMINI_API_KEY is not set in environment or .env file."
            )
        raw_response = _call_gemini(
            user_prompt=user_prompt,
            model_name=settings.gemini_model,
            api_key=settings.gemini_api_key,
        )
    elif active_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OpenAI provider selected but OPENAI_API_KEY is not set in environment or .env file."
            )
        raw_response = _call_openai(
            user_prompt=user_prompt,
            model_name=settings.openai_model,
            api_key=settings.openai_api_key,
        )
    else:
        raise ValueError(
            "No LLM API key detected. Please configure GEMINI_API_KEY or OPENAI_API_KEY "
            "in your .env file or environment variables to enable Text-to-SQL generation."
        )

    sql = extract_sql_from_response(raw_response)
    if not sql:
        raise ValueError("LLM did not return a valid SQL query.")

    return sql
