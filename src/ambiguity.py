"""Ambiguity Classifier and Intent Clarification Agent for DataSense AI.

Distinguishes between unambiguous queries (CLEAR), mild underspecifications (SOFT_DEFAULT),
and conflicting business definitions (HARD_AMBIGUITY), presenting structured choices
derived from the business metric dictionary.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.config import settings


@dataclass
class AmbiguityResult:
    """Structured analysis of user query ambiguity and clarification choices."""

    is_ambiguous: bool
    ambiguity_type: str  # 'CLEAR' | 'SOFT_DEFAULT' | 'HARD_AMBIGUITY'
    reasoning: str = ""
    clarification_prompt: str = ""
    options: List[Dict[str, str]] = field(default_factory=list)
    default_interpretation: str = ""


AMBIGUITY_SYSTEM_PROMPT = """You are a Senior Data Analyst and Intent Clarification Agent for a DuckDB analytics platform.
Your task is to analyze a natural language user question in the context of the database schema and retrieved business metrics, and classify whether the request requires user clarification.

### 3-TIER TAXONOMY:

1. "CLEAR":
   - The user query is specific, direct, or explicitly states the target metric, filter conditions, and table entities.
   - Example: "What is the account balance of customer#000000001?"
   - Example: "What is the net revenue for customers in Germany?"
   - Classification: is_ambiguous = false.

2. "SOFT_DEFAULT":
   - The query leaves minor operational choices unstated (e.g., missing sort order or row limit), but the analytical goal is obvious.
   - Safe defaults exist (e.g., default to LIMIT 10, or order by date descending).
   - Do NOT interrupt the user.
   - Example: "Show orders" -> Default to: "SELECT * FROM orders ORDER BY o_orderdate DESC LIMIT 10;"
   - Example: "List customers" -> Default to: "SELECT * FROM customer LIMIT 10;"
   - Classification: is_ambiguous = false, provide default_interpretation.

3. "HARD_AMBIGUITY":
   - The query has multiple conflicting, semantically divergent interpretations of core business definitions where guessing wrong produces misleading analytical results.
   - Example: "Show top customers" -> Ambiguous: Top by Net Revenue? Top by Total Orders Placed? Top by Account Balance?
   - Example: "Which parts sell best?" -> Ambiguous: By Total Quantity Sold vs. By Total Gross Revenue?
   - Example: "How is Europe performing?" -> Ambiguous: By Net Revenue vs. By Order Count vs. By Customer Count?
   - Classification: is_ambiguous = true.
   - Formulate 2-3 structured options. When formulating options, ALWAYS prioritize metrics defined in the RETRIEVED BUSINESS METRICS & RULES if applicable.

### OUTPUT FORMAT:
Output ONLY a valid JSON object with the following schema:
{
  "ambiguity_type": "CLEAR" | "SOFT_DEFAULT" | "HARD_AMBIGUITY",
  "reasoning": "Brief explanation of your classification",
  "clarification_prompt": "A concise question asking the user for their intent (empty if CLEAR)",
  "options": [
    {"label": "Short label for choice 1", "details": "Formula or join path"}
  ],
  "default_interpretation": "Safe fallback interpretation if user bypasses clarification"
}
"""


def parse_ambiguity_json(raw_text: str) -> AmbiguityResult:
    """Parse raw LLM response text into a typed AmbiguityResult with robust fence stripping."""
    if not raw_text:
        return AmbiguityResult(
            is_ambiguous=False,
            ambiguity_type="CLEAR",
            reasoning="Empty LLM response, defaulting to CLEAR.",
        )

    clean_text = raw_text.strip()

    # Strip markdown ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text, re.IGNORECASE)
    if match:
        clean_text = match.group(1).strip()
    else:
        # Try to find { ... } block directly
        start_idx = clean_text.find("{")
        end_idx = clean_text.rfind("}")
    try:
        data = json.loads(clean_text)
    except Exception as exc:
        return AmbiguityResult(
            is_ambiguous=False,
            ambiguity_type="CLEAR",
            reasoning=f"Fallback on invalid JSON: {exc}",
        )

    if not isinstance(data, dict):
        return AmbiguityResult(
            is_ambiguous=False,
            ambiguity_type="CLEAR",
            reasoning="Parsed JSON is not an object, defaulting to CLEAR.",
        )

    ambiguity_type = str(data.get("ambiguity_type", "CLEAR")).upper()
    if ambiguity_type not in ("CLEAR", "SOFT_DEFAULT", "HARD_AMBIGUITY"):
        ambiguity_type = "CLEAR"

    is_ambiguous = ambiguity_type == "HARD_AMBIGUITY"
    reasoning = str(data.get("reasoning", ""))
    clarification_prompt = str(data.get("clarification_prompt", ""))
    raw_options = data.get("options", [])
    options: List[Dict[str, str]] = []

    if isinstance(raw_options, list):
        for opt in raw_options:
            if isinstance(opt, dict):
                options.append({
                    "label": str(opt.get("label", "Option")),
                    "details": str(opt.get("details", "")),
                })
            elif isinstance(opt, str):
                options.append({"label": opt, "details": ""})

    default_interpretation = str(data.get("default_interpretation", ""))

    return AmbiguityResult(
        is_ambiguous=is_ambiguous,
        ambiguity_type=ambiguity_type,
        reasoning=reasoning,
        clarification_prompt=clarification_prompt,
        options=options,
        default_interpretation=default_interpretation,
    )


class AmbiguityClassifier:
    """Classifies user intent ambiguity and generates domain-grounded clarification options."""

    def __init__(self, classifier_fn: Optional[Callable[..., str]] = None):
        """Initialize the classifier with an optional custom LLM invoking function."""
        self.classifier_fn = classifier_fn

    def classify_intent(
        self,
        user_question: str,
        schema_info: str,
        rag_context: str = "",
        provider: Optional[str] = None,
    ) -> AmbiguityResult:
        """Classify user query intent into CLEAR, SOFT_DEFAULT, or HARD_AMBIGUITY.

        Args:
            user_question: Natural language query submitted by user.
            schema_info: Database DDL or table schema summary.
            rag_context: Business metric and filter rules retrieved via RAG.
            provider: LLM provider override ('gemini', 'openai').

        Returns:
            AmbiguityResult with 3-tier classification, reasoning, and structured options.
        """
        if not user_question or not user_question.strip():
            return AmbiguityResult(
                is_ambiguous=False,
                ambiguity_type="CLEAR",
                reasoning="Empty question, proceeding directly.",
            )

        # Build prompt
        prompt_parts = [
            f"Database Schema:\n{schema_info}",
        ]
        if rag_context and rag_context.strip():
            prompt_parts.append(f"### RETRIEVED BUSINESS METRICS & RULES:\n{rag_context.strip()}")
        prompt_parts.append(f"### USER QUESTION:\n{user_question.strip()}")
        prompt_parts.append(
            "Analyze the ambiguity of this question according to the 3-Tier Taxonomy. Output strictly a JSON object."
        )

        user_prompt = "\n\n".join(prompt_parts)

        # If a custom classifier function was injected (e.g. for testing)
        if self.classifier_fn is not None:
            try:
                raw_response = self.classifier_fn(user_prompt)
                return parse_ambiguity_json(raw_response)
            except Exception as exc:
                return AmbiguityResult(
                    is_ambiguous=False,
                    ambiguity_type="CLEAR",
                    reasoning=f"Custom classifier fallback on exception: {exc}",
                )

        # Call configured LLM provider
        active_provider = provider.lower() if provider else settings.get_effective_provider()

        try:
            if active_provider == "gemini":
                if not settings.gemini_api_key:
                    return AmbiguityResult(
                        is_ambiguous=False,
                        ambiguity_type="CLEAR",
                        reasoning="Gemini API key not configured, bypassing classifier.",
                    )
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=settings.gemini_api_key)
                response = client.models.generate_content(
                    model=settings.gemini_model,
                    contents=[AMBIGUITY_SYSTEM_PROMPT, user_prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                    ),
                )
                raw_response = response.text or ""
            elif active_provider == "openai":
                if not settings.openai_api_key:
                    return AmbiguityResult(
                        is_ambiguous=False,
                        ambiguity_type="CLEAR",
                        reasoning="OpenAI API key not configured, bypassing classifier.",
                    )
                from openai import OpenAI

                client = OpenAI(api_key=settings.openai_api_key)
                response = client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[
                        {"role": "system", "content": AMBIGUITY_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                raw_response = response.choices[0].message.content or ""
            else:
                return AmbiguityResult(
                    is_ambiguous=False,
                    ambiguity_type="CLEAR",
                    reasoning="No API key detected, bypassing classifier.",
                )

            return parse_ambiguity_json(raw_response)

        except Exception as exc:
            # Robust Fail-Safe Fallback: never let an ambiguity error crash the execution flow
            return AmbiguityResult(
                is_ambiguous=False,
                ambiguity_type="CLEAR",
                reasoning=f"Fallback on classifier exception: {exc}",
            )
