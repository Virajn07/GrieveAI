"""Optional LLM summarisation with deterministic, auditable routing.

Routing is never decided by the LLM. Category -> department mapping comes from
configuration so staff can audit it. LLMs are used only for a concise factual
summary when API credentials are configured; otherwise an extractive summary
keeps the demo fully functional.
"""

from __future__ import annotations

import os
import re
import json
import logging
from pathlib import Path

SUMMARY_PROMPT = """You are summarising a student grievance for a college administrator.
Write ONE factual sentence of no more than 25 words.
Do not invent facts, names, dates, causes, or recommendations.
Category: {category}
Sub-category: {subcategory}
Text: {text}
"""


def _extractive_summary(text: str, max_words=25) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned
    return " ".join(words[:max_words]).rstrip(".,;:") + "..."


def _anthropic_summary(text, category, subcategory):
    import anthropic

    model = os.getenv("ANTHROPIC_MODEL")
    if not model:
        raise RuntimeError("ANTHROPIC_MODEL is not set")
    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
    )
    response = client.messages.create(
        model=model,
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": SUMMARY_PROMPT.format(category=category, subcategory=subcategory, text=text),
        }],
    )
    return response.content[0].text.strip()


def _openrouter_summary(text, category, subcategory):
    from openai import OpenAI

    model = os.getenv("OPENROUTER_MODEL")
    if not model:
        raise RuntimeError("OPENROUTER_MODEL is not set")
    client = OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
    )
    response = client.chat.completions.create(
        model=model,
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": SUMMARY_PROMPT.format(category=category, subcategory=subcategory, text=text),
        }],
    )
    return response.choices[0].message.content.strip()


def summarize_and_route(text: str, category: str, subcategory: str, departments_path="config/departments.json") -> dict:
    summary = _extractive_summary(text)
    used_fallback = True
    llm_provider = "deterministic"

    if os.getenv("ANTHROPIC_API_KEY") and os.getenv("ANTHROPIC_MODEL"):
        try:
            summary = _anthropic_summary(text, category, subcategory)
            used_fallback = False
            llm_provider = "anthropic"
        except Exception:
            logging.getLogger(__name__).warning("Anthropic summary unavailable; using configured fallback")

    if used_fallback and os.getenv("OPENROUTER_API_KEY") and os.getenv("OPENROUTER_MODEL"):
        try:
            summary = _openrouter_summary(text, category, subcategory)
            used_fallback = False
            llm_provider = "openrouter"
        except Exception:
            logging.getLogger(__name__).warning("OpenRouter summary unavailable; using extractive fallback")

    try:
        configured = json.loads(Path(departments_path).read_text(encoding="utf-8"))
        departments = configured.get("mapping", {})
        if not isinstance(departments, dict):
            departments = {}
    except (OSError, ValueError):
        departments = {}
    return {
        "summary": summary,
        "used_fallback": used_fallback,
        "provider": llm_provider,
        "recommended_department": departments.get(category),
    }
