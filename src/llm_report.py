"""Structured local Ollama analysis with optional OpenRouter and safe fallback."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from urllib.request import Request, urlopen

from src.language_id import redact_pii

logger = logging.getLogger(__name__)
_REQUIRED_FIELDS = (
    "summary",
    "root_cause",
    "recommended_action",
    "department_recommendation",
    "urgency_reason",
    "recurring_issue",
    "recurring_interpretation",
)
_PRIMARY_MODEL = "google/gemma-4-26b-a4b-it:free"
_BACKUP_MODEL = "apodex/apodex-1.1-mini:free"
_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
_OLLAMA_MODEL = "qwen3:8b"
_FALLBACK_LABEL = "Deterministic fallback — LLM unavailable"


def _load_departments(path):
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        mapping = config.get("mapping", {})
        return mapping if isinstance(mapping, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _extractive_summary(text: str, max_words=25) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned
    return " ".join(words[:max_words]).rstrip(".,;:") + "..."


def _fallback_analysis(text, priority, deterministic_department, recurring_matches):
    recurring = bool(recurring_matches)
    count = len(recurring_matches) if hasattr(recurring_matches, "__len__") else 0
    return {
        "summary": _extractive_summary(text),
        "root_cause": "Not established from the grievance text.",
        "recommended_action": "Review the report and verify the relevant records before taking action.",
        "department": deterministic_department,
        "department_recommendation": deterministic_department,
        "urgency_reason": (
            f"The classifier assigned priority {priority}/5; an administrator should confirm urgency."
            if int(priority) >= 4 else "The classifier did not assign a high priority; review remains available."
        ),
        "recurring_issue": recurring,
        "recurring_interpretation": (
            f"Similarity search found {count} related prior grievance(s); review whether they describe one issue."
            if recurring else "Similarity search did not identify a related prior grievance."
        ),
        "provider": "deterministic",
        "provider_label": "Deterministic",
        "fallback_label": _FALLBACK_LABEL,
        "model_name": None,
        "used_fallback": True,
    }


def _validate_analysis(value, departments):
    # Accept the former internal spelling while prompting providers for the
    # concise `department` key required by the prototype response contract.
    if isinstance(value, dict) and "department" in value and "department_recommendation" not in value:
        value = {**value, "department_recommendation": value["department"]}
    if not isinstance(value, dict) or any(key not in value for key in _REQUIRED_FIELDS):
        raise ValueError("LLM response is missing required structured fields")
    result = {}
    for key in _REQUIRED_FIELDS:
        item = value[key]
        if key == "recurring_issue":
            if not isinstance(item, bool):
                raise ValueError("recurring_issue must be a boolean")
            result[key] = item
        elif key == "department_recommendation":
            if item is not None and (not isinstance(item, str) or item not in departments):
                raise ValueError("department recommendation must match configured departments")
            result[key] = item
        else:
            if not isinstance(item, str) or not item.strip() or len(item) > 600:
                raise ValueError(f"{key} must be a non-empty string of at most 600 characters")
            result[key] = item.strip()
    result["department"] = result["department_recommendation"]
    return result


def _extract_json_object(content):
    """Decode JSON from plain text, a fenced block, or surrounding prose."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response did not contain text")
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    decoder = json.JSONDecoder()
    objects = []
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                objects.append(value)
        except (TypeError, ValueError):
            pass
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start():])
            except ValueError:
                continue
            if isinstance(value, dict):
                objects.append(value)
    if objects:
        # Qwen3 may emit JSON-like reasoning before its final answer. Prefer
        # the last object that resembles our contract; schema validation still
        # decides whether it is safe to use.
        matching = [value for value in objects if any(key in value for key in _REQUIRED_FIELDS)]
        return (matching or objects)[-1]
    raise ValueError("LLM response did not contain a valid JSON object")


def _redact_context(value):
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, list):
        return [_redact_context(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_context(item) for key, item in value.items()}
    return value


def _request_ollama(text, prediction, duplicate_context, recurring_context, departments):
    """Call local Ollama using only PII-redacted grievance and match text."""
    context = {
        "redacted_grievance": redact_pii(text),
        "prediction": {
            "category": prediction["category"],
            "subcategory": prediction["subcategory"],
            "priority": int(prediction["priority"]),
            "confidence": float(prediction["confidence"]),
        },
        "duplicate_context": _redact_context(duplicate_context or []),
        "recurring_context": _redact_context(recurring_context or []),
        "allowed_departments": sorted(set(departments.values())),
    }
    system = (
        "Analyze one student grievance and return only a JSON object with exactly these keys: "
        "summary, root_cause, recommended_action, department, urgency_reason, recurring_issue, "
        "recurring_interpretation. Understand English, Devanagari Hindi, Romanised Hindi and "
        "Hindi-English code-mixed text. Analyze only the supplied grievance and context; "
        "distinguish the primary issue from contributing causes or affected areas. Be concise, "
        "do not invent facts, and say 'Not established from the report.' when a cause is unknown. "
        "Recommend an actionable next step and an appropriate department from the supplied list, "
        "or null. recurring_issue must be a boolean. Never reveal or reproduce personal identifiers. "
        "Recommendations are for human review, must not make irreversible decisions, and must "
        "not change the classifier output or configured route."
    )
    payload = json.dumps({
        "model": os.getenv("OLLAMA_MODEL", "").strip() or _OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "think": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
        "options": {"temperature": 0, "num_predict": 300},
    }).encode("utf-8")
    base_url = os.getenv("OLLAMA_BASE_URL", "").strip() or _OLLAMA_BASE_URL
    request = Request(
        base_url.rstrip("/") + "/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    with urlopen(request, timeout=timeout) as response:
        response_body = json.loads(response.read().decode("utf-8"))
    if not isinstance(response_body, dict) or not isinstance(response_body.get("message"), dict):
        raise ValueError("Ollama response is missing its message")
    return _extract_json_object(response_body["message"].get("content"))


def _request_openrouter(text, prediction, duplicate_context, recurring_context, departments, model):
    from openai import OpenAI

    def redact_context(value):
        if isinstance(value, str):
            return redact_pii(value)
        if isinstance(value, list):
            return [redact_context(item) for item in value]
        if isinstance(value, dict):
            return {key: redact_context(item) for key, item in value.items()}
        return value

    context = {
        "redacted_grievance": redact_pii(text),
        "prediction": {
            "category": prediction["category"],
            "subcategory": prediction["subcategory"],
            "priority": int(prediction["priority"]),
            "confidence": float(prediction["confidence"]),
        },
        "duplicate_context": redact_context(duplicate_context or []),
        "recurring_context": redact_context(recurring_context or []),
        "allowed_departments": sorted(set(departments.values())),
    }
    system = (
        "Analyze one student grievance. Return a JSON object with exactly these keys: "
        "summary, root_cause, recommended_action, department, urgency_reason, "
        "recurring_issue, recurring_interpretation. Use concise text. Do not invent facts; "
        "write 'Not established from the report.' when a cause is unknown. "
        "department must be one allowed department or null. "
        "Predictions and recommendations are advisory and must not change the classifier output."
    )
    client = OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
    )
    response = client.chat.completions.create(
        model=model,
        max_tokens=500,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
    )
    content = response.choices[0].message.content
    return _extract_json_object(content)


def analyze_with_llm(
    text: str,
    prediction: dict,
    departments_path="config/departments.json",
    duplicate_context=None,
    recurring_context=None,
) -> dict:
    """Return validated structured analysis; classifier/routing stay primary."""
    mapping = _load_departments(departments_path)
    deterministic_department = mapping.get(prediction.get("category"))
    fallback = _fallback_analysis(
        text, prediction.get("priority", 1), deterministic_department, recurring_context
    )
    providers = [("ollama", os.getenv("OLLAMA_MODEL", "").strip() or _OLLAMA_MODEL, None)]
    if os.getenv("OPENROUTER_API_KEY"):
        primary_model = os.getenv("OPENROUTER_MODEL", "").strip() or _PRIMARY_MODEL
        backup_model = os.getenv("OPENROUTER_BACKUP_MODEL", "").strip() or _BACKUP_MODEL
        providers.extend([
            ("openrouter", primary_model, "Gemma" if primary_model == _PRIMARY_MODEL else "OpenRouter"),
            ("openrouter", backup_model, "Apodex" if backup_model == _BACKUP_MODEL else "OpenRouter backup"),
        ])
    attempted = set()
    for provider, model, provider_label in providers:
        key = (provider, model)
        if key in attempted:
            continue
        attempted.add(key)
        try:
            if provider == "ollama":
                response = _request_ollama(text, prediction, duplicate_context, recurring_context, mapping)
            else:
                response = _request_openrouter(text, prediction, duplicate_context, recurring_context, mapping, model)
            validated = _validate_analysis(
                response,
                set(mapping.values()),
            )
            validated["provider"] = provider
            validated["provider_label"] = "Ollama" if provider == "ollama" else provider_label
            validated["model_name"] = model
            validated["fallback_label"] = None
            validated["used_fallback"] = False
            return validated
        except Exception as exc:
            logger.warning("%s analysis unavailable or invalid (%s)", provider, exc.__class__.__name__)
    return fallback


def summarize_and_route(
    text: str,
    category: str,
    subcategory: str,
    departments_path="config/departments.json",
    *,
    priority: int = 1,
    confidence: float = 0.0,
    duplicate_context=None,
    recurring_context=None,
) -> dict:
    """Compatibility adapter for existing app services.

    Department routing remains the configured category mapping; the LLM's
    department recommendation is stored separately as advisory analysis.
    """
    analysis = analyze_with_llm(
        text,
        {"category": category, "subcategory": subcategory, "priority": priority, "confidence": confidence},
        departments_path,
        duplicate_context,
        recurring_context,
    )
    mapping = _load_departments(departments_path)
    return {
        **analysis,
        "department": analysis["department_recommendation"],
        "recommended_department": mapping.get(category),
        "llm_department_recommendation": analysis["department_recommendation"],
        "provider_label": analysis["provider_label"],
        "fallback_label": analysis["fallback_label"],
        "model_name": analysis["model_name"],
        "summary": analysis["summary"],
    }
