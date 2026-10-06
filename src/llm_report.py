"""Optional structured OpenRouter analysis with deterministic safe fallback."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

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


def _request_openrouter(text, prediction, duplicate_context, recurring_context, departments):
    from openai import OpenAI

    context = {
        "redacted_grievance": text,
        "prediction": {
            "category": prediction["category"],
            "subcategory": prediction["subcategory"],
            "priority": int(prediction["priority"]),
            "confidence": float(prediction["confidence"]),
        },
        "duplicate_context": duplicate_context or [],
        "recurring_context": recurring_context or [],
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
        model=os.environ["OPENROUTER_MODEL"],
        max_tokens=500,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
    )
    content = response.choices[0].message.content
    return json.loads(content)


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
    if not (os.getenv("OPENROUTER_API_KEY") and os.getenv("OPENROUTER_MODEL")):
        return fallback
    try:
        validated = _validate_analysis(
            _request_openrouter(text, prediction, duplicate_context, recurring_context, mapping),
            set(mapping.values()),
        )
        validated["provider"] = "openrouter"
        validated["used_fallback"] = False
        return validated
    except Exception:
        logger.warning("OpenRouter analysis unavailable or invalid; using deterministic result", exc_info=True)
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
        "summary": analysis["summary"],
    }
