"""
Summarisation + department-routing recommendation (Week 4).

Primary: Anthropic Claude API.
Fallback: OpenRouter (OpenAI-compatible client), routed to an open-source
model, used only if the primary call fails, times out, or is rate-limited.
This is an engineering-reliability choice, not a research contribution -
keep it framed that way in the project report.

Env vars expected (put these in a .env file, never commit it - see .gitignore):
    ANTHROPIC_API_KEY=...
    OPENROUTER_API_KEY=...
"""

import os

CATEGORY_TO_DEPARTMENT = {
    "Academics": "Academic Affairs Office",
    "Examinations": "Examination Cell",
    "Fees_Accounts": "Accounts Office",
    "IT_Library": "IT Services / Library",
    "Infrastructure": "Facilities & Maintenance",
    "Transport": "Transport Committee",
    "Canteen": "Canteen Management",
}

SUMMARY_PROMPT = """You are summarising a student grievance for a college
administrator. Write ONE factual sentence (no more than 25 words)
describing what the student is reporting. Do not add opinions or
recommendations - just state what happened.

Grievance category: {category} / {subcategory}
Grievance text: {text}

One-sentence summary:"""


def summarize_with_claude(text: str, category: str, subcategory: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": SUMMARY_PROMPT.format(category=category, subcategory=subcategory, text=text),
        }],
    )
    return response.content[0].text.strip()


def summarize_with_openrouter(text: str, category: str, subcategory: str,
                               model: str = "meta-llama/llama-3.1-8b-instruct:free") -> str:
    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
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


def summarize_and_route(text: str, category: str, subcategory: str) -> dict:
    """Tries Claude first, falls back to OpenRouter on any failure.
    Always returns a department recommendation even if both LLM calls
    fail, since routing must not depend on an external API being up."""
    summary = None
    used_fallback = False

    try:
        summary = summarize_with_claude(text, category, subcategory)
    except Exception as primary_err:
        used_fallback = True
        try:
            summary = summarize_with_openrouter(text, category, subcategory)
        except Exception as fallback_err:
            summary = None  # both failed - dashboard shows raw text instead

    return {
        "summary": summary,
        "used_fallback": used_fallback,
        "recommended_department": CATEGORY_TO_DEPARTMENT.get(category, "General Grievance Cell"),
    }
