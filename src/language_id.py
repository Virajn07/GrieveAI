"""
Language/script detection + PII redaction (Week 2).

Plan:
    - fastText LID (lid.176.bin, pretrained) for English / Hindi (Devanagari)
      / Hinglish (Latin-script Hindi) detection.
    - Regex-based redaction pass for roll numbers, phone numbers, and a
      name-lookup redaction using the student directory (if VCET provides
      one) before any grievance text is stored or shown on the dashboard.

This is a stub so the rest of the pipeline can call a stable interface
while the real fastText integration is built.
"""

import re

ROLL_NUMBER_PATTERN = re.compile(r"\b[A-Z]{2,4}\d{4,8}\b")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")


def detect_language(text: str) -> str:
    """Returns one of 'en', 'hi', 'hinglish'.
    TODO: replace with fastText lid.176.bin inference."""
    if re.search(r"[\u0900-\u097F]", text):  # Devanagari Unicode block
        return "hi"
    # crude placeholder heuristic until fastText is wired in
    hinglish_markers = {"hai", "nahi", "kya", "mera", "kaise", "kab", "chahiye"}
    if any(w in text.lower().split() for w in hinglish_markers):
        return "hinglish"
    return "en"


def redact_pii(text: str) -> str:
    """Masks roll numbers and phone numbers. TODO: add name redaction
    once a student directory / NER approach is decided."""
    text = ROLL_NUMBER_PATTERN.sub("[ROLL_NUMBER]", text)
    text = PHONE_PATTERN.sub("[PHONE]", text)
    return text
