"""
GrieveAI routes - the /submit endpoint is a REAL, working pipeline today:

    raw text
      -> redact_pii                        (src/language_id.py)
      -> detect_language                   (src/language_id.py)
      -> BaselineGrievanceClassifier        (src/baseline_classifier.py)
      -> check_duplicate_against_recent     (src/priority_dedup.py)
      -> confidence gate (manual_review flag)
      -> summarize_and_route (Claude -> OpenRouter fallback, skipped
         gracefully if no API keys are set - see src/llm_report.py)
      -> saved to SQLite via app/models_db.py, ack number returned

Swap-in point for later: once src/model.py (MuRIL+LoRA) is trained in
Colab, replace `get_classifier()` below with a loader for that
checkpoint - predict()/explain() have matching signatures so nothing
else in this file needs to change.
"""

from datetime import datetime, timedelta
import uuid

from flask import Blueprint, render_template, jsonify, request

from src.language_id import detect_language, redact_pii
from src.baseline_classifier import BaselineGrievanceClassifier
from src.priority_dedup import check_duplicate_against_recent
from src.llm_report import summarize_and_route
from app.models_db import db, Grievance, AuditLog

bp = Blueprint("main", __name__)

_classifier = None
CONFIDENCE_THRESHOLD = 0.55  # below this, route to manual review instead of auto-routing


def get_classifier():
    """Loads the trained baseline model once per process (not per request)."""
    global _classifier
    if _classifier is None:
        _classifier = BaselineGrievanceClassifier.load()
    return _classifier


@bp.route("/")
def index():
    grievances = Grievance.query.order_by(Grievance.submitted_at.desc()).limit(20).all()
    return render_template("dashboard.html", grievances=grievances)


@bp.route("/health")
def health():
    return jsonify({"status": "ok", "service": "GrieveAI"})


@bp.route("/submit", methods=["POST"])
def submit_grievance():
    data = request.get_json(force=True, silent=True) or {}
    raw_text = (data.get("text") or "").strip()
    if not raw_text:
        return jsonify({"error": "text is required"}), 400

    text = redact_pii(raw_text)
    language = detect_language(text)

    clf = get_classifier()
    prediction = clf.predict(text)
    explanation = clf.explain(text)

    # Duplicate check: only compare against recent grievances in the SAME
    # predicted category (cheaper, and more meaningful than comparing
    # across unrelated categories).
    recent_rows = (
        Grievance.query.filter(
            Grievance.category == prediction["category"],
            Grievance.submitted_at >= datetime.utcnow() - timedelta(days=30),
        )
        .with_entities(Grievance.id, Grievance.text)
        .limit(200)
        .all()
    )
    dup_id, dup_sim = check_duplicate_against_recent(text, recent_rows, clf.vectorizer)

    manual_review = prediction["confidence"] < CONFIDENCE_THRESHOLD

    llm_result = None
    if not manual_review:
        # Fails gracefully to summary=None if ANTHROPIC_API_KEY /
        # OPENROUTER_API_KEY aren't set - see src/llm_report.py.
        llm_result = summarize_and_route(text, prediction["category"], prediction["subcategory"])

    ack_number = f"GRV-{uuid.uuid4().hex[:8].upper()}"
    department = llm_result["recommended_department"] if llm_result else None

    grievance = Grievance(
        ack_number=ack_number,
        text=text,
        language=language,
        category=prediction["category"],
        subcategory=prediction["subcategory"],
        priority=prediction["priority"],
        confidence=prediction["confidence"],
        status="submitted" if manual_review else "routed",
        routed_department=department,
        manual_review=manual_review,
        duplicate_of_id=dup_id,
        llm_summary=llm_result["summary"] if llm_result else None,
        sla_deadline=datetime.utcnow() + timedelta(days=max(1, 6 - prediction["priority"])),
    )
    db.session.add(grievance)
    db.session.commit()

    db.session.add(AuditLog(
        grievance_id=grievance.id,
        action="submitted",
        actor="system",
        detail=f"category={prediction['category']} confidence={prediction['confidence']}",
    ))
    db.session.commit()

    return jsonify({
        "ack_number": ack_number,
        "language": language,
        "category": prediction["category"],
        "subcategory": prediction["subcategory"],
        "priority": prediction["priority"],
        "confidence": prediction["confidence"],
        "manual_review": manual_review,
        "duplicate_of": dup_id,
        "duplicate_similarity": dup_sim,
        "top_explanation_words": [w for w, _ in explanation],
        "routed_department": department,
        "summary": grievance.llm_summary,
    })
