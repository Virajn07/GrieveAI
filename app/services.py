"""Business services shared by the browser UI and REST API."""

from __future__ import annotations

from datetime import timedelta
import json
import os
from pathlib import Path
import uuid

from flask import current_app

from app.models_db import AuditLog, Grievance, db, utc_now
from src.baseline_classifier import BaselineGrievanceClassifier
from src.language_id import redact_pii
from src.llm_report import summarize_and_route
from src.pipeline import analyze_grievance
from src.priority_dedup import check_duplicate_against_recent, make_embedder

_classifier = None
_embedder = None


def get_classifier():
    global _classifier
    backend = os.getenv("MODEL_BACKEND", "baseline").lower()
    if backend == "muril":
        if _classifier is None:
            try:
                from src.model import MuRILInference
                checkpoint = os.getenv("MURIL_CHECKPOINT_DIR", "checkpoints/muril_lora/run1")
                _classifier = MuRILInference(checkpoint, os.getenv("TAXONOMY_CONFIG", "config/taxonomy.json"))
            except Exception:
                current_app.logger.exception("MuRIL model unavailable; falling back to configured local baseline")
        if _classifier is not None:
            return _classifier
    if _classifier is None:
        _classifier = BaselineGrievanceClassifier.load(
            os.getenv("BASELINE_CHECKPOINT_DIR", "checkpoints/baseline"),
            os.getenv("TAXONOMY_CONFIG", "config/taxonomy.json"),
        )
    return _classifier


def get_embedder():
    global _embedder
    if _embedder is None and os.getenv("DISABLE_SEMANTIC_DEDUP", "1") != "1":
        _embedder = make_embedder()
    return _embedder


def sla_hours(priority: int) -> int | None:
    """Read proposed demo SLA values from configuration."""
    path = os.getenv("SLA_CONFIG", "config/sla_rules.json")
    try:
        rules = json.loads(Path(path).read_text(encoding="utf-8"))["priority_hours"]
        value = rules.get(str(int(priority)))
        return int(value) if value is not None else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _audit(gid, action, actor, detail):
    db.session.add(AuditLog(grievance_id=gid, action=action, actor=redact_pii(actor), detail=redact_pii(detail or "")))


def submit_grievance(raw_text: str):
    """Redact, analyze, route/gate, and persist a submission; never render UI."""
    clf = get_classifier()
    analysis = analyze_grievance(raw_text, clf, current_app.config["CONFIDENCE_THRESHOLD"])
    text = analysis["text"]
    language_info = analysis["language"]
    language = language_info["language"]
    prediction = analysis["prediction"]
    try:
        explanation = clf.explain(text)
    except Exception:
        current_app.logger.exception("Could not generate submission explanation")
        explanation = []

    recent_rows = (
        Grievance.query.filter(
            Grievance.category == prediction["category"],
            Grievance.submitted_at >= utc_now() - timedelta(days=30),
        )
        .with_entities(Grievance.id, Grievance.text)
        .order_by(Grievance.submitted_at.desc())
        .limit(200)
        .all()
    )
    dup_id, dup_sim, dup_method = check_duplicate_against_recent(
        text,
        recent_rows,
        vectorizer=getattr(clf, "vectorizer", None),
        threshold=current_app.config["DEDUPE_THRESHOLD"],
        embedder=get_embedder(),
    )

    threshold = analysis["threshold"]
    llm_result = summarize_and_route(text, prediction["category"], prediction["subcategory"], os.getenv("DEPARTMENTS_CONFIG", "config/departments.json"))
    manual_review = analysis["manual_review"] or not llm_result["recommended_department"]
    department = None if manual_review else llm_result["recommended_department"]
    status = "submitted" if manual_review else "routed"

    now = utc_now()
    ack_number = f"GRV-{now:%Y%m%d}-{uuid.uuid4().hex[:12].upper()}"
    grievance = Grievance(
        ack_number=ack_number,
        text=text,
        language=language,
        script=language_info["script"],
        language_confidence=language_info["confidence"],
        category=prediction["category"],
        subcategory=prediction["subcategory"],
        priority=prediction["priority"],
        predicted_category=prediction["category"],
        predicted_subcategory=prediction["subcategory"],
        predicted_priority=prediction["priority"],
        priority_raw=prediction.get("priority_raw"),
        confidence=prediction["confidence"],
        confidence_threshold=float(threshold),
        subcategory_confidence=prediction.get("subcategory_confidence"),
        status=status,
        routed_department=department,
        model_department=llm_result["recommended_department"],
        manual_review=manual_review,
        automated_route=not manual_review,
        duplicate_of_id=dup_id,
        duplicate_similarity=dup_sim,
        duplicate_method=dup_method,
        llm_summary=llm_result["summary"],
        summary_provider=llm_result["provider"],
        submitted_at=now,
        prediction_at=now,
        routing_at=None if manual_review else now,
        manual_review_at=now if manual_review else None,
        sla_deadline=(now + timedelta(hours=sla_duration)) if (sla_duration := sla_hours(prediction["priority"])) is not None else None,
    )
    db.session.add(grievance)
    db.session.flush()
    _audit(grievance.id, "submitted", "system", f"category={prediction['category']}; confidence={prediction['confidence']}; language={language}")
    if manual_review:
        reason = "confidence_below_threshold" if analysis["manual_review"] else "department_mapping_unavailable"
        _audit(grievance.id, "manual_review_queued", "system", f"reason={reason}; threshold={threshold}")
    else:
        _audit(grievance.id, "routed", "system", f"department={department}")
    db.session.commit()
    return grievance, explanation
