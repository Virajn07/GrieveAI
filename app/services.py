"""Business services shared by the browser UI and REST API."""

from __future__ import annotations

from datetime import timedelta
import json
import os
from pathlib import Path
import uuid

from flask import current_app

from app.models_db import AuditLog, Grievance, db, utc_now
from src.inference import get_classifier, predict_grievance
from src.language_id import redact_pii
from src.llm_report import summarize_and_route
from src.configuration import load_threshold_settings
from src.priority_dedup import find_similar_grievances, cluster_recurring_grievances, make_embedder

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None and os.getenv("DISABLE_SEMANTIC_DEDUP", "0") != "1":
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
    inference = predict_grievance(
        raw_text, classifier=clf, confidence_threshold=current_app.config["CONFIDENCE_THRESHOLD"]
    )
    text = inference["redacted_text"]
    language = inference["language"]
    prediction = {
        "category": inference["category"],
        "subcategory": inference["subcategory"],
        "priority": inference["priority"],
        "priority_raw": inference["priority_raw"],
        "confidence": inference["confidence"],
        "subcategory_confidence": inference["subcategory_confidence"],
    }
    recent_rows = (
        Grievance.query.filter(
            Grievance.category == prediction["category"],
            Grievance.submitted_at >= utc_now() - timedelta(days=30),
        )
        .with_entities(Grievance.id, Grievance.text, Grievance.category, Grievance.subcategory)
        .order_by(Grievance.submitted_at.desc())
        .limit(200)
        .all()
    )
    similar_matches = find_similar_grievances(
        text,
        recent_rows,
        vectorizer=getattr(clf, "vectorizer", None),
        embedder=get_embedder(),
        duplicate_threshold=current_app.config["DEDUPE_THRESHOLD"],
        related_threshold=load_threshold_settings()["related_similarity_threshold"],
        top_k=5,
    )
    duplicate_match = next(
        (match for match in similar_matches if match["relationship"] in {"exact_duplicate", "near_duplicate"}),
        None,
    )
    dup_id = duplicate_match["grievance_id"] if duplicate_match else None
    dup_sim = duplicate_match["similarity"] if duplicate_match else (similar_matches[0]["similarity"] if similar_matches else 0.0)
    dup_method = duplicate_match["method"] if duplicate_match else (similar_matches[0]["method"] if similar_matches else "none")

    threshold = inference["confidence_threshold"]
    llm_result = summarize_and_route(
        text,
        prediction["category"],
        prediction["subcategory"],
        os.getenv("DEPARTMENTS_CONFIG", "config/departments.json"),
        priority=prediction["priority"],
        confidence=prediction["confidence"],
        duplicate_context=similar_matches,
        recurring_context=similar_matches,
    )
    manual_review = inference["requires_human_review"] or not llm_result["recommended_department"]
    department = None if manual_review else llm_result["recommended_department"]
    status = "submitted" if manual_review else "routed"

    now = utc_now()
    ack_number = f"GRV-{now:%Y%m%d}-{uuid.uuid4().hex[:12].upper()}"
    grievance = Grievance(
        ack_number=ack_number,
        text=text,
        language=language,
        script=inference["script"],
        language_confidence=inference["language_confidence"],
        category=prediction["category"],
        subcategory=prediction["subcategory"],
        priority=prediction["priority"],
        explanation=None,
        predicted_category=prediction["category"],
        predicted_subcategory=prediction["subcategory"],
        predicted_priority=prediction["priority"],
        priority_raw=prediction.get("priority_raw"),
        confidence=prediction["confidence"],
        confidence_threshold=float(threshold),
        subcategory_confidence=prediction.get("subcategory_confidence"),
        model_version=getattr(clf, "model_version", clf.__class__.__name__),
        status=status,
        routed_department=department,
        manual_review=manual_review,
        automated_route=not manual_review,
        duplicate_of_id=dup_id,
        duplicate_similarity=dup_sim,
        duplicate_method=dup_method,
        related_matches=[
            {key: match[key] for key in ("grievance_id", "similarity", "relationship", "method")}
            for match in similar_matches
        ],
        llm_summary=llm_result["summary"],
        summary_provider=llm_result["provider"],
        model_department=llm_result["llm_department_recommendation"],
        llm_analysis={key: llm_result[key] for key in (
            "summary", "root_cause", "recommended_action", "department_recommendation",
            "urgency_reason", "recurring_issue", "recurring_interpretation", "provider", "used_fallback",
        )},
        submitted_at=now,
        prediction_at=now,
        routing_at=None if manual_review else now,
        manual_review_at=now if manual_review else None,
        sla_deadline=(now + timedelta(hours=sla_duration)) if (sla_duration := sla_hours(prediction["priority"])) is not None else None,
    )
    db.session.add(grievance)
    db.session.flush()
    cluster_rows = list(recent_rows) + [(grievance.id, text, prediction["category"], prediction["subcategory"])]
    recurring_clusters = cluster_recurring_grievances(
        cluster_rows,
        vectorizer=getattr(clf, "vectorizer", None),
        embedder=get_embedder(),
    )
    current_cluster = next(
        (cluster for cluster in recurring_clusters if str(grievance.id) in cluster["grievance_ids"]),
        None,
    )
    if current_cluster:
        grievance.recurring_cluster_id = current_cluster["cluster_id"]
        member_ids = [int(value) for value in current_cluster["grievance_ids"] if value.isdigit() and int(value) != grievance.id]
        if member_ids:
            Grievance.query.filter(Grievance.id.in_(member_ids)).update(
                {Grievance.recurring_cluster_id: current_cluster["cluster_id"]},
                synchronize_session=False,
            )
    _audit(grievance.id, "submitted", "system", f"category={prediction['category']}; confidence={prediction['confidence']}; language={language}")
    if manual_review:
        reason = "confidence_below_threshold" if inference["requires_human_review"] else "department_mapping_unavailable"
        _audit(grievance.id, "manual_review_queued", "system", f"reason={reason}; threshold={threshold}")
    else:
        _audit(grievance.id, "routed", "system", f"department={department}")
    db.session.commit()
    return grievance, None
