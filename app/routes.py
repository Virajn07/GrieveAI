"""Student submission, tracking and admin workflow routes."""

from __future__ import annotations

import json
import queue
import threading
from collections import Counter
from functools import lru_cache
from pathlib import Path
import os

import pandas as pd

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash
from urllib.parse import urljoin, urlsplit
from datetime import datetime, time, timedelta

from app.models_db import AuditLog, Grievance, as_utc, db, utc_now
from app.services import _audit, get_classifier, get_embedder, get_similarity_vectorizer, submit_grievance as submit_grievance_service
from src.language_id import redact_pii
from src.priority_dedup import cluster_recurring_grievances

bp = Blueprint("main", __name__)
_shap_lock = threading.Lock()

STATUS_VALUES = {"submitted", "routed", "in_progress", "resolved", "rejected", "duplicate", "closed"}
STATUS_TRANSITIONS = {
    "submitted": {"routed", "rejected", "duplicate"},
    "routed": {"in_progress", "resolved", "rejected", "duplicate"},
    "in_progress": {"resolved", "rejected", "duplicate"},
    "resolved": {"closed"}, "rejected": {"closed"}, "duplicate": {"closed"}, "closed": set(),
}


def sla_status(grievance, at=None) -> str:
    if not grievance.sla_deadline:
        return "unavailable"
    at = as_utc(at) if at else utc_now()
    deadline = as_utc(grievance.sla_deadline)
    if grievance.resolved_at:
        return "met" if as_utc(grievance.resolved_at) <= deadline else "overdue"
    return "overdue" if at > deadline else "on_track"


def _admin_configured():
    return bool(current_app.config.get("ADMIN_USERNAME") and current_app.config.get("ADMIN_PASSWORD_HASH"))


def _admin_authenticated():
    return (
        _admin_configured()
        and session.get("admin_authenticated") is True
        and session.get("admin_username") == current_app.config.get("ADMIN_USERNAME")
    )


def _require_admin():
    if not _admin_configured():
        return jsonify({"error": "admin actions are disabled; configure ADMIN_USERNAME and ADMIN_PASSWORD"}), 503
    if not _admin_authenticated():
        return jsonify({"error": "admin authentication required"}), 401
    return None


def _safe_next(target):
    if not target:
        return None
    candidate = urlsplit(urljoin(request.host_url, target))
    base = urlsplit(request.host_url)
    if candidate.scheme in {"http", "https"} and candidate.netloc == base.netloc:
        return target
    return None


@bp.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    configured = _admin_configured()
    if request.method == "GET":
        return render_template("admin_login.html", configured=configured)
    if not configured:
        return render_template(
            "admin_login.html", configured=False,
            error="Admin sign-in is unavailable. Configure ADMIN_USERNAME and ADMIN_PASSWORD locally.",
        ), 503
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    if not username or not password:
        return render_template(
            "admin_login.html", configured=True,
            error="Enter both username and password.",
        ), 400
    valid = username == current_app.config["ADMIN_USERNAME"]
    try:
        password_valid = check_password_hash(current_app.config["ADMIN_PASSWORD_HASH"], password)
    except (TypeError, ValueError):
        password_valid = False
    if not (valid and password_valid):
        return render_template(
            "admin_login.html", configured=True,
            error="Invalid username or password.",
        ), 401
    target = _safe_next(request.args.get("next"))
    session.clear()
    session["admin_authenticated"] = True
    session["admin_username"] = current_app.config["ADMIN_USERNAME"]
    return redirect(target or url_for("main.index"))

@bp.route("/admin/logout", methods=["POST"])
def admin_logout():
    session.clear()
    return redirect(url_for("main.admin_login"))


@bp.route("/")
def index():
    if not _admin_authenticated():
        return redirect(url_for("main.admin_login"))
    query = Grievance.query
    department = request.args.get("department", "").strip()
    status = request.args.get("status", "").strip()
    category = request.args.get("category", "").strip()
    subcategory = request.args.get("subcategory", "").strip()
    priority = request.args.get("priority", "").strip()
    review = request.args.get("review", "").strip()
    related = request.args.get("related", "").strip()
    duplicate = request.args.get("duplicate", "").strip()
    recurring = request.args.get("recurring", "").strip()
    date_from = request.args.get("date_from", "").strip()
    date_to = request.args.get("date_to", "").strip()
    search = request.args.get("q", "").strip()[:80]
    sort = request.args.get("sort", "newest").strip()
    page = request.args.get("page", 1, type=int)
    if department:
        query = query.filter_by(routed_department=department)
    if category:
        query = query.filter_by(category=category)
    if subcategory:
        query = query.filter_by(subcategory=subcategory)
    if priority in {"1", "2", "3", "4", "5"}:
        query = query.filter_by(priority=int(priority))
    if review == "needed":
        query = query.filter_by(manual_review=True)
    elif review == "reviewed":
        reviewed = db.session.query(AuditLog.grievance_id).filter(
            AuditLog.action.in_(["classification_accepted", "classification_override"])
        )
        query = query.filter(Grievance.id.in_(reviewed))
    if status == "review":
        query = query.filter_by(manual_review=True, status="submitted")
    elif status:
        query = query.filter_by(status=status)
    if related == "yes":
        query = query.filter(db.or_(Grievance.duplicate_of_id.isnot(None), Grievance.recurring_cluster_id.isnot(None)))
    elif related == "no":
        query = query.filter(Grievance.duplicate_of_id.is_(None), Grievance.recurring_cluster_id.is_(None))
    if duplicate == "yes":
        query = query.filter(Grievance.duplicate_of_id.isnot(None))
    elif duplicate == "no":
        query = query.filter(Grievance.duplicate_of_id.is_(None))
    if recurring == "yes":
        query = query.filter(Grievance.recurring_cluster_id.isnot(None))
    elif recurring == "no":
        query = query.filter(Grievance.recurring_cluster_id.is_(None))
    if date_from:
        try:
            query = query.filter(Grievance.submitted_at >= datetime.combine(datetime.strptime(date_from, "%Y-%m-%d").date(), time.min))
        except ValueError:
            date_from = ""
    if date_to:
        try:
            query = query.filter(Grievance.submitted_at < datetime.combine(datetime.strptime(date_to, "%Y-%m-%d").date() + timedelta(days=1), time.min))
        except ValueError:
            date_to = ""
    if search:
        pattern = f"%{search}%"
        query = query.filter(db.or_(Grievance.ack_number.ilike(pattern), Grievance.text.ilike(pattern)))
    if sort == "oldest":
        query = query.order_by(Grievance.submitted_at.asc())
    elif sort == "priority":
        query = query.order_by(Grievance.priority.desc(), Grievance.submitted_at.desc())
    elif sort == "confidence":
        query = query.order_by(Grievance.confidence.asc(), Grievance.submitted_at.desc())
    else:
        sort = "newest"
        query = query.order_by(Grievance.submitted_at.desc())
    pagination = query.paginate(page=max(1, page), per_page=25, error_out=False)
    grievances = pagination.items
    reviewed_ids = {
        row[0] for row in db.session.query(AuditLog.grievance_id).filter(
            AuditLog.grievance_id.in_([g.id for g in grievances] or [-1]),
            AuditLog.action.in_(["classification_accepted", "classification_override"]),
        ).distinct().all()
    }
    counts = {
        "total": Grievance.query.count(),
        "pending": Grievance.query.filter_by(status="submitted").count(),
        "review": Grievance.query.filter_by(manual_review=True).count(),
        "under_review": Grievance.query.filter_by(status="in_progress").count(),
        "open": Grievance.query.filter(Grievance.status.in_(["submitted", "routed", "in_progress"])).count(),
        "resolved": Grievance.query.filter_by(status="resolved").count(),
        "high_priority": Grievance.query.filter(Grievance.priority >= 4).count(),
        "overrides": AuditLog.query.filter_by(action="override").count(),
    }
    try:
        departments = json.loads(Path(os.getenv("DEPARTMENTS_CONFIG", "config/departments.json")).read_text(encoding="utf-8")).get("mapping", {})
    except (OSError, ValueError):
        departments = {}
    analytics = _build_analytics()
    taxonomy = _load_taxonomy()
    return render_template(
        "dashboard.html", grievances=grievances, counts=counts, analytics=analytics,
        departments=sorted(set(departments.values())), categories=sorted(taxonomy),
        selected_department=department, selected_status=status, selected_category=category,
        selected_subcategory=subcategory, selected_priority=priority, selected_review=review,
        selected_related=related, selected_duplicate=duplicate, selected_recurring=recurring,
        date_from=date_from, date_to=date_to, search=search, sort=sort,
        subcategories=_load_taxonomy(), pagination=pagination, reviewed_ids=reviewed_ids,
        admin_name=current_app.config.get("ADMIN_USERNAME") or "Grievance Admin", now=utc_now(),
    )


def _load_taxonomy():
    path = Path(os.getenv("TAXONOMY_CONFIG", "config/taxonomy.json"))
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("categories", {})
    except (OSError, ValueError):
        return {}


@bp.route("/admin/grievances/<ack_number>")
def grievance_detail(ack_number):
    if not _admin_authenticated():
        return redirect(url_for("main.admin_login", next=request.path))
    grievance = Grievance.query.filter_by(ack_number=ack_number).first_or_404()
    entries = AuditLog.query.filter_by(grievance_id=grievance.id).order_by(AuditLog.timestamp.asc()).all()
    try:
        mapping = json.loads(Path(os.getenv("DEPARTMENTS_CONFIG", "config/departments.json")).read_text(encoding="utf-8")).get("mapping", {})
    except (OSError, ValueError):
        mapping = {}
    taxonomy = _load_taxonomy()
    latest_review = next((entry for entry in reversed(entries) if entry.action in {"classification_accepted", "classification_override"}), None)
    return render_template(
        "grievance_detail.html", grievance=grievance, entries=entries,
        latest_review=latest_review, taxonomy=taxonomy, departments=sorted(set(mapping.values())),
        sla_status=sla_status(grievance), admin_name=current_app.config.get("ADMIN_USERNAME") or "Grievance Admin",
    )


def _build_analytics(*, include_synthetic_dataset=False):
    rows = Grievance.query.order_by(Grievance.submitted_at.desc()).all()
    categories = Counter(row.category for row in rows)
    subcategories = Counter(row.subcategory for row in rows)
    priorities = Counter(str(row.priority) for row in rows)
    departments = Counter(row.routed_department for row in rows if row.routed_department)
    statuses = Counter(row.status for row in rows)
    languages = Counter(row.language for row in rows)
    trends = Counter(as_utc(row.submitted_at).date().isoformat() for row in rows if row.submitted_at)
    classifier = get_classifier() if rows else None
    recurrence_rows = [
        (row.id, row.text, row.category, row.subcategory)
        for row in rows[:200]
    ]
    clusters = cluster_recurring_grievances(
        recurrence_rows,
        vectorizer=(getattr(classifier, "vectorizer", None) or get_similarity_vectorizer()),
        embedder=get_embedder(),
    ) if recurrence_rows else []
    stored_clusters = Counter(row.recurring_cluster_id for row in rows if row.recurring_cluster_id)
    recurring_group_summary = []
    for cluster_id, count in stored_clusters.most_common():
        members = [row for row in rows if row.recurring_cluster_id == cluster_id]
        representative = members[0]
        recurring_group_summary.append({
            "cluster_id": cluster_id,
            "count": count,
            "category": representative.category,
            "subcategory": representative.subcategory,
            "latest_at": max((as_utc(row.submitted_at) for row in members), default=None),
        })
    return {
        "source": "local synthetic/demo submissions; workflow counts are not model evaluation metrics",
        "total_grievances": len(rows),
        "open_grievances": sum(row.status in {"submitted", "routed", "in_progress"} for row in rows),
        "resolved_grievances": statuses.get("resolved", 0),
        "category_distribution": dict(sorted(categories.items())),
        "subcategory_distribution": dict(sorted(subcategories.items())),
        "priority_distribution": dict(sorted(priorities.items())),
        "high_priority_count": sum(row.priority >= 4 for row in rows),
        "low_confidence_count": sum(
            row.confidence_threshold is not None and row.confidence < row.confidence_threshold
            for row in rows
        ),
        "human_review_count": sum(bool(row.manual_review) for row in rows),
        "unresolved_count": sum(row.status not in {"resolved", "rejected", "closed"} for row in rows),
        "duplicate_count": sum(row.duplicate_of_id is not None for row in rows),
        "recurring_clusters": clusters,
        "stored_recurring_group_count": len(stored_clusters),
        "recurring_issue_count": len(stored_clusters),
        "recurring_group_summary": recurring_group_summary,
        "department_workload": dict(sorted(departments.items())),
        "status_distribution": dict(sorted(statuses.items())),
        "language_distribution": dict(sorted(languages.items())),
        "model_version_distribution": dict(sorted(Counter(row.model_version or "unknown" for row in rows).items())),
        "daily_submissions": dict(sorted(trends.items())),
        "synthetic_training_dataset": _synthetic_dataset_analytics() if include_synthetic_dataset else {"available": False},
    }


@lru_cache(maxsize=2)
def _load_synthetic_dataset_analytics(dataset_path: str, modified_ns: int):
    dataset = pd.read_csv(dataset_path)
    category_counts = Counter(dataset["category"].astype(str))
    subcategory_counts = Counter(dataset["subcategory"].astype(str))
    priority_counts = Counter(str(int(value)) for value in dataset["priority"])
    mapping_path = Path(os.getenv("DEPARTMENTS_CONFIG", "config/departments.json"))
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8")).get("mapping", {})
    except (OSError, ValueError):
        mapping = {}
    recommendation_counts = Counter()
    for category, count in category_counts.items():
        if mapping.get(category):
            recommendation_counts[mapping[category]] += count
    cluster_rows = [
        (row.id, row.text, row.category, row.subcategory)
        for row in dataset.head(500).itertuples(index=False)
    ]
    clusters = cluster_recurring_grievances(
        cluster_rows,
        vectorizer=get_similarity_vectorizer(),
        embedder=get_embedder(),
    )
    threshold = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.70"))
    confidence_review_count = None
    backend = os.getenv("MODEL_BACKEND", "indicbert").strip().lower()
    model_version = {
        "indicbert": os.getenv("INDICBERT_MODEL_VERSION", "indicbert_lora_run1"),
        "baseline": "tfidf_baseline_synthetic_v2",
        "muril": "muril_lora",
    }.get(backend, backend)
    if os.getenv("SYNTHETIC_ANALYTICS_RUN_INFERENCE", "0").strip() == "1":
        classifier = get_classifier()
        predictions = [classifier.predict(str(text)) for text in dataset["text"]]
        threshold = float(getattr(classifier, "confidence_threshold", threshold))
        model_version = getattr(classifier, "model_version", classifier.__class__.__name__)
        confidence_review_count = sum(
            float(prediction["confidence"]) < threshold for prediction in predictions
        )
    timestamp_column = next((name for name in ("submitted_at", "timestamp", "created_at") if name in dataset.columns), None)
    return {
        "available": True,
        "source": "data/processed/grievances_synthetic.csv; descriptive synthetic data only",
        "total_grievances": int(len(dataset)),
        "category_distribution": dict(sorted(category_counts.items())),
        "subcategory_distribution": dict(sorted(subcategory_counts.items())),
        "priority_distribution": dict(sorted(priority_counts.items())),
        "high_priority_count": int((dataset["priority"].astype(int) >= 4).sum()),
        "low_confidence_prediction_count": (
            int(confidence_review_count) if confidence_review_count is not None else None
        ),
        "confidence_threshold": threshold,
        "duplicate_count": int(dataset["duplicate_of"].notna().sum()) if "duplicate_of" in dataset.columns else 0,
        "recurring_clusters": clusters,
        "common_themes": [
            {"subcategory": name, "count": int(count)}
            for name, count in subcategory_counts.most_common(10)
        ],
        "department_recommendation_distribution": dict(sorted(recommendation_counts.items())),
        "daily_trends": None,
        "daily_trends_note": "The checked-in synthetic dataset has no timestamp column." if timestamp_column is None else "Available from timestamp column.",
        "unresolved_count": None,
        "human_review_workflow_count": None,
        "workflow_counts_note": "Ticket status and review counts come from app submissions, not training-corpus rows.",
        "model_version": model_version,
    }


def _synthetic_dataset_analytics():
    path = Path(os.getenv("SYNTHETIC_DATASET_PATH", "data/processed/grievances_synthetic.csv"))
    try:
        modified_ns = path.stat().st_mtime_ns
    except OSError:
        return {"available": False, "source": str(path), "note": "Synthetic training dataset is unavailable."}
    return _load_synthetic_dataset_analytics(str(path.resolve()), modified_ns)


@bp.route("/submit", methods=["GET", "POST"])
def submit_grievance():
    if request.method == "GET":
        return render_template("submit.html")
    data = request.get_json(silent=True) or request.form
    if not hasattr(data, "get"):
        return jsonify({"error": "request body must be an object"}), 400
    payload, error = _submission_payload(data.get("text"))
    if error:
        if not request.is_json:
            response, status_code = error
            raw_text = data.get("text")
            return render_template(
                "submit.html", error=(response.get_json() or {}).get("error", "Submission failed."),
                submitted_text=raw_text[:2000] if isinstance(raw_text, str) else "",
            ), status_code
        return error
    if request.is_json:
        return jsonify(payload)
    return render_template("submitted.html", result=payload)


def _submission_payload(raw_value):
    if raw_value is not None and not isinstance(raw_value, str):
        return None, (jsonify({"error": "text must be a string"}), 400)
    raw_text = (raw_value or "").strip()
    if not raw_text:
        return None, (jsonify({"error": "text is required"}), 400)
    if len(raw_text) > 2000:
        return None, (jsonify({"error": "text exceeds 2000 characters"}), 400)
    try:
        grievance, explanation = submit_grievance_service(raw_text)
    except (FileNotFoundError, RuntimeError) as exc:
        db.session.rollback()
        return None, (jsonify({"error": str(exc), "model_backend": os.getenv("MODEL_BACKEND", "indicbert")}), 503)
    except Exception:
        db.session.rollback()
        current_app.logger.error("Submission processing failed")
        return None, (jsonify({"error": "submission processing failed"}), 500)

    payload = {
        "ack_number": grievance.ack_number,
        "track_url": url_for("main.track", ack_number=grievance.ack_number, _external=False),
        "language": grievance.language,
        "script": grievance.script,
        "language_confidence": grievance.language_confidence,
        "category": grievance.category,
        "subcategory": grievance.subcategory,
        "priority": grievance.priority,
        "predicted_category": grievance.predicted_category,
        "predicted_subcategory": grievance.predicted_subcategory,
        "predicted_priority": grievance.predicted_priority,
        "confidence": grievance.confidence,
        "confidence_threshold": grievance.confidence_threshold,
        "model_version": grievance.model_version,
        "manual_review": grievance.manual_review,
        "duplicate_of": grievance.duplicate_of_id,
        "duplicate_similarity": grievance.duplicate_similarity,
        "duplicate_method": grievance.duplicate_method,
        "routed_department": grievance.routed_department,
        "summary": grievance.llm_summary,
        "analysis": grievance.llm_analysis,
        "related_matches": grievance.related_matches or [],
        "recurring_cluster_id": grievance.recurring_cluster_id,
        "explanation": explanation,
    }
    return payload, None


@bp.route("/track/<ack_number>")
def track(ack_number):
    grievance = Grievance.query.filter_by(ack_number=ack_number).first_or_404()
    last_updated = db.session.query(db.func.max(AuditLog.timestamp)).filter_by(grievance_id=grievance.id).scalar() or grievance.submitted_at
    return render_template("track.html", grievance=grievance, sla_status=sla_status(grievance), last_updated=last_updated)


@bp.route("/api/grievances/<ack_number>")
@bp.route("/api/v1/grievances/<ack_number>")
def api_grievance(ack_number):
    grievance = Grievance.query.filter_by(ack_number=ack_number).first_or_404()
    return jsonify({
        "ack_number": grievance.ack_number,
        "status": grievance.status,
        "category": grievance.category,
        "subcategory": grievance.subcategory,
        "priority": grievance.priority,
        "predicted_category": grievance.predicted_category,
        "predicted_subcategory": grievance.predicted_subcategory,
        "predicted_priority": grievance.predicted_priority,
        "confidence": grievance.confidence,
        "confidence_threshold": grievance.confidence_threshold,
        "requires_human_review": grievance.manual_review,
        "model_version": grievance.model_version,
        "routed_department": grievance.routed_department,
        "duplicate_of": grievance.duplicate_of_id,
        "duplicate_similarity": grievance.duplicate_similarity,
        "related_matches": grievance.related_matches or [],
        "recurring_cluster_id": grievance.recurring_cluster_id,
        "llm_analysis": grievance.llm_analysis,
        "submitted_at": as_utc(grievance.submitted_at).isoformat(),
        "sla_deadline": as_utc(grievance.sla_deadline).isoformat() if grievance.sla_deadline else None,
        "sla_status": sla_status(grievance),
        "resolved_at": as_utc(grievance.resolved_at).isoformat() if grievance.resolved_at else None,
    })


@bp.route("/admin/grievances/<ack_number>/status", methods=["POST"])
@bp.route("/api/v1/grievances/<ack_number>/status", methods=["POST"])
def update_status(ack_number):
    denied = _require_admin()
    if denied:
        return denied
    grievance = Grievance.query.filter_by(ack_number=ack_number).first_or_404()
    data = request.get_json(silent=True) or request.form
    if not hasattr(data, "get"):
        return jsonify({"error": "request body must be an object"}), 400
    status_value, actor_value, note_value = data.get("status"), data.get("actor", "admin"), data.get("note", "")
    if not all(isinstance(v, str) for v in (status_value, actor_value, note_value)):
        return jsonify({"error": "status, actor, and note must be strings"}), 400
    status, actor, note = status_value.strip(), actor_value.strip(), note_value.strip()
    if status not in STATUS_VALUES:
        return jsonify({"error": f"invalid status; use one of {sorted(STATUS_VALUES)}"}), 400
    old = grievance.status
    if status != old and status not in STATUS_TRANSITIONS.get(old, set()):
        return jsonify({"error": f"status transition {old} -> {status} is not allowed"}), 409
    if status == "routed" and not grievance.routed_department:
        return jsonify({"error": "a department must be assigned before marking a ticket routed"}), 409
    if status == "in_progress" and not grievance.routed_department:
        return jsonify({"error": "a department must be assigned before work starts"}), 409
    grievance.status = status
    if status == "resolved":
        grievance.resolved_at = utc_now()
    _audit(grievance.id, "status_change", actor, f"{old} -> {status}; {note}".strip("; "))
    db.session.commit()
    return jsonify({"ack_number": ack_number, "old_status": old, "status": status})


@bp.route("/admin/grievances/<ack_number>/override", methods=["POST"])
@bp.route("/api/v1/grievances/<ack_number>/override", methods=["POST"])
def override_route(ack_number):
    denied = _require_admin()
    if denied:
        return denied
    grievance = Grievance.query.filter_by(ack_number=ack_number).first_or_404()
    data = request.get_json(silent=True) or request.form
    if not hasattr(data, "get"):
        return jsonify({"error": "request body must be an object"}), 400
    department_value, actor_value, note_value = data.get("department"), data.get("actor", "admin"), data.get("note", "")
    if not all(isinstance(v, str) for v in (department_value, actor_value, note_value)):
        return jsonify({"error": "department, actor, and note must be strings"}), 400
    department, actor, note = department_value.strip(), actor_value.strip(), note_value.strip()
    if not department:
        return jsonify({"error": "department is required"}), 400
    try:
        allowed_departments = set(json.loads(Path(os.getenv("DEPARTMENTS_CONFIG", "config/departments.json")).read_text(encoding="utf-8")).get("mapping", {}).values())
    except (OSError, ValueError):
        allowed_departments = set()
    if department not in allowed_departments:
        return jsonify({"error": "department must be present in configured demo department mappings"}), 400
    if grievance.status in {"resolved", "rejected", "duplicate", "closed"}:
        return jsonify({"error": "cannot override routing for a terminal ticket"}), 409
    old = grievance.routed_department
    old_status = grievance.status
    grievance.routed_department = department
    grievance.routing_at = grievance.routing_at or utc_now()
    grievance.manual_review = False
    if old_status == "submitted":
        grievance.status = "routed"
        _audit(grievance.id, "status_change", actor, "submitted -> routed; route approved")
    _audit(grievance.id, "override", actor, f"{old} -> {department}; {note}".strip("; "))
    db.session.commit()
    return jsonify({"ack_number": ack_number, "previous_department": old, "department": department})


@bp.route("/api/v1/grievances/<ack_number>/classification", methods=["POST"])
def override_classification(ack_number):
    denied = _require_admin()
    if denied:
        return denied
    grievance = Grievance.query.filter_by(ack_number=ack_number).first_or_404()
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400
    if grievance.status in {"resolved", "rejected", "duplicate", "closed"}:
        return jsonify({"error": "cannot correct classification for a terminal ticket"}), 409
    actor_value = data.get("actor", "admin")
    if not isinstance(actor_value, str):
        return jsonify({"error": "actor must be a string"}), 400
    note_value = data.get("note", "")
    if not isinstance(note_value, str):
        return jsonify({"error": "note must be a string"}), 400
    try:
        taxonomy_path = os.getenv("TAXONOMY_CONFIG", "config/taxonomy.json")
        taxonomy = json.loads(Path(taxonomy_path).read_text(encoding="utf-8"))["categories"]
    except (OSError, ValueError, KeyError):
        return jsonify({"error": "taxonomy configuration is unavailable"}), 503
    category = data.get("category", grievance.category)
    subcategory = data.get("subcategory", grievance.subcategory)
    priority = data.get("priority", grievance.priority)
    if isinstance(priority, bool):
        return jsonify({"error": "priority must be an integer from 1 to 5"}), 400
    if not isinstance(category, str) or not isinstance(subcategory, str) or category not in taxonomy or subcategory not in taxonomy[category].get("subcategories", []):
        return jsonify({"error": "subcategory must belong to a configured category"}), 400
    try:
        priority = int(priority)
    except (ValueError, TypeError):
        return jsonify({"error": "priority must be an integer from 1 to 5"}), 400
    if not 1 <= priority <= 5:
        return jsonify({"error": "priority must be an integer from 1 to 5"}), 400
    actor = actor_value.strip()
    old = f"{grievance.category}/{grievance.subcategory}/P{grievance.priority}"
    classification_changed = (
        grievance.category,
        grievance.subcategory,
        grievance.priority,
    ) != (category, subcategory, priority)
    old_department = grievance.routed_department
    try:
        department_mapping = json.loads(Path(os.getenv("DEPARTMENTS_CONFIG", "config/departments.json")).read_text(encoding="utf-8")).get("mapping", {})
    except (OSError, ValueError):
        return jsonify({"error": "department configuration is unavailable"}), 503
    new_department = department_mapping.get(category)
    if not new_department:
        return jsonify({"error": "corrected category has no configured department"}), 409
    old_status = grievance.status
    grievance.category, grievance.subcategory, grievance.priority = category, subcategory, priority
    grievance.routed_department = new_department
    grievance.routing_at = grievance.routing_at or utc_now()
    grievance.manual_review = False
    if old_status == "submitted":
        grievance.status = "routed"
        _audit(grievance.id, "status_change", actor, "submitted -> routed; classification approved")
    action = "classification_override" if classification_changed else "classification_accepted"
    detail = (
        f"{old} -> {category}/{subcategory}/P{priority}"
        if classification_changed
        else f"accepted {category}/{subcategory}/P{priority}"
    )
    if note_value.strip():
        detail = f"{detail}; reviewer note: {note_value.strip()}"
    _audit(grievance.id, action, actor, detail)
    if old_department != new_department:
        _audit(grievance.id, "routing_after_classification_override", actor, f"{old_department} -> {new_department}")
    db.session.commit()
    return jsonify({"ack_number": ack_number, "category": category, "subcategory": subcategory, "priority": priority, "routed_department": new_department})


@bp.route("/api/v1/grievances/<ack_number>/summary-review", methods=["POST"])
def review_summary(ack_number):
    denied = _require_admin()
    if denied:
        return denied
    grievance = Grievance.query.filter_by(ack_number=ack_number).first_or_404()
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400
    factuality_value = data.get("factuality")
    if not isinstance(factuality_value, str):
        return jsonify({"error": "factuality must be a string"}), 400
    factuality = factuality_value.strip()
    if factuality not in {"factual", "partially_factual", "inaccurate"}:
        return jsonify({"error": "factuality must be factual, partially_factual, or inaccurate"}), 400
    actor_value, note_value = data.get("actor", "admin"), data.get("note", "")
    if not isinstance(actor_value, str) or not isinstance(note_value, str):
        return jsonify({"error": "actor and note must be strings"}), 400
    actor, note = actor_value.strip(), note_value.strip()
    grievance.summary_factuality = factuality
    grievance.summary_review_note = redact_pii(note)
    grievance.summary_reviewed_by = redact_pii(actor)
    grievance.summary_reviewed_at = utc_now()
    _audit(grievance.id, "summary_review", actor, f"factuality={factuality}; {note}".strip("; "))
    db.session.commit()
    return jsonify({"ack_number": ack_number, "factuality": factuality, "reviewed_at": as_utc(grievance.summary_reviewed_at).isoformat()})


@bp.route("/api/v1/grievances/<ack_number>/explanation")
def grievance_explanation(ack_number):
    denied = _require_admin()
    if denied:
        return denied
    grievance = Grievance.query.filter_by(ack_number=ack_number).first_or_404()
    if grievance.explanation is not None:
        return jsonify({"available": True, "features": grievance.explanation})
    classifier = get_classifier()
    if not _shap_lock.acquire(blocking=False):
        return jsonify({"available": False, "features": [], "error": "A SHAP explanation is already running. Please retry shortly."}), 503
    result_queue = queue.Queue(maxsize=1)

    def calculate_explanation():
        try:
            result_queue.put((True, classifier.explain(grievance.text)))
        except Exception as exc:
            result_queue.put((False, exc))
        finally:
            _shap_lock.release()

    worker = threading.Thread(target=calculate_explanation, name="grieveai-shap", daemon=True)
    worker.start()
    try:
        succeeded, result = result_queue.get(timeout=current_app.config["SHAP_TIMEOUT_SECONDS"])
        if not succeeded:
            raise result
        explanation = result
    except queue.Empty:
        current_app.logger.warning("SHAP explanation timed out for grievance id=%s", grievance.id)
        return jsonify({"available": False, "features": [], "error": "SHAP explanation timed out. Retry to try again."}), 504
    except Exception:
        current_app.logger.exception("On-demand explanation failed for grievance id=%s", grievance.id)
        return jsonify({"available": False, "features": [], "error": "SHAP explanation failed. Retry to try again."}), 500
    grievance.explanation = explanation
    db.session.commit()
    return jsonify({"available": True, "features": explanation})


@bp.route("/health")
def health():
    try:
        Grievance.query.limit(1).all()
        db_ok = True
    except Exception:
        db_ok = False
    model_ok = False
    model_version = None
    model_error = None
    try:
        classifier = get_classifier()
        model_ok = True
        model_version = getattr(classifier, "model_version", classifier.__class__.__name__)
    except Exception as exc:
        model_error = str(exc)
    status = "ok" if db_ok and model_ok else "degraded"
    return jsonify({
        "status": status,
        "database": db_ok,
        "model": {"ready": model_ok, "backend": os.getenv("MODEL_BACKEND", "indicbert"), "version": model_version, "error": model_error},
        "service": "GrieveAI",
    }), (200 if status == "ok" else 503)


@bp.route("/api/v1/health")
def health_v1():
    return health()


@bp.route("/api/v1/grievances", methods=["POST"])
def submit_api_v1():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    data = request.get_json(silent=True)
    if not hasattr(data, "get"):
        return jsonify({"error": "request body must be an object"}), 400
    payload, error = _submission_payload(data.get("text"))
    return error if error else jsonify(payload)


@bp.route("/api/v1/review-queue")
def review_queue():
    denied = _require_admin()
    if denied:
        return denied
    rows = Grievance.query.filter_by(manual_review=True, status="submitted").order_by(Grievance.submitted_at.asc()).all()
    return jsonify({"items": [{"ack_number": g.ack_number, "category": g.category, "subcategory": g.subcategory, "confidence": g.confidence, "submitted_at": as_utc(g.submitted_at).isoformat()} for g in rows]})


@bp.route("/api/v1/admin/metrics")
def admin_metrics():
    denied = _require_admin()
    if denied:
        return denied
    routed = Grievance.query.filter_by(automated_route=True).count()
    overrides = (
        db.session.query(Grievance.id)
        .join(AuditLog, AuditLog.grievance_id == Grievance.id)
        .filter(AuditLog.action == "override", Grievance.automated_route.is_(True))
        .distinct().count()
    )
    return jsonify({"source": "observed local application events; not research evaluation", "submissions": Grievance.query.count(), "manual_review_queue": Grievance.query.filter_by(manual_review=True, status="submitted").count(), "automated_routing_decisions": routed, "automated_decisions_overridden": overrides, "override_rate": round(overrides / routed, 4) if routed else None, "override_rate_denominator": "automatically routed tickets; manually reviewed cases are excluded", "routing_accuracy": None, "routing_accuracy_note": "Requires validated gold department labels."})


@bp.route("/api/v1/admin/analytics")
def admin_analytics():
    denied = _require_admin()
    if denied:
        return denied
    return jsonify(_build_analytics(include_synthetic_dataset=True))


@bp.route("/admin/audit/<ack_number>")
def audit_trail(ack_number):
    if not _admin_authenticated():
        return redirect(url_for("main.admin_login"))
    grievance = Grievance.query.filter_by(ack_number=ack_number).first_or_404()
    entries = AuditLog.query.filter_by(grievance_id=grievance.id).order_by(AuditLog.timestamp.asc()).all()
    return render_template("audit.html", grievance=grievance, entries=entries)
