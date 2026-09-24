"""Student submission, tracking and admin workflow routes."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import os

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, session, url_for

from app.models_db import AuditLog, Grievance, db
from app.services import _audit, get_classifier, submit_grievance as submit_grievance_service

bp = Blueprint("main", __name__)

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
    at = at or datetime.utcnow()
    if grievance.resolved_at:
        return "met" if grievance.resolved_at <= grievance.sla_deadline else "overdue"
    return "overdue" if at > grievance.sla_deadline else "on_track"


def _require_admin():
    expected = os.getenv("ADMIN_TOKEN", "").strip()
    if not expected:
        return jsonify({"error": "admin actions are disabled; configure ADMIN_TOKEN"}), 503
    supplied = request.headers.get("X-ADMIN-TOKEN", "")
    import hmac
    if session.get("admin_authenticated") is not True and not hmac.compare_digest(supplied, expected):
        return jsonify({"error": "admin authentication required"}), 401
    return None


@bp.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "GET":
        return render_template("admin_login.html", configured=bool(os.getenv("ADMIN_TOKEN", "").strip()))
    expected = os.getenv("ADMIN_TOKEN", "").strip()
    supplied = (request.form.get("token") or "").strip()
    import hmac
    if not expected:
        return render_template("admin_login.html", configured=False, error="Admin actions are disabled until ADMIN_TOKEN is configured."), 503
    if not hmac.compare_digest(supplied, expected):
        return render_template("admin_login.html", configured=True, error="Invalid admin token."), 401
    session["admin_authenticated"] = True
    return redirect(url_for("main.index"))


@bp.route("/admin/logout", methods=["POST"])
def admin_logout():
    session.clear()
    return redirect(url_for("main.admin_login"))


@bp.route("/")
def index():
    if session.get("admin_authenticated") is not True:
        return redirect(url_for("main.admin_login"))
    query = Grievance.query
    department = request.args.get("department", "").strip()
    status = request.args.get("status", "").strip()
    if department:
        query = query.filter_by(routed_department=department)
    if status == "review":
        query = query.filter_by(manual_review=True, status="submitted")
    elif status:
        query = query.filter_by(status=status)
    grievances = query.order_by(Grievance.submitted_at.desc()).limit(100).all()
    counts = {
        "total": Grievance.query.count(),
        "review": Grievance.query.filter_by(manual_review=True, status="submitted").count(),
        "open": Grievance.query.filter(Grievance.status.in_(["submitted", "routed", "in_progress"])).count(),
        "resolved": Grievance.query.filter_by(status="resolved").count(),
        "overrides": AuditLog.query.filter_by(action="override").count(),
    }
    try:
        departments = json.loads(Path(os.getenv("DEPARTMENTS_CONFIG", "config/departments.json")).read_text(encoding="utf-8")).get("mapping", {})
    except (OSError, ValueError):
        departments = {}
    return render_template("dashboard.html", grievances=grievances, counts=counts, departments=sorted(set(departments.values())), selected_department=department, selected_status=status, now=datetime.utcnow())


@bp.route("/submit", methods=["GET", "POST"])
def submit_grievance():
    if request.method == "GET":
        return render_template("submit.html")
    data = request.get_json(silent=True) or request.form
    if not hasattr(data, "get"):
        return jsonify({"error": "request body must be an object"}), 400
    payload, error = _submission_payload(data.get("text"))
    if error:
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
    except FileNotFoundError:
        db.session.rollback()
        return None, (jsonify({"error": "model checkpoint unavailable"}), 503)
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
        "confidence": grievance.confidence,
        "manual_review": grievance.manual_review,
        "duplicate_of": grievance.duplicate_of_id,
        "duplicate_similarity": grievance.duplicate_similarity,
        "duplicate_method": grievance.duplicate_method,
        "routed_department": grievance.routed_department,
        "summary": grievance.llm_summary,
        "explanation": explanation,
    }
    return payload, None


@bp.route("/track/<ack_number>")
def track(ack_number):
    grievance = Grievance.query.filter_by(ack_number=ack_number).first_or_404()
    return render_template("track.html", grievance=grievance, sla_status=sla_status(grievance))


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
        "routed_department": grievance.routed_department,
        "submitted_at": grievance.submitted_at.isoformat(),
        "sla_deadline": grievance.sla_deadline.isoformat() if grievance.sla_deadline else None,
        "sla_status": sla_status(grievance),
        "resolved_at": grievance.resolved_at.isoformat() if grievance.resolved_at else None,
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
        grievance.resolved_at = datetime.utcnow()
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
    grievance.routing_at = grievance.routing_at or datetime.utcnow()
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
    grievance.routing_at = grievance.routing_at or datetime.utcnow()
    grievance.manual_review = False
    if old_status == "submitted":
        grievance.status = "routed"
        _audit(grievance.id, "status_change", actor, "submitted -> routed; classification approved")
    _audit(grievance.id, "classification_override", actor, f"{old} -> {category}/{subcategory}/P{priority}")
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
    grievance.summary_review_note = note
    grievance.summary_reviewed_by = actor
    grievance.summary_reviewed_at = datetime.utcnow()
    _audit(grievance.id, "summary_review", actor, f"factuality={factuality}; {note}".strip("; "))
    db.session.commit()
    return jsonify({"ack_number": ack_number, "factuality": factuality, "reviewed_at": grievance.summary_reviewed_at.isoformat()})


@bp.route("/api/v1/grievances/<ack_number>/explanation")
def grievance_explanation(ack_number):
    denied = _require_admin()
    if denied:
        return denied
    grievance = Grievance.query.filter_by(ack_number=ack_number).first_or_404()
    try:
        explanation = get_classifier().explain(grievance.text)
    except Exception:
        current_app.logger.exception("On-demand explanation failed for grievance id=%s", grievance.id)
        return jsonify({"available": False, "features": []})
    return jsonify({"available": True, "features": explanation})


@bp.route("/health")
def health():
    try:
        Grievance.query.limit(1).all()
        db_ok = True
    except Exception:
        db_ok = False
    return jsonify({"status": "ok" if db_ok else "degraded", "database": db_ok, "service": "GrieveAI"}), (200 if db_ok else 503)


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
    return jsonify({"items": [{"ack_number": g.ack_number, "category": g.category, "subcategory": g.subcategory, "confidence": g.confidence, "submitted_at": g.submitted_at.isoformat()} for g in rows]})


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


@bp.route("/admin/audit/<ack_number>")
def audit_trail(ack_number):
    if session.get("admin_authenticated") is not True:
        return redirect(url_for("main.admin_login"))
    grievance = Grievance.query.filter_by(ack_number=ack_number).first_or_404()
    entries = AuditLog.query.filter_by(grievance_id=grievance.id).order_by(AuditLog.timestamp.asc()).all()
    return render_template("audit.html", grievance=grievance, entries=entries)
