"""SQLAlchemy models for grievances and an auditable state-transition log."""

from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


def utc_now() -> datetime:
    """Return an aware UTC timestamp for new application data."""
    return datetime.now(timezone.utc)


def as_utc(value: datetime) -> datetime:
    """Normalize database-naive or aware timestamps to aware UTC."""
    if value.tzinfo is None:
        # Existing SQLite DateTime columns round-trip UTC without tzinfo.
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class Grievance(db.Model):
    __tablename__ = "grievances"

    id = db.Column(db.Integer, primary_key=True)
    ack_number = db.Column(db.String(32), unique=True, nullable=False, index=True)
    text = db.Column(db.Text, nullable=False)
    language = db.Column(db.String(16), nullable=False)
    script = db.Column(db.String(16), nullable=False, default="unknown")
    language_confidence = db.Column(db.Float)
    category = db.Column(db.String(64), nullable=False)
    subcategory = db.Column(db.String(64), nullable=False)
    priority = db.Column(db.Integer, nullable=False)
    predicted_category = db.Column(db.String(64))
    predicted_subcategory = db.Column(db.String(64))
    predicted_priority = db.Column(db.Integer)
    priority_raw = db.Column(db.Float)
    confidence = db.Column(db.Float, nullable=False)
    confidence_threshold = db.Column(db.Float)
    subcategory_confidence = db.Column(db.Float)
    status = db.Column(db.String(32), default="submitted", nullable=False, index=True)
    routed_department = db.Column(db.String(80))
    model_department = db.Column(db.String(80))
    manual_review = db.Column(db.Boolean, default=False, nullable=False, index=True)
    automated_route = db.Column(db.Boolean, default=False, nullable=False, index=True)
    duplicate_of_id = db.Column(db.Integer, db.ForeignKey("grievances.id"), nullable=True)
    duplicate_similarity = db.Column(db.Float)
    duplicate_method = db.Column(db.String(40))
    llm_summary = db.Column(db.Text)
    summary_provider = db.Column(db.String(32))
    submitted_at = db.Column(db.DateTime, default=utc_now, nullable=False, index=True)
    sla_deadline = db.Column(db.DateTime)
    resolved_at = db.Column(db.DateTime)
    summary_factuality = db.Column(db.String(24))
    summary_review_note = db.Column(db.Text)
    summary_reviewed_by = db.Column(db.String(80))
    summary_reviewed_at = db.Column(db.DateTime)
    prediction_at = db.Column(db.DateTime)
    routing_at = db.Column(db.DateTime)
    manual_review_at = db.Column(db.DateTime)


class AuditLog(db.Model):
    __tablename__ = "audit_log"

    id = db.Column(db.Integer, primary_key=True)
    grievance_id = db.Column(db.Integer, db.ForeignKey("grievances.id"), nullable=False, index=True)
    action = db.Column(db.String(50), nullable=False)
    actor = db.Column(db.String(80), nullable=False)
    detail = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=utc_now, nullable=False, index=True)
