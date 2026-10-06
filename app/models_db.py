"""
SQLAlchemy models for the ticketing backend (Week 4).

Structured, relational data lives here (PostgreSQL in production, SQLite
for local dev) - grievance records, categories, SLA timers, audit trail.
Raw text / SHAP payloads / embeddings would go to MongoDB only if data
volume ever justifies it (see README note on the dual-database decision -
at VCET's scale, this table alone is very likely sufficient).
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Grievance(db.Model):
    __tablename__ = "grievances"

    id = db.Column(db.Integer, primary_key=True)
    ack_number = db.Column(db.String(20), unique=True, nullable=False)
    text = db.Column(db.Text, nullable=False)
    language = db.Column(db.String(10))
    category = db.Column(db.String(50))
    subcategory = db.Column(db.String(50))
    priority = db.Column(db.Integer)
    confidence = db.Column(db.Float)
    status = db.Column(db.String(20), default="submitted")  # submitted -> routed -> in_progress -> resolved
    routed_department = db.Column(db.String(50))
    manual_review = db.Column(db.Boolean, default=False)
    duplicate_of_id = db.Column(db.Integer, db.ForeignKey("grievances.id"), nullable=True)
    llm_summary = db.Column(db.Text)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    sla_deadline = db.Column(db.DateTime)
    resolved_at = db.Column(db.DateTime)


class AuditLog(db.Model):
    __tablename__ = "audit_log"

    id = db.Column(db.Integer, primary_key=True)
    grievance_id = db.Column(db.Integer, db.ForeignKey("grievances.id"), nullable=False)
    action = db.Column(db.String(50))          # e.g. "routed", "override", "resolved"
    actor = db.Column(db.String(50))            # "system" or staff username
    detail = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
