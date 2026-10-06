"""Create grievance and audit tables.

Revision ID: 0001_initial
Revises:
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    existing = sa.inspect(bind).get_table_names()
    if "grievances" not in existing:
        op.create_table(
            "grievances",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("ack_number", sa.String(length=32), nullable=False, unique=True),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column("language", sa.String(length=16), nullable=False),
            sa.Column("script", sa.String(length=16), nullable=False, server_default="unknown"),
            sa.Column("language_confidence", sa.Float()),
            sa.Column("category", sa.String(length=64), nullable=False),
            sa.Column("subcategory", sa.String(length=64), nullable=False),
            sa.Column("priority", sa.Integer(), nullable=False),
            sa.Column("predicted_category", sa.String(length=64)),
            sa.Column("predicted_subcategory", sa.String(length=64)),
            sa.Column("predicted_priority", sa.Integer()),
            sa.Column("priority_raw", sa.Float()),
            sa.Column("confidence", sa.Float(), nullable=False),
            sa.Column("confidence_threshold", sa.Float()),
            sa.Column("subcategory_confidence", sa.Float()),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="submitted"),
            sa.Column("routed_department", sa.String(length=80)),
            sa.Column("model_department", sa.String(length=80)),
            sa.Column("manual_review", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("automated_route", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("duplicate_of_id", sa.Integer(), sa.ForeignKey("grievances.id")),
            sa.Column("duplicate_similarity", sa.Float()),
            sa.Column("duplicate_method", sa.String(length=40)),
            sa.Column("llm_summary", sa.Text()),
            sa.Column("summary_provider", sa.String(length=32)),
            sa.Column("submitted_at", sa.DateTime(), nullable=False),
            sa.Column("sla_deadline", sa.DateTime()),
            sa.Column("resolved_at", sa.DateTime()),
            sa.Column("summary_factuality", sa.String(length=24)),
            sa.Column("summary_review_note", sa.Text()),
            sa.Column("summary_reviewed_by", sa.String(length=80)),
            sa.Column("summary_reviewed_at", sa.DateTime()),
            sa.Column("prediction_at", sa.DateTime()),
            sa.Column("routing_at", sa.DateTime()),
            sa.Column("manual_review_at", sa.DateTime()),
        )
        op.create_index("ix_grievances_ack_number", "grievances", ["ack_number"], unique=True)
        op.create_index("ix_grievances_status", "grievances", ["status"])
        op.create_index("ix_grievances_manual_review", "grievances", ["manual_review"])
        op.create_index("ix_grievances_automated_route", "grievances", ["automated_route"])
        op.create_index("ix_grievances_submitted_at", "grievances", ["submitted_at"])
    if "audit_log" not in existing:
        op.create_table(
            "audit_log",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("grievance_id", sa.Integer(), sa.ForeignKey("grievances.id"), nullable=False),
            sa.Column("action", sa.String(length=50), nullable=False),
            sa.Column("actor", sa.String(length=80), nullable=False),
            sa.Column("detail", sa.Text()),
            sa.Column("timestamp", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_audit_log_grievance_id", "audit_log", ["grievance_id"])
        op.create_index("ix_audit_log_timestamp", "audit_log", ["timestamp"])


def downgrade():
    raise RuntimeError(
        "The initial migration may adopt an existing database; refusing to drop "
        "grievance data during downgrade. Back up and migrate data explicitly."
    )
