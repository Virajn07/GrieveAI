"""Persist model version, similarity context, recurring groups and LLM analysis.

Revision ID: 0003_inference_context
Revises: 0002_explanations_indexes
"""
from alembic import op
import sqlalchemy as sa


revision = "0003_inference_context"
down_revision = "0002_explanations_indexes"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("grievances", sa.Column("model_version", sa.String(length=80), nullable=True))
    op.add_column("grievances", sa.Column("related_matches", sa.JSON(), nullable=True))
    op.add_column("grievances", sa.Column("recurring_cluster_id", sa.String(length=40), nullable=True))
    op.add_column("grievances", sa.Column("llm_analysis", sa.JSON(), nullable=True))
    op.create_index("ix_grievances_recurring_cluster_id", "grievances", ["recurring_cluster_id"])


def downgrade():
    op.drop_index("ix_grievances_recurring_cluster_id", table_name="grievances")
    op.drop_column("grievances", "llm_analysis")
    op.drop_column("grievances", "recurring_cluster_id")
    op.drop_column("grievances", "related_matches")
    op.drop_column("grievances", "model_version")
