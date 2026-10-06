"""Persist model explanations and index common grievance filters.

Revision ID: 0002_explanations_indexes
Revises: 0001_initial_schema
"""
from alembic import op
import sqlalchemy as sa

revision = "0002_explanations_indexes"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("grievances", sa.Column("explanation", sa.JSON(), nullable=True))
    op.create_index("ix_grievances_routed_department", "grievances", ["routed_department"])
    op.create_index("ix_grievances_category", "grievances", ["category"])
    op.create_index("ix_grievances_priority", "grievances", ["priority"])
    op.create_index("ix_grievances_sla_deadline", "grievances", ["sla_deadline"])
    op.create_index("ix_grievances_duplicate_of_id", "grievances", ["duplicate_of_id"])


def downgrade():
    op.drop_index("ix_grievances_duplicate_of_id", table_name="grievances")
    op.drop_index("ix_grievances_sla_deadline", table_name="grievances")
    op.drop_index("ix_grievances_priority", table_name="grievances")
    op.drop_index("ix_grievances_category", table_name="grievances")
    op.drop_index("ix_grievances_routed_department", table_name="grievances")
    op.drop_column("grievances", "explanation")
