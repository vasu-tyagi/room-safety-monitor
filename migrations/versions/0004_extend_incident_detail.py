"""extend incidents table with detail columns for Slice 8a

Adds five nullable columns used by the Inspector page, the feedback loop,
and the full incident detail endpoint.

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-19
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("incidents", sa.Column("vlm_prompt", sa.Text(), nullable=True))
    op.add_column("incidents", sa.Column("fired_rules", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("incidents", sa.Column("confidence_breakdown", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("incidents", sa.Column("kb_matches", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("incidents", sa.Column("operator_decision", sa.String(64), nullable=True))


def downgrade() -> None:
    op.drop_column("incidents", "operator_decision")
    op.drop_column("incidents", "kb_matches")
    op.drop_column("incidents", "confidence_breakdown")
    op.drop_column("incidents", "fired_rules")
    op.drop_column("incidents", "vlm_prompt")
