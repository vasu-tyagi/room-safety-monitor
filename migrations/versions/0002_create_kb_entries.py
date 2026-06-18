"""create kb_entries table with pgvector HNSW index

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-19
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("""
        CREATE TABLE kb_entries (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            embedding vector(768) NOT NULL,
            prompt_text TEXT NOT NULL,
            rationale_text TEXT NOT NULL,
            label VARCHAR(64) NOT NULL,
            operator_decision VARCHAR(64) NOT NULL DEFAULT 'pending',
            metadata JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute(
        "CREATE INDEX ix_kb_entries_embedding_hnsw "
        "ON kb_entries USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_kb_entries_embedding_hnsw")
    op.execute("DROP TABLE IF EXISTS kb_entries")
