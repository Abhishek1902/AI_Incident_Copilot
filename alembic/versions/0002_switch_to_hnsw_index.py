"""switch embedding index from IVFFlat to HNSW

IVFFlat under-probes on small datasets (requires lists << row_count).
HNSW is graph-based: works correctly at any size, no pre-training step,
and supports concurrent inserts without a full rebuild.

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-26
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS documents_embedding_cosine_idx")
    op.execute("DROP INDEX IF EXISTS documents_embedding_hnsw_idx")
    op.execute(
        "CREATE INDEX documents_embedding_hnsw_idx "
        "ON documents USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS documents_embedding_hnsw_idx")
    op.execute(
        "CREATE INDEX documents_embedding_cosine_idx "
        "ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )
