"""initial: enable pgvector and create documents table

Revision ID: 0001
Revises:
Create Date: 2026-03-26
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 384


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("metadata", sa.dialects.postgresql.JSONB(), nullable=True),
    )

    # HNSW index for approximate nearest-neighbour search (cosine).
    # HNSW supports concurrent inserts, requires no rebuild after bulk load,
    # and performs correctly at any dataset size — unlike IVFFlat which
    # needs lists << row_count to avoid under-probing.
    op.execute(
        "CREATE INDEX documents_embedding_hnsw_idx "
        "ON documents USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS documents_embedding_hnsw_idx")
    op.drop_table("documents")
