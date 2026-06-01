"""add incident_events table

Revision ID: 0004
Revises: 0003
Create Date: 2026-03-28

The vector extension is already enabled by migration 0001.
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

EMBEDDING_DIM = 384


def upgrade() -> None:
    op.create_table(
        "incident_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        # SHA-256 content hash — enforces deduplication at the DB level.
        sa.Column("event_id", sa.String(64), nullable=False),
        # Signal type: log | deployment | alert | metadata
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("service", sa.String(128), nullable=False),
        sa.Column("severity", sa.String(16), nullable=True),
        # Temporal anchor — the most critical column for incident correlation.
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("correlation_id", sa.String(64), nullable=True),
        sa.Column("metadata", sa.dialects.postgresql.JSONB(), nullable=True),
    )

    # Unique constraint for ON CONFLICT deduplication.
    op.create_unique_constraint("uq_incident_events_event_id", "incident_events", ["event_id"])

    # Primary access patterns:
    #   1. Time-bounded queries: "events in the 30 min before an incident"
    op.create_index("ix_incident_events_occurred_at", "incident_events", [sa.text("occurred_at DESC")])

    #   2. Per-service timeline: "all errors from payment-service this hour"
    op.create_index(
        "ix_incident_events_service_time",
        "incident_events",
        ["service", sa.text("occurred_at DESC")],
    )

    #   3. Type filtering: "only deployments in this window"
    op.create_index("ix_incident_events_event_type", "incident_events", ["event_type"])

    #   4. Correlation group lookup.
    op.create_index("ix_incident_events_correlation_id", "incident_events", ["correlation_id"])

    #   5. HNSW vector index for ANN semantic search — same parameters as documents table.
    op.execute(
        "CREATE INDEX ix_incident_events_embedding_hnsw "
        "ON incident_events USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_incident_events_embedding_hnsw")
    op.drop_index("ix_incident_events_correlation_id", table_name="incident_events")
    op.drop_index("ix_incident_events_event_type", table_name="incident_events")
    op.drop_index("ix_incident_events_service_time", table_name="incident_events")
    op.drop_index("ix_incident_events_occurred_at", table_name="incident_events")
    op.drop_constraint("uq_incident_events_event_id", "incident_events", type_="unique")
    op.drop_table("incident_events")
