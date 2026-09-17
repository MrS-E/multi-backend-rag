"""initial schema: knowledge_bases, documents, sources

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-17

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_bases",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("owner_id", sa.String(length=255), nullable=True),
        sa.Column("overrides", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "sources",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("kb_id", sa.Uuid(), nullable=False),
        sa.Column("type", sa.String(length=64), nullable=False),
        sa.Column("mode", sa.Enum("event", "pull", name="sourcemode", native_enum=False), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("sync_interval_s", sa.Integer(), nullable=False),
        sa.Column("cursor", sa.JSON(), nullable=True),
        sa.Column("last_sync_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["kb_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_sources_kb_id"), "sources", ["kb_id"], unique=False)
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("kb_id", sa.Uuid(), nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=True),
        sa.Column("uri", sa.String(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("mime", sa.String(length=255), nullable=False),
        sa.Column("raw_path", sa.String(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["kb_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_documents_kb_id"), "documents", ["kb_id"], unique=False)
    op.create_index(op.f("ix_documents_source_id"), "documents", ["source_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_documents_source_id"), table_name="documents")
    op.drop_index(op.f("ix_documents_kb_id"), table_name="documents")
    op.drop_table("documents")
    op.drop_index(op.f("ix_sources_kb_id"), table_name="sources")
    op.drop_table("sources")
    op.drop_table("knowledge_bases")
