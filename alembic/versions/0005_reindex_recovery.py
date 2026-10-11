"""persist reindex takeover fencing metadata

Revision ID: 0005_reindex_recovery
Revises: 0004_raw_file_cleanup
Create Date: 2026-09-23
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0005_reindex_recovery"
down_revision: str | None = "0004_raw_file_cleanup"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _knowledge_bases_at_0004() -> sa.Table:
    """Describe the pre-0005 table so SQLite batch SQL also works offline."""
    return sa.Table(
        "knowledge_bases",
        sa.MetaData(),
        sa.Column("id", sa.Uuid(), nullable=False, primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("owner_id", sa.String(length=255), nullable=True),
        sa.Column("overrides", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("config_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def _knowledge_bases_at_0005(*, with_generation_default: bool = False) -> sa.Table:
    """Describe the post-0005 table for an offline SQLite downgrade."""
    table = _knowledge_bases_at_0004()
    table.append_column(
        sa.Column(
            "reindex_generation",
            sa.Integer(),
            nullable=False,
            server_default="0" if with_generation_default else None,
        )
    )
    table.append_column(sa.Column("reindex_started_at", sa.DateTime(timezone=True), nullable=True))
    return table


def upgrade() -> None:
    with op.batch_alter_table(
        "knowledge_bases", copy_from=_knowledge_bases_at_0004()
    ) as batch_op:
        batch_op.add_column(
            sa.Column("reindex_generation", sa.Integer(), nullable=False, server_default="0")
        )
        batch_op.add_column(sa.Column("reindex_started_at", sa.DateTime(timezone=True)))

    with op.batch_alter_table(
        "knowledge_bases", copy_from=_knowledge_bases_at_0005(with_generation_default=True)
    ) as batch_op:
        batch_op.alter_column("reindex_generation", server_default=None)


def downgrade() -> None:
    with op.batch_alter_table(
        "knowledge_bases", copy_from=_knowledge_bases_at_0005()
    ) as batch_op:
        batch_op.drop_column("reindex_started_at")
        batch_op.drop_column("reindex_generation")