"""persist raw-file cleanup retry markers

Revision ID: 0004_raw_file_cleanup
Revises: 0003_source_sync_claim
Create Date: 2026-09-23
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004_raw_file_cleanup"
down_revision: str | None = "0003_source_sync_claim"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "raw_file_cleanups",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("kb_id", sa.Uuid(), nullable=False),
        sa.Column("raw_path", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("raw_path"),
    )
    op.create_index("ix_raw_file_cleanups_kb_id", "raw_file_cleanups", ["kb_id"])


def downgrade() -> None:
    op.drop_index("ix_raw_file_cleanups_kb_id", table_name="raw_file_cleanups")
    op.drop_table("raw_file_cleanups")