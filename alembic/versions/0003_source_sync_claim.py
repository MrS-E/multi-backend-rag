"""persist pull-source scheduler claims

Revision ID: 0003_source_sync_claim
Revises: 0002_config_version
Create Date: 2026-09-23

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003_source_sync_claim"
down_revision: str | None = "0002_config_version"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("sources", sa.Column("sync_claim_token", sa.String(length=64), nullable=True))
    op.add_column("sources", sa.Column("sync_claimed_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_sources_sync_claim_token", "sources", ["sync_claim_token"])


def downgrade() -> None:
    op.drop_index("ix_sources_sync_claim_token", table_name="sources")
    op.drop_column("sources", "sync_claimed_at")
    op.drop_column("sources", "sync_claim_token")