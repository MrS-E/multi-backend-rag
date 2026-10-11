"""Database integrity checks for SQLite runtime configuration."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.persistence.db import create_db_engine
from app.persistence.models import Base, Document


def test_given_sqlite_engine_when_inserting_orphan_document_then_foreign_key_rejects(
    tmp_path: Path,
) -> None:
    engine = create_db_engine(f"sqlite:///{tmp_path / 'integrity.db'}")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add(
            Document(
                id=uuid.uuid4(),
                kb_id=uuid.uuid4(),
                uri="orphan.md",
                content_hash="a" * 64,
                mime="text/markdown",
                raw_path="missing",
                status="pending",
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()

    engine.dispose()