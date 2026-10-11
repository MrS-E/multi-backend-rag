"""Helpers for durable raw-file cleanup markers."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.persistence.models import RawFileCleanup


def ensure_raw_file_cleanup_marker(session: Session, kb_id: uuid.UUID, raw_path: str) -> None:
    """Add a cleanup marker unless the unique path is already marked."""
    existing = session.scalar(
        select(RawFileCleanup.id).where(RawFileCleanup.raw_path == raw_path)
    )
    if existing is None:
        session.add(RawFileCleanup(kb_id=kb_id, raw_path=raw_path))


def remove_raw_file_cleanup_marker(session: Session, raw_path: str) -> None:
    """Remove a marker after its raw file has been handled successfully."""
    session.query(RawFileCleanup).filter(RawFileCleanup.raw_path == raw_path).delete(
        synchronize_session=False
    )