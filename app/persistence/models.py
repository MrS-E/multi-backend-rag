"""SQLAlchemy ORM models: knowledge bases, documents, sources.

Job status is intentionally *not* persisted: it lives in arq's ephemeral Redis
metadata (see the plan), so there is no ``jobs`` table.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class SourceMode(enum.StrEnum):
    """How a source receives changes."""

    event = "event"
    pull = "pull"


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False, default="")
    owner_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    overrides: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    documents: Mapped[list[Document]] = relationship(
        back_populates="knowledge_base", cascade="all, delete-orphan"
    )
    sources: Mapped[list[Source]] = relationship(
        back_populates="knowledge_base", cascade="all, delete-orphan"
    )


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    kb_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("sources.id", ondelete="SET NULL"), nullable=True, index=True
    )
    uri: Mapped[str] = mapped_column(String, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    mime: Mapped[str] = mapped_column(String(255), nullable=False)
    raw_path: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    knowledge_base: Mapped[KnowledgeBase] = relationship(back_populates="documents")


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    kb_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    mode: Mapped[SourceMode] = mapped_column(
        Enum(SourceMode, native_enum=False), nullable=False, default=SourceMode.event
    )
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    sync_interval_s: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cursor: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    knowledge_base: Mapped[KnowledgeBase] = relationship(back_populates="sources")
