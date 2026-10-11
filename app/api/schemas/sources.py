"""Source registration and event-push schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SourceCreate(BaseModel):
    """Request body for source registration."""

    type: str = Field(min_length=1, max_length=64)
    mode: Literal["event", "pull"]
    config: dict[str, Any] = Field(default_factory=dict)
    sync_interval_s: int = Field(default=60, ge=0)
    enabled: bool = True


class SourceUpdate(BaseModel):
    """Partial source update."""

    type: str | None = Field(default=None, min_length=1, max_length=64)
    mode: Literal["event", "pull"] | None = None
    config: dict[str, Any] | None = None
    sync_interval_s: int | None = Field(default=None, ge=0)
    enabled: bool | None = None


class SourceOut(BaseModel):
    """Persisted source representation."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kb_id: uuid.UUID
    type: str
    mode: str
    config: dict[str, Any]
    sync_interval_s: int
    cursor: dict[str, Any] | None
    last_sync_at: datetime | None
    enabled: bool


class SourceEventIn(BaseModel):
    """JSON form of an upsert or delete event accepted by the event endpoint."""

    op: Literal["upsert", "delete"]
    doc_id: uuid.UUID
    uri: str = Field(default="", max_length=2048)
    mime: str = Field(default="application/octet-stream", max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: str | None = Field(default=None, min_length=64, max_length=64,
                                     pattern=r"^[0-9a-fA-F]{64}$")
    data_b64: str | None = None
    path: str | None = Field(default=None, max_length=4096)


class EventPushRequest(BaseModel):
    """Batch event request; ``source_id`` attributes events to an event source."""

    source_id: uuid.UUID | None = None
    events: list[SourceEventIn] = Field(min_length=1, max_length=10_000)