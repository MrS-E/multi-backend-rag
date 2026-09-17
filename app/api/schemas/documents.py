"""Document and job API schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    """A document as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kb_id: uuid.UUID
    uri: str
    content_hash: str
    mime: str
    status: str
    created_at: datetime
    updated_at: datetime


class JobAccepted(BaseModel):
    """Response for an endpoint that enqueued asynchronous work."""

    job_id: str


class JobStatusOut(BaseModel):
    """Ephemeral job status served from arq's Redis metadata."""

    status: str
    error: str | None = None
