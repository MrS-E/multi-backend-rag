"""Knowledge-base API schemas.

``overrides`` is a sparse, nullable partial model: any field left out (or set to
``null``) inherits the global default, matching the ``override ?? global`` rule
implemented in :mod:`app.domain.config_resolution`.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.config_resolution import EffectiveConfig


class KBOverrides(BaseModel):
    """Per-KB model/retrieval overrides; every field is optional."""

    embed_model: str | None = Field(default=None, max_length=255)
    chat_model: str | None = Field(default=None, max_length=255)
    vision_model: str | None = Field(default=None, max_length=255)
    chunk_size: int | None = Field(default=None, gt=0, le=100_000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=100_000)
    top_k: int | None = Field(default=None, gt=0, le=100)


class KBCreate(BaseModel):
    """Request body for ``POST /kbs``."""

    name: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=10_000)
    overrides: KBOverrides | None = None


class KBUpdate(BaseModel):
    """Request body for ``PATCH /kbs/{id}``; omitted fields are left unchanged.

    ``overrides`` is merged field-by-field onto the KB's existing overrides, so a
    client can change one model without resending the rest. An explicit ``null``
    on a field clears that override (falling back to the global default).
    """

    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=10_000)
    overrides: KBOverrides | None = None


class KBOut(BaseModel):
    """A knowledge base as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str
    owner_id: str | None
    overrides: dict[str, Any]
    status: str
    created_at: datetime


class EffectiveConfigOut(BaseModel):
    """Resolved configuration for a KB or the global defaults."""

    embed_model: str
    chat_model: str
    vision_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_ctx_chars: int

    @classmethod
    def from_effective(cls, config: EffectiveConfig) -> EffectiveConfigOut:
        """Build the response model from a resolved domain config."""
        return cls(**asdict(config))
