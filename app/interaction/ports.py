"""Contracts implemented by interaction-layer source connectors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from app.domain.events import SourceEvent


@dataclass(frozen=True, slots=True)
class PullResult:
    """Changes discovered by a pull connector and its next cursor."""

    events: list[SourceEvent]
    cursor: dict[str, Any]


@runtime_checkable
class SourceConnector(Protocol):
    """Port for a source that can discover document changes."""

    type_name: str

    def pull(
        self,
        source_id: UUID,
        config: Mapping[str, Any],
        cursor: Mapping[str, Any] | None,
    ) -> PullResult:
        """Return changes since ``cursor`` and the cursor to persist on success."""
        ...