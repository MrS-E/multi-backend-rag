"""Source events crossing the broker as JSON-serializable job payloads.

Connectors (and the upload API) emit :class:`UpsertEvent` / :class:`DeleteEvent`
values; the worker's ``ingest_events`` job consumes them. The payload form is
plain JSON — inline bytes travel base64-encoded, and a caller that already
stored the raw file can pass its ``path`` instead.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from typing import Any

#: Discriminator values for the serialized event form.
EVENT_UPSERT = "upsert"
EVENT_DELETE = "delete"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def normalize_sha256(value: str | None) -> str | None:
    """Normalize a valid SHA-256 hex digest before it crosses a boundary."""
    if isinstance(value, str) and _SHA256_RE.fullmatch(value):
        return value.lower()
    return value


@dataclass(frozen=True, slots=True)
class UpsertEvent:
    """A document was added or changed; its content must (re)enter the index.

    Exactly one content source is expected: ``data`` (inline bytes, base64 in the
    payload) or ``path`` (a path relative to the storage root). ``content_hash``
    is optional; the pipeline computes the sha256 when it is absent.
    """

    doc_id: str
    uri: str
    mime: str
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str | None = None
    data: bytes | None = None
    path: str | None = None

    def __post_init__(self) -> None:
        """Keep valid digest values canonical for direct and serialized events."""
        object.__setattr__(self, "content_hash", normalize_sha256(self.content_hash))

    def to_payload(self) -> dict[str, Any]:
        """Serialize to a JSON-safe mapping for the arq job arguments."""
        payload: dict[str, Any] = {
            "op": EVENT_UPSERT,
            "doc_id": self.doc_id,
            "uri": self.uri,
            "mime": self.mime,
            "metadata": dict(self.metadata),
        }
        if self.content_hash is not None:
            payload["content_hash"] = self.content_hash
        if self.data is not None:
            payload["data_b64"] = base64.b64encode(self.data).decode("ascii")
        if self.path is not None:
            payload["path"] = self.path
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> UpsertEvent:
        """Rebuild an event from its serialized form."""
        encoded = payload.get("data_b64")
        return cls(
            doc_id=str(payload["doc_id"]),
            uri=str(payload["uri"]),
            mime=str(payload["mime"]),
            metadata=dict(payload.get("metadata") or {}),
            content_hash=normalize_sha256(payload.get("content_hash")),
            data=base64.b64decode(encoded, validate=True) if encoded is not None else None,
            path=payload.get("path"),
        )


@dataclass(frozen=True, slots=True)
class DeleteEvent:
    """A document was removed; its vectors, stored file and DB row must go."""

    doc_id: str
    uri: str = ""

    def to_payload(self) -> dict[str, Any]:
        """Serialize to a JSON-safe mapping for the arq job arguments."""
        return {"op": EVENT_DELETE, "doc_id": self.doc_id, "uri": self.uri}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> DeleteEvent:
        """Rebuild an event from its serialized form."""
        return cls(doc_id=str(payload["doc_id"]), uri=str(payload.get("uri", "")))


#: Union of the event kinds the ingestion pipeline understands.
SourceEvent = UpsertEvent | DeleteEvent


def event_from_payload(payload: dict[str, Any]) -> SourceEvent:
    """Decode a serialized event, dispatching on its ``op`` discriminator."""
    op = payload.get("op")
    if op == EVENT_UPSERT:
        return UpsertEvent.from_payload(payload)
    if op == EVENT_DELETE:
        return DeleteEvent.from_payload(payload)
    raise ValueError(f"unknown source event op {op!r}")
