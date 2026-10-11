"""Domain-level value objects shared across layers.

These types carry no persistence or framework concerns: they are the vocabulary
used by the ports in :mod:`app.domain.ports` and by the API edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

#: Subject used by the ``none`` identity provider (single-tenant v1).
SYSTEM_SUBJECT = "system"

#: Knowledge-base lifecycle statuses.
KB_STATUS_ACTIVE = "active"
KB_STATUS_REINDEX_REQUIRED = "reindex_required"
KB_STATUS_REINDEXING = "reindexing"
KB_STATUS_COLLECTION_PENDING = "collection_pending"

#: Document ingestion statuses persisted on the ``documents`` row.
DOC_STATUS_PENDING = "pending"
DOC_STATUS_INDEXED = "indexed"
DOC_STATUS_FAILED = "failed"


class Message(TypedDict):
    """A single chat message handed to a :class:`ChatProvider`."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller: an opaque subject plus optional group claims."""

    subject: str
    groups: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RequestCredentials:
    """Transport-agnostic credentials extracted from an incoming request.

    Only the ``Authorization`` header is modelled today; the ``oidc`` provider
    will read the same bearer token and validate it against the issuer's JWKS.
    """

    authorization: str | None = None
