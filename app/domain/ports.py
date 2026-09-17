"""Provider ports: the only contracts the RAG core uses to reach model backends.

Implementations live in :mod:`app.providers`; the RAG core and workers depend on
these protocols only, never on the Ollama/OpenAI SDKs directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain.models import Message, Principal, RequestCredentials


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turn text into dense vectors for indexing and retrieval."""

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Embed each text, returning one vector per input in order."""
        ...


@runtime_checkable
class ChatProvider(Protocol):
    """Generate an answer from a chat-style message list."""

    def chat(self, messages: list[Message], model: str, **opts: object) -> str:
        """Return the assistant reply for ``messages``.

        ``opts`` carries generation knobs (``temperature``, ``num_ctx``,
        ``top_p``) in a provider-agnostic way.
        """
        ...


@runtime_checkable
class VisionProvider(Protocol):
    """Caption an image so it becomes retrievable as text."""

    def caption(self, image: bytes, prompt: str, model: str) -> str:
        """Return a literal description of ``image`` guided by ``prompt``."""
        ...


@runtime_checkable
class IdentityProvider(Protocol):
    """Resolve request credentials into a :class:`Principal` (or ``None``)."""

    def authenticate(self, creds: RequestCredentials) -> Principal | None:
        """Return the caller's principal, or ``None`` when unauthenticated."""
        ...
