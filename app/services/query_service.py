"""Query application service: retrieve → assemble → generate for one KB.

Sits between the query router and the RAG core, exactly as
:class:`~app.services.document_service.DocumentService` sits between the
documents router and the broker. The API edge stays free of RAG imports; this
service owns the synchronous read path the plan calls out in Key Decision 4.

Two guard rails live here:

* a KB flagged ``reindex_required`` cannot be queried (its vectors are about to
  be rebuilt) → :class:`ConflictError` → ``409``;
* a KB with no indexed content returns a graceful "no documents indexed yet"
  answer without embedding or calling the chat provider.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.settings import Settings
from app.domain.config_resolution import resolve_effective_config
from app.domain.errors import ConflictError, NotFoundError
from app.domain.models import KB_STATUS_REINDEX_REQUIRED
from app.persistence.models import KnowledgeBase
from app.persistence.qdrant_store import QdrantStore
from app.providers.registry import ProviderRegistry
from app.rag.generate import Source, generate_answer
from app.rag.retrieve import retrieve_chunks

#: Answer returned when a KB has no indexed content; no provider call is made.
NO_DOCUMENTS_ANSWER = (
    "No documents have been indexed for this knowledge base yet, so there is "
    "nothing to answer from. Upload or sync documents and try again."
)


@dataclass(frozen=True, slots=True)
class QueryResult:
    """The answer plus its cited sources, ready for the API schema."""

    answer: str
    sources: list[Source]


class QueryService:
    """Answer questions against one knowledge base's index."""

    def __init__(
        self,
        session: Session,
        settings: Settings,
        registry: ProviderRegistry,
        qdrant: QdrantStore,
    ) -> None:
        self._session = session
        self._settings = settings
        self._registry = registry
        self._qdrant = qdrant

    def query(
        self,
        kb_id: uuid.UUID,
        question: str,
        top_k: int | None = None,
    ) -> QueryResult:
        """Answer ``question`` from ``kb_id``'s index.

        Raises :class:`NotFoundError` for an unknown KB and
        :class:`ConflictError` when the KB needs reindexing.
        """
        kb = self._session.get(KnowledgeBase, kb_id)
        if kb is None:
            raise NotFoundError(f"knowledge base {kb_id} not found")
        if kb.status == KB_STATUS_REINDEX_REQUIRED:
            raise ConflictError(
                f"knowledge base {kb_id} requires reindexing before it can be queried"
            )

        config = resolve_effective_config(self._settings.defaults, kb.overrides)
        if not self._has_content(kb_id):
            return QueryResult(answer=NO_DOCUMENTS_ANSWER, sources=[])

        chunks = retrieve_chunks(
            kb_id, question, config, self._registry, self._qdrant, top_k=top_k
        )
        if not chunks:
            return QueryResult(answer=NO_DOCUMENTS_ANSWER, sources=[])

        generated = generate_answer(question, chunks, config, self._registry)
        return QueryResult(answer=generated.answer, sources=generated.sources)

    def _has_content(self, kb_id: uuid.UUID) -> bool:
        """Return whether the KB's collection exists and holds at least one point."""
        if not self._qdrant.collection_exists(kb_id):
            return False
        return self._qdrant.count(kb_id) > 0
