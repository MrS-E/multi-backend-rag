"""Shared FastAPI dependencies: settings, DB session, providers, storage, auth.

Everything the routers need is injected here. Resources that tests need to
substitute (DB session factory, provider registry, Qdrant store, file storage)
are read from ``app.state`` first and fall back to process-wide singletons, so a
test can build an app with in-memory SQLite and mocked backends.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from typing import Annotated

from arq.connections import ArqRedis
from fastapi import Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session, sessionmaker

from app.core.broker import JobQueue
from app.core.security import Principal, get_principal, require_kb_access
from app.core.settings import Settings
from app.persistence.db import SessionLocal
from app.persistence.file_storage import FileStorage
from app.persistence.qdrant_store import QdrantStore
from app.providers.registry import ProviderRegistry, get_registry
from app.services.document_service import DocumentService
from app.services.kb_service import KBService


def get_settings_dep(request: Request) -> Settings:
    """Return the settings instance bound to the running app."""
    settings: Settings = request.app.state.settings
    return settings


def get_session(request: Request) -> Iterator[Session]:
    """Yield a DB session for the request, closing it afterwards.

    Routers commit explicitly; the session is never auto-committed here so that
    a failed request leaves no partial writes.
    """
    factory: sessionmaker[Session] = (
        getattr(request.app.state, "session_factory", None) or SessionLocal
    )
    session = factory()
    try:
        yield session
    finally:
        session.close()


def get_provider_registry(request: Request) -> ProviderRegistry:
    """Return the app-scoped provider registry (or the process singleton)."""
    registry: ProviderRegistry | None = getattr(request.app.state, "provider_registry", None)
    return registry if registry is not None else get_registry()


def get_qdrant_store(request: Request) -> QdrantStore:
    """Return the app-scoped Qdrant store, constructing one on first use."""
    store: QdrantStore | None = getattr(request.app.state, "qdrant_store", None)
    if store is None:
        store = QdrantStore()
        request.app.state.qdrant_store = store
    return store


def get_file_storage(request: Request) -> FileStorage:
    """Return the app-scoped file storage, constructing one on first use."""
    storage: FileStorage | None = getattr(request.app.state, "file_storage", None)
    if storage is None:
        storage = FileStorage()
        request.app.state.file_storage = storage
    return storage


def get_kb_service(
    session: SessionDep,
    settings: SettingsDep,
    registry: RegistryDep,
    qdrant: QdrantDep,
    storage: FileStorageDep,
) -> KBService:
    """Build the KB application service for the request."""
    return KBService(session, settings, registry, qdrant, storage)


def get_arq_pool(request: Request) -> ArqRedis:
    """Return the app-scoped arq pool created at startup.

    Raises ``503`` when the pool is unavailable (Redis down at startup), so job
    enqueuing and status lookups fail with a clear service error.
    """
    pool: ArqRedis | None = getattr(request.app.state, "arq_pool", None)
    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="job broker unavailable",
        )
    return pool


def get_job_queue(request: Request) -> JobQueue:
    """Return the app-scoped job queue, wrapping the arq pool on first use."""
    queue: JobQueue | None = getattr(request.app.state, "job_queue", None)
    if queue is None:
        queue = JobQueue(get_arq_pool(request))
        request.app.state.job_queue = queue
    return queue


def get_document_service(
    session: SessionDep,
    storage: FileStorageDep,
    queue: JobQueueDep,
) -> DocumentService:
    """Build the document application service for the request."""
    return DocumentService(session, storage, queue)


def kb_access_dependency(role: str) -> Callable[..., None]:
    """Build a dependency enforcing ``role`` on the ``kb_id`` path parameter.

    v1 is allow-all (see :func:`app.core.security.require_kb_access`); using a
    factory keeps the role explicit at every KB-scoped endpoint so the future
    membership check slots in without touching routers.
    """

    def dependency(
        kb_id: uuid.UUID,
        principal: Annotated[Principal, Depends(get_principal)],
    ) -> None:
        require_kb_access(kb_id, role, principal)

    return dependency


def kb_query_access_dependency(role: str) -> Callable[..., None]:
    """Build a dependency enforcing ``role`` on the ``kb_id`` query parameter.

    Like :func:`kb_access_dependency`, but for endpoints whose KB reference is a
    query parameter rather than a path parameter (e.g. ``GET /config/effective``).
    When ``kb_id`` is omitted the endpoint is not KB-scoped, so no check runs.
    """

    def dependency(
        principal: Annotated[Principal, Depends(get_principal)],
        kb_id: Annotated[uuid.UUID | None, Query()] = None,
    ) -> None:
        if kb_id is not None:
            require_kb_access(kb_id, role, principal)

    return dependency


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
SessionDep = Annotated[Session, Depends(get_session)]
PrincipalDep = Annotated[Principal, Depends(get_principal)]
RegistryDep = Annotated[ProviderRegistry, Depends(get_provider_registry)]
QdrantDep = Annotated[QdrantStore, Depends(get_qdrant_store)]
FileStorageDep = Annotated[FileStorage, Depends(get_file_storage)]
ArqPoolDep = Annotated[ArqRedis, Depends(get_arq_pool)]
JobQueueDep = Annotated[JobQueue, Depends(get_job_queue)]
KBServiceDep = Annotated[KBService, Depends(get_kb_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]

#: Permission to read a KB (list/get/config).
require_viewer = kb_access_dependency("viewer")
#: Permission to read a KB referenced by query parameter (``GET /config/effective``).
require_query_viewer = kb_query_access_dependency("viewer")
#: Permission to modify a KB's content (documents, sources, events).
require_editor = kb_access_dependency("editor")
#: Permission to modify KB configuration or delete it.
require_owner = kb_access_dependency("owner")
