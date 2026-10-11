"""Source registration, event push, and manual synchronization endpoints."""

from __future__ import annotations

import base64
import binascii
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import (
    KBServiceDep,
    SettingsDep,
    SourceServiceDep,
    require_editor,
    require_viewer,
)
from app.api.schemas.documents import JobAccepted
from app.api.schemas.sources import (
    EventPushRequest,
    SourceCreate,
    SourceEventIn,
    SourceOut,
    SourceUpdate,
)
from app.domain.events import DeleteEvent, SourceEvent, UpsertEvent
from app.interaction.registry import get_source_registry

router = APIRouter(prefix="/kbs/{kb_id}/sources", tags=["sources"])


def _validate_type(type_name: str, mode: str) -> None:
    """Ensure pull sources name a registered connector."""
    if mode == "pull":
        try:
            get_source_registry().get(type_name)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "", response_model=SourceOut, status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_editor)],
)
async def create_source(
    kb_id: uuid.UUID,
    payload: SourceCreate,
    kb_service: KBServiceDep,
    service: SourceServiceDep,
) -> SourceOut:
    """Register a source on an existing KB."""
    kb_service.get_kb(kb_id)
    _validate_type(payload.type, payload.mode)
    return SourceOut.model_validate(
        await service.create_source(
            kb_id, payload.type, payload.mode, payload.config,
            payload.sync_interval_s, payload.enabled,
        )
    )


@router.get("", response_model=list[SourceOut], dependencies=[Depends(require_viewer)])
def list_sources(
    kb_id: uuid.UUID, kb_service: KBServiceDep, service: SourceServiceDep
) -> list[SourceOut]:
    """List sources registered on a KB."""
    kb_service.get_kb(kb_id)
    return [SourceOut.model_validate(source) for source in service.list_sources(kb_id)]


@router.get("/{source_id}", response_model=SourceOut, dependencies=[Depends(require_viewer)])
def get_source(
    kb_id: uuid.UUID, source_id: uuid.UUID, service: SourceServiceDep
) -> SourceOut:
    """Get one source."""
    return SourceOut.model_validate(service.get_source(kb_id, source_id))


@router.patch("/{source_id}", response_model=SourceOut, dependencies=[Depends(require_editor)])
async def update_source(
    kb_id: uuid.UUID,
    source_id: uuid.UUID,
    payload: SourceUpdate,
    service: SourceServiceDep,
) -> SourceOut:
    """Update source configuration."""
    updates = payload.model_dump(exclude_unset=True)
    if "type" in updates or "mode" in updates:
        current = service.get_source(kb_id, source_id)
        _validate_type(
            updates.get("type", current.type),
            updates.get("mode", current.mode.value),
        )
    return SourceOut.model_validate(await service.update_source(kb_id, source_id, **updates))


@router.delete(
    "/{source_id}", status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_editor)],
)
async def delete_source(
    kb_id: uuid.UUID, source_id: uuid.UUID, service: SourceServiceDep
) -> None:
    """Delete a source registration."""
    await service.delete_source(kb_id, source_id)


@router.post(
    "/{source_id}/sync", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_editor)],
)
async def sync_source(
    kb_id: uuid.UUID,
    source_id: uuid.UUID,
    service: SourceServiceDep,
) -> JobAccepted:
    """Enqueue a manual pull synchronization."""
    try:
        job_id = await service.enqueue_sync(kb_id, source_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return JobAccepted(job_id=job_id)


events_router = APIRouter(prefix="/kbs/{kb_id}", tags=["sources"])


_MEGABYTE = 1024 * 1024


def _event_from_input(
    event: SourceEventIn, source_id: uuid.UUID | None, max_upload_bytes: int
) -> SourceEvent:
    metadata = dict(event.metadata)
    if source_id is not None:
        metadata["source_id"] = str(source_id)
    max_encoded_length = 4 * ((max_upload_bytes + 2) // 3)
    if event.data_b64 is not None and len(event.data_b64) > max_encoded_length:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"inline event data exceeds the {max_upload_bytes // _MEGABYTE} MB limit",
        )
    if event.op == "delete":
        return DeleteEvent(doc_id=str(event.doc_id), uri=event.uri)
    if event.path is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="path is not accepted by the public event API; send data_b64 instead",
        )
    if event.data_b64 is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="upsert events require data_b64",
        )
    try:
        data = base64.b64decode(event.data_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="data_b64 must be valid base64",
        ) from exc
    if len(data) > max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"inline event data exceeds the {max_upload_bytes // _MEGABYTE} MB limit",
        )
    return UpsertEvent(
        doc_id=str(event.doc_id), uri=event.uri, mime=event.mime, metadata=metadata,
        content_hash=event.content_hash, data=data, path=event.path,
    )


@events_router.post(
    "/events", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_editor)],
)
async def push_events(
    kb_id: uuid.UUID,
    payload: EventPushRequest | SourceEventIn,
    kb_service: KBServiceDep,
    service: SourceServiceDep,
    settings: SettingsDep,
) -> JobAccepted:
    """Accept event-mode changes from an interaction-layer caller."""
    kb_service.get_kb(kb_id)
    if isinstance(payload, SourceEventIn):
        source_id = None
        inputs = [payload]
    else:
        source_id = payload.source_id
        assert source_id is not None
        inputs = payload.events
        source = service.get_source(kb_id, source_id)
        if source.mode.value != "event":
            raise HTTPException(status_code=422, detail="source does not use event mode")
    if len(inputs) > settings.server.max_event_batch:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"event batch exceeds the {settings.server.max_event_batch} event limit",
        )
    service.validate_document_ids(kb_id, [str(event.doc_id) for event in inputs])
    max_upload_bytes = settings.server.max_upload_mb * _MEGABYTE
    events = [_event_from_input(event, source_id, max_upload_bytes) for event in inputs]
    try:
        job_id = await service.enqueue_events(kb_id, source_id, events)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return JobAccepted(job_id=job_id)