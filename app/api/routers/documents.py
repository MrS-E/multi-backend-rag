"""Document upload, list and delete endpoints (event mode via the API)."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.api.deps import (
    DocumentServiceDep,
    KBServiceDep,
    SettingsDep,
    require_editor,
    require_viewer,
)
from app.api.schemas.documents import DocumentOut, JobAccepted
from app.core.settings import Settings
from app.services.document_service import DocumentTooLargeError, UnsupportedFileTypeError

router = APIRouter(prefix="/kbs/{kb_id}/documents", tags=["documents"])

_MEGABYTE = 1024 * 1024

#: Size of the chunks an upload body is read in, so an over-limit body aborts early.
_READ_CHUNK_BYTES = _MEGABYTE


def _max_upload_bytes(settings: Settings) -> int:
    """Return the configured upload limit in bytes."""
    return settings.server.max_upload_mb * _MEGABYTE


def _content_length_exceeds_limit(request: Request, max_upload_bytes: int) -> bool:
    """Return whether the request's declared ``Content-Length`` is over the limit."""
    raw = request.headers.get("content-length")
    if raw is None:
        return False
    try:
        return int(raw) > max_upload_bytes
    except ValueError:
        return False


def _too_large(max_upload_bytes: int) -> HTTPException:
    """Build the ``413`` raised for an over-limit upload."""
    return HTTPException(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        detail=f"upload exceeds the {max_upload_bytes // _MEGABYTE} MB limit",
    )


async def _read_upload(file: UploadFile, max_upload_bytes: int) -> bytes:
    """Read an upload in bounded chunks, aborting with ``413`` as soon as it is too big."""
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await file.read(_READ_CHUNK_BYTES)
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > max_upload_bytes:
            raise _too_large(max_upload_bytes)
        chunks.append(chunk)


@router.post(
    "",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_editor)],
)
async def upload_document(
    request: Request,
    kb_id: uuid.UUID,
    service: DocumentServiceDep,
    kb_service: KBServiceDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File()],
) -> JobAccepted:
    """Accept a multipart upload, store it and enqueue indexing.

    Returns ``202`` with the ingestion job id; the client polls
    ``GET /jobs/{id}`` for the terminal status.
    """
    kb_service.get_kb(kb_id)
    uri = file.filename or ""
    max_upload_bytes = _max_upload_bytes(settings)
    if _content_length_exceeds_limit(request, max_upload_bytes):
        raise _too_large(max_upload_bytes)
    data = await _read_upload(file, max_upload_bytes)
    try:
        _, job_id = await service.upload(kb_id, uri, data, max_upload_bytes=max_upload_bytes)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc
    except DocumentTooLargeError as exc:
        raise _too_large(max_upload_bytes) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    return JobAccepted(job_id=job_id)


@router.get("", response_model=list[DocumentOut], dependencies=[Depends(require_viewer)])
def list_documents(
    kb_id: uuid.UUID,
    service: DocumentServiceDep,
    kb_service: KBServiceDep,
) -> list[DocumentOut]:
    """List a KB's documents."""
    kb_service.get_kb(kb_id)
    return [DocumentOut.model_validate(document) for document in service.list_documents(kb_id)]


@router.delete(
    "/{doc_id}",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_editor)],
)
async def delete_document(
    kb_id: uuid.UUID,
    doc_id: uuid.UUID,
    service: DocumentServiceDep,
    kb_service: KBServiceDep,
) -> JobAccepted:
    """Enqueue removal of a document (its vectors, stored file and row)."""
    kb_service.get_kb(kb_id)
    job_id = await service.delete(kb_id, doc_id)
    return JobAccepted(job_id=job_id)
