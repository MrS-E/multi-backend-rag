"""Document upload, list and delete endpoints (event mode via the API)."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

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


def _max_upload_bytes(settings: Settings) -> int:
    """Return the configured upload limit in bytes."""
    return settings.server.max_upload_mb * _MEGABYTE


@router.post(
    "",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_editor)],
)
async def upload_document(
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
    data = await file.read()
    try:
        _, job_id = await service.upload(
            kb_id, uri, data, max_upload_bytes=_max_upload_bytes(settings)
        )
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc
    except DocumentTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc
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
