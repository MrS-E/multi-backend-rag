"""Knowledge-base CRUD endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, status

from app.api.deps import KBServiceDep, PrincipalDep, require_owner, require_viewer
from app.api.schemas.kbs import KBCreate, KBOut, KBUpdate
from app.domain.models import SYSTEM_SUBJECT

router = APIRouter(prefix="/kbs", tags=["kbs"])


def _owner_id(principal_subject: str) -> str | None:
    """Owner is the caller's subject; ``None`` for the single-tenant system user."""
    return None if principal_subject == SYSTEM_SUBJECT else principal_subject


@router.post("", response_model=KBOut, status_code=status.HTTP_201_CREATED)
def create_kb(
    payload: KBCreate,
    service: KBServiceDep,
    principal: PrincipalDep,
) -> KBOut:
    """Create a knowledge base and provision its Qdrant collection."""
    overrides = payload.overrides.model_dump() if payload.overrides is not None else None
    kb = service.create_kb(
        name=payload.name,
        description=payload.description,
        overrides=overrides,
        owner_id=_owner_id(principal.subject),
    )
    return KBOut.model_validate(kb)


@router.get("", response_model=list[KBOut])
def list_kbs(service: KBServiceDep, principal: PrincipalDep) -> list[KBOut]:
    """List all knowledge bases."""
    return [KBOut.model_validate(kb) for kb in service.list_kbs()]


@router.get("/{kb_id}", response_model=KBOut, dependencies=[Depends(require_viewer)])
def get_kb(kb_id: uuid.UUID, service: KBServiceDep) -> KBOut:
    """Fetch a single knowledge base."""
    return KBOut.model_validate(service.get_kb(kb_id))


@router.patch("/{kb_id}", response_model=KBOut, dependencies=[Depends(require_owner)])
def update_kb(kb_id: uuid.UUID, payload: KBUpdate, service: KBServiceDep) -> KBOut:
    """Partially update a knowledge base (including model overrides)."""
    overrides = (
        payload.overrides.model_dump(exclude_unset=True) if payload.overrides is not None else None
    )
    kb = service.update_kb(
        kb_id=kb_id,
        name=payload.name,
        description=payload.description,
        overrides_update=overrides,
    )
    return KBOut.model_validate(kb)


@router.delete(
    "/{kb_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_owner)],
)
def delete_kb(kb_id: uuid.UUID, service: KBServiceDep) -> None:
    """Delete a KB, its Qdrant collection, stored files and DB rows."""
    service.delete_kb(kb_id)
