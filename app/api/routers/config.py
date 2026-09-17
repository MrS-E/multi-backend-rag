"""Effective-config endpoint: ``override ?? global`` resolution."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import KBServiceDep, PrincipalDep
from app.api.schemas.kbs import EffectiveConfigOut

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/effective", response_model=EffectiveConfigOut)
def get_effective_config(
    service: KBServiceDep,
    principal: PrincipalDep,
    kb_id: Annotated[
        uuid.UUID | None, Query(description="KB to resolve; omit for globals")
    ] = None,
) -> EffectiveConfigOut:
    """Return the resolved config for a KB, or the global defaults when omitted."""
    kb = service.get_kb(kb_id) if kb_id is not None else None
    return EffectiveConfigOut.from_effective(service.effective_config(kb))
