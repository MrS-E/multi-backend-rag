"""Query endpoint: cited answers with injection-protected prompting.

``POST /kbs/{id}/query`` is a synchronous read (plan Key Decision 4): it
retrieves from the KB's Qdrant collection, assembles the CDATA-wrapped context
and calls the KB's effective chat model. It is behind ``require_viewer``, the
same authorization choke point as the other KB-scoped endpoints.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends

from app.api.deps import QueryServiceDep, require_viewer
from app.api.schemas.query import QueryRequest, QueryResponse, SourceOut

router = APIRouter(prefix="/kbs/{kb_id}/query", tags=["query"])


@router.post("", response_model=QueryResponse, dependencies=[Depends(require_viewer)])
def query_kb(
    kb_id: uuid.UUID,
    payload: QueryRequest,
    service: QueryServiceDep,
) -> QueryResponse:
    """Answer a question from a KB's index, returning the answer and its sources."""
    result = service.query(kb_id, payload.question, top_k=payload.top_k)
    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceOut(uri=source.uri, score=source.score, snippet=source.snippet)
            for source in result.sources
        ],
    )
