"""Query API schemas: request body and cited-answer response."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for ``POST /kbs/{id}/query``."""

    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, gt=0, le=100)


class SourceOut(BaseModel):
    """One cited source backing the answer."""

    uri: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    """A cited answer: the generated text plus the sources it was given."""

    answer: str
    sources: list[SourceOut]
