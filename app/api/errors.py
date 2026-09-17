"""Translate domain errors into HTTP responses.

Keeping this mapping in one place means routers can raise plain domain errors
(``NotFoundError``, ``ConflictError``, ``ProviderError``) and still return the
right status codes.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.domain.errors import ConflictError, DomainError, NotFoundError, ProviderError


def _error_response(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


async def _handle_not_found(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(status.HTTP_404_NOT_FOUND, str(exc))


async def _handle_conflict(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(status.HTTP_409_CONFLICT, str(exc))


async def _handle_provider(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(status.HTTP_502_BAD_GATEWAY, str(exc))


async def _handle_domain(request: Request, exc: Exception) -> JSONResponse:
    return _error_response(status.HTTP_400_BAD_REQUEST, str(exc))


def register_error_handlers(app: FastAPI) -> None:
    """Attach domain-error handlers to the application."""
    app.add_exception_handler(NotFoundError, _handle_not_found)
    app.add_exception_handler(ConflictError, _handle_conflict)
    app.add_exception_handler(ProviderError, _handle_provider)
    app.add_exception_handler(DomainError, _handle_domain)
