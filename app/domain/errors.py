"""Domain error types.

Routers translate these into HTTP responses; the domain and providers never
import FastAPI. ``NotFoundError``/``ConflictError`` map to ``404``/``409`` and
``ProviderError`` to ``502`` (see ``app/api/errors.py`` once added).
"""

from __future__ import annotations


class DomainError(Exception):
    """Base class for all domain-level errors."""


class NotFoundError(DomainError):
    """A requested entity (KB, document, source, job) does not exist."""


class ConflictError(DomainError):
    """The request conflicts with current state (e.g. duplicate name)."""


class ProviderError(DomainError):
    """A model provider call failed (unreachable, bad model, upstream error)."""


class ConfigurationError(DomainError):
    """Configuration is invalid or incomplete (e.g. missing API key env var)."""
