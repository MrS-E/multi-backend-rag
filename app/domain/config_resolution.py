"""Effective-config resolution: ``effective = override ?? global`` per field.

Pure and framework-free so it can be unit-tested exhaustively and reused by the
API, workers and RAG core alike. A KB's ``overrides`` is a sparse mapping: any
missing or ``None`` field inherits the global default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.settings import ModelDefaults

#: Overridable field names, in the order they appear in the effective config.
OVERRIDABLE_FIELDS: tuple[str, ...] = (
    "embed_model",
    "chat_model",
    "vision_model",
    "chunk_size",
    "chunk_overlap",
    "top_k",
)


@dataclass(frozen=True, slots=True)
class EffectiveConfig:
    """Fully-resolved model + retrieval configuration for one KB (or globally)."""

    embed_model: str
    chat_model: str
    vision_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_ctx_chars: int


def resolve_effective_config(
    defaults: ModelDefaults, overrides: dict[str, Any] | None
) -> EffectiveConfig:
    """Merge a KB's sparse overrides onto the global defaults.

    ``overrides`` may be ``None``, empty, partial, or contain explicit ``None``
    values; in every such case the corresponding global default is used. Unknown
    keys are ignored. ``max_ctx_chars`` is global-only and never overridden.
    """
    overrides = overrides or {}
    resolved: dict[str, Any] = {}
    for field in OVERRIDABLE_FIELDS:
        override = overrides.get(field)
        resolved[field] = override if override is not None else getattr(defaults, field)
    return EffectiveConfig(max_ctx_chars=defaults.max_ctx_chars, **resolved)
