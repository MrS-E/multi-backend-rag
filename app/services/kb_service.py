"""Knowledge-base application service: CRUD, collection lifecycle, config merge.

Sits between the KB router and persistence/providers. It owns the two rules the
plan calls out explicitly:

* **effective config** = ``override ?? global`` (delegated to
  :func:`app.domain.config_resolution.resolve_effective_config`);
* **embedding-model change** marks the KB ``reindex_required`` (the reindex job
  itself lands in Step 5).

Collection creation is *best-effort*: the vector dimension is obtained by
embedding a probe string with the KB's effective embed model. If the provider or
Qdrant is unreachable at create time the KB row is still created, with status
``collection_pending`` so the caller can see that the collection has not been
provisioned yet; Step 3's ingestion pipeline will ensure it before indexing.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.settings import Settings
from app.domain.config_resolution import EffectiveConfig, resolve_effective_config
from app.domain.errors import NotFoundError
from app.domain.models import (
    KB_STATUS_ACTIVE,
    KB_STATUS_COLLECTION_PENDING,
    KB_STATUS_REINDEX_REQUIRED,
)
from app.persistence.file_storage import FileStorage
from app.persistence.models import KnowledgeBase
from app.persistence.qdrant_store import QdrantStore
from app.providers.registry import ProviderRegistry, parse_model_str

logger = logging.getLogger(__name__)

#: Short string embedded to discover the model's vector dimension.
_DIMENSION_PROBE = "dimension probe"


class KBService:
    """Create, read, update and delete knowledge bases."""

    def __init__(
        self,
        session: Session,
        settings: Settings,
        registry: ProviderRegistry,
        qdrant: QdrantStore,
        storage: FileStorage,
    ) -> None:
        self._session = session
        self._settings = settings
        self._registry = registry
        self._qdrant = qdrant
        self._storage = storage

    def list_kbs(self) -> list[KnowledgeBase]:
        """Return all knowledge bases, oldest first."""
        return list(self._session.scalars(select(KnowledgeBase).order_by(KnowledgeBase.created_at)))

    def get_kb(self, kb_id: uuid.UUID) -> KnowledgeBase:
        """Return a KB or raise :class:`NotFoundError`."""
        kb = self._session.get(KnowledgeBase, kb_id)
        if kb is None:
            raise NotFoundError(f"knowledge base {kb_id} not found")
        return kb

    def effective_config(self, kb: KnowledgeBase | None) -> EffectiveConfig:
        """Resolve the effective config for ``kb`` (``None`` → global defaults)."""
        overrides = kb.overrides if kb is not None else None
        return resolve_effective_config(self._settings.defaults, overrides)

    def create_kb(
        self,
        name: str,
        description: str,
        overrides: dict[str, Any] | None,
        owner_id: str | None,
    ) -> KnowledgeBase:
        """Create a KB row and provision its Qdrant collection (best-effort)."""
        cleaned = {key: value for key, value in (overrides or {}).items() if value is not None}
        kb = KnowledgeBase(
            name=name,
            description=description,
            owner_id=owner_id,
            overrides=cleaned,
            status=KB_STATUS_ACTIVE,
        )
        self._session.add(kb)
        self._session.flush()

        if self._provision_collection(kb):
            kb.status = KB_STATUS_ACTIVE
        else:
            kb.status = KB_STATUS_COLLECTION_PENDING
        self._session.commit()
        self._session.refresh(kb)
        return kb

    def update_kb(
        self,
        kb_id: uuid.UUID,
        name: str | None,
        description: str | None,
        overrides_update: dict[str, Any] | None,
    ) -> KnowledgeBase:
        """Apply a partial update; flag ``reindex_required`` on embed-model change."""
        kb = self.get_kb(kb_id)
        previous_embed_model = self.effective_config(kb).embed_model

        if name is not None:
            kb.name = name
        if description is not None:
            kb.description = description
        if overrides_update is not None:
            kb.overrides = self._merge_overrides(kb.overrides, overrides_update)

        if self.effective_config(kb).embed_model != previous_embed_model:
            kb.status = KB_STATUS_REINDEX_REQUIRED

        self._session.commit()
        self._session.refresh(kb)
        return kb

    def delete_kb(self, kb_id: uuid.UUID) -> None:
        """Delete a KB: Qdrant collection, stored files, then DB rows (cascade)."""
        kb = self.get_kb(kb_id)
        raw_paths = [document.raw_path for document in kb.documents]

        try:
            self._qdrant.delete_collection(kb_id)
        except Exception as exc:  # noqa: BLE001 - Qdrant may be down; DB delete must proceed
            logger.warning(
                "failed to delete qdrant collection for kb",
                extra={"kb_id": str(kb_id), "error": str(exc)},
            )

        for raw_path in raw_paths:
            try:
                self._storage.delete(raw_path)
            except Exception as exc:  # noqa: BLE001 - best-effort file cleanup
                logger.warning(
                    "failed to delete stored file",
                    extra={"kb_id": str(kb_id), "raw_path": raw_path, "error": str(exc)},
                )

        self._session.delete(kb)
        self._session.commit()

    def _merge_overrides(
        self, existing: dict[str, Any], update: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge a PATCH's override fields; explicit ``None`` clears an override."""
        merged = dict(existing)
        for key, value in update.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        return merged

    def _provision_collection(self, kb: KnowledgeBase) -> bool:
        """Create the KB's Qdrant collection; return ``False`` if it could not be done."""
        config = self.effective_config(kb)
        try:
            _, model = parse_model_str(config.embed_model)
            embedder = self._registry.get_embedder(config.embed_model)
            vectors = embedder.embed([_DIMENSION_PROBE], model)
            dimension = len(vectors[0])
            self._qdrant.create_collection(kb.id, dimension)
            return True
        except Exception as exc:  # noqa: BLE001 - provider/Qdrant outage must not block KB creation
            logger.warning(
                "could not provision qdrant collection; kb marked collection_pending",
                extra={"kb_id": str(kb.id), "embed_model": config.embed_model, "error": str(exc)},
            )
            return False
