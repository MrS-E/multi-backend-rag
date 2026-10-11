"""Content-addressed raw file storage under ``{storage_root}/{kb_id}/``.

Files are named by the sha256 of their bytes, so identical content is stored
once and re-uploads are idempotent. Paths returned by :meth:`save` are relative
to the storage root so they can be persisted in the ``documents`` table.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from app.core.settings import get_settings


class FileStorage:
    """Store raw uploads keyed by content hash."""

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            root = get_settings().storage.root
        self._root = Path(root)

    def _resolve(self, rel_path: str) -> Path:
        """Resolve a stored relative path, refusing to escape the storage root."""
        candidate = (self._root / rel_path).resolve()
        root = self._root.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"path escapes storage root: {rel_path}")
        return candidate

    def save(self, kb_id: uuid.UUID | str, data: bytes) -> tuple[str, str]:
        """Persist ``data`` and return ``(rel_path, sha256)``."""
        digest = hashlib.sha256(data).hexdigest()
        rel_path = f"{kb_id}/{digest}"
        target = self._resolve(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return rel_path, digest

    def read(self, rel_path: str, max_bytes: int | None = None) -> bytes:
        """Read raw bytes for a stored relative path, optionally with a hard bound."""
        path = self._resolve(rel_path)
        if max_bytes is None:
            return path.read_bytes()
        if path.stat().st_size > max_bytes:
            raise ValueError("stored event data exceeds the configured upload limit")
        with path.open("rb") as file:
            data = file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("stored event data exceeds the configured upload limit")
        return data

    def delete(self, rel_path: str) -> None:
        """Delete a stored file if present."""
        self._resolve(rel_path).unlink(missing_ok=True)
