"""Reference pull connector for a directory of supported documents."""

from __future__ import annotations

import hashlib
import mimetypes
import uuid
from collections.abc import Iterable, Mapping
from itertools import chain
from pathlib import Path
from typing import Any

from app.domain.events import DeleteEvent, UpsertEvent
from app.domain.filetypes import is_allowed_uri, mime_for_uri
from app.interaction.ports import PullResult


class LocalFolderConnector:
    """Walk a folder and use SHA-256 plus stable UUIDs for change detection."""

    type_name = "local_folder"
    _DOC_NAMESPACE = uuid.UUID("c4d7a188-2918-4e99-8bf0-0b9a6d6e2f12")
    _CONTINUATION_KEY = "_local_folder_continuation"

    def pull(
        self,
        source_id: uuid.UUID,
        config: Mapping[str, Any],
        cursor: Mapping[str, Any] | None,
    ) -> PullResult:
        """Emit only added/changed files and deletes since the previous cursor."""
        root = self._root(config)
        max_events = config.get("_max_events")
        if not isinstance(max_events, int) or max_events <= 0:
            max_events = None

        continuation = (cursor or {}).get(self._CONTINUATION_KEY)
        if continuation is not None:
            if not isinstance(continuation, dict):
                raise ValueError("invalid local_folder continuation cursor")
            previous = dict(continuation.get("previous") or {})
            current = dict(continuation.get("current") or {})
            offset = int(continuation.get("offset", 0))
        else:
            previous = dict((cursor or {}).get("files") or {})
            current = self._snapshot(source_id, root, config.get("_max_file_bytes"))
            offset = 0

        selected: list[tuple[str, str]] = []
        total = 0
        pending: Iterable[tuple[str, str]] = (
            ("upsert", uri)
            for uri in sorted(current)
            if previous.get(uri, {}).get("hash") != current[uri].get("hash")
        )
        pending = chain(
            pending,
            (("delete", uri) for uri in sorted(previous) if uri not in current),
        )
        for item in pending:
            if total >= offset and (max_events is None or len(selected) < max_events):
                selected.append(item)
            total += 1
        end = min(offset + len(selected), total)
        events: list[UpsertEvent | DeleteEvent] = []
        for kind, uri in selected:
            if kind == "delete":
                events.append(DeleteEvent(doc_id=str(previous[uri]["doc_id"]), uri=uri))
                continue
            details = current[uri]
            path = root / uri
            data = self._read_bounded(path, config.get("_max_file_bytes"))
            events.append(
                UpsertEvent(
                    doc_id=str(details["doc_id"]),
                    uri=uri,
                    mime=(
                        mime_for_uri(uri)
                        or mimetypes.guess_type(uri)[0]
                        or "application/octet-stream"
                    ),
                    content_hash=str(details["hash"]),
                    data=data,
                    metadata={"source_id": str(source_id)},
                )
            )

        if end < total:
            next_cursor = {
                self._CONTINUATION_KEY: {
                    "previous": previous,
                    "current": current,
                    "offset": end,
                }
            }
        else:
            next_cursor = {"files": current}
        return PullResult(events=events, cursor=next_cursor)

    def _snapshot(
        self, source_id: uuid.UUID, root: Path, max_file_bytes: Any
    ) -> dict[str, dict[str, str]]:
        """Hash the folder without retaining any document bytes in the cursor."""
        current: dict[str, dict[str, str]] = {}
        if not root.exists():
            return current
        if not root.is_dir():
            raise ValueError("local_folder source root must be a directory")
        files = (path for path in root.rglob("*") if not path.is_symlink() and path.is_file())
        for path in sorted(files):
            uri = path.relative_to(root).as_posix()
            if not is_allowed_uri(uri):
                continue
            data = self._read_bounded(path, max_file_bytes)
            current[uri] = {
                "hash": hashlib.sha256(data).hexdigest(),
                "doc_id": str(uuid.uuid5(self._DOC_NAMESPACE, f"{source_id}:{uri}")),
            }
        return current

    @staticmethod
    def _read_bounded(path: Path, max_bytes: Any) -> bytes:
        """Read a file without allowing a connector event to exceed its bound."""
        if max_bytes is None:
            return path.read_bytes()
        if not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("local_folder max file size must be a positive integer")
        if path.stat().st_size > max_bytes:
            raise ValueError(f"local_folder file exceeds the {max_bytes} byte limit")
        with path.open("rb") as file:
            data = file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"local_folder file exceeds the {max_bytes} byte limit")
        return data

    @staticmethod
    def _root(config: Mapping[str, Any]) -> Path:
        raw_root = config.get("path", config.get("root"))
        if not isinstance(raw_root, str) or not raw_root:
            raise ValueError("local_folder source config requires a non-empty 'path'")
        root = Path(raw_root).expanduser().resolve()
        allowed_raw = config.get("_allowed_root")
        if allowed_raw is not None:
            if not isinstance(allowed_raw, str) or not allowed_raw:
                raise ValueError("local_folder allowed source root must be non-empty")
            allowed = Path(allowed_raw).expanduser().resolve()
            if root != allowed and allowed not in root.parents:
                raise ValueError("local_folder source path is outside the configured sources root")
        return root