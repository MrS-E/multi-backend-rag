"""Focused tests for source connector contracts and local-folder cursors.

The ``local_folder`` connector is the reference pull implementation: it walks a
directory, detects added/changed/deleted files by sha256 and emits
``UpsertEvent``/``DeleteEvent`` values with stable document ids. Everything here
runs against a real ``tmp_path`` — no network, Qdrant or Redis.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import pytest

from app.domain.events import DeleteEvent, UpsertEvent
from app.interaction.local_folder import LocalFolderConnector
from app.interaction.registry import SourceConnectorRegistry

_HELLO_SHA256 = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def _connector() -> LocalFolderConnector:
    return LocalFolderConnector()


def test_given_new_file_when_pulling_local_folder_then_upsert_contains_sha256(
    tmp_path: Path,
) -> None:
    source_id = uuid.uuid4()
    path = tmp_path / "notes.md"
    path.write_text("hello", encoding="utf-8")

    result = _connector().pull(source_id, {"path": str(tmp_path)}, None)

    assert len(result.events) == 1
    event = result.events[0]
    assert isinstance(event, UpsertEvent)
    assert event.uri == "notes.md"
    assert event.content_hash == _HELLO_SHA256
    assert event.data == b"hello"
    assert event.metadata == {"source_id": str(source_id)}
    assert result.cursor == {
        "files": {
            "notes.md": {"hash": _HELLO_SHA256, "doc_id": event.doc_id},
        }
    }


def test_given_unchanged_file_when_pulling_with_cursor_then_no_events(tmp_path: Path) -> None:
    source_id = uuid.uuid4()
    (tmp_path / "notes.md").write_text("hello", encoding="utf-8")

    first = _connector().pull(source_id, {"path": str(tmp_path)}, None)
    unchanged = _connector().pull(source_id, {"path": str(tmp_path)}, first.cursor)

    assert unchanged.events == []
    assert unchanged.cursor == first.cursor


def test_given_changed_file_when_pulling_with_cursor_then_upsert_carries_new_hash(
    tmp_path: Path,
) -> None:
    source_id = uuid.uuid4()
    path = tmp_path / "notes.md"
    path.write_text("hello", encoding="utf-8")
    first = _connector().pull(source_id, {"path": str(tmp_path)}, None)
    path.write_text("hello world", encoding="utf-8")

    result = _connector().pull(source_id, {"path": str(tmp_path)}, first.cursor)

    assert len(result.events) == 1
    event = result.events[0]
    assert isinstance(event, UpsertEvent)
    assert event.content_hash == hashlib.sha256(b"hello world").hexdigest()
    assert event.content_hash != _HELLO_SHA256
    # The document keeps its identity so the pipeline replaces, not duplicates.
    assert event.doc_id == first.events[0].doc_id
    assert result.cursor == {
        "files": {
            "notes.md": {"hash": event.content_hash, "doc_id": event.doc_id},
        }
    }


def test_given_deleted_file_when_pulling_local_folder_then_delete_uses_stable_doc_id(
    tmp_path: Path,
) -> None:
    source_id = uuid.uuid4()
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    connector = _connector()
    first = connector.pull(source_id, {"path": str(tmp_path)}, None)
    path.unlink()

    result = connector.pull(source_id, {"path": str(tmp_path)}, first.cursor)

    assert len(result.events) == 1
    assert isinstance(result.events[0], DeleteEvent)
    assert result.events[0].doc_id == first.events[0].doc_id
    assert result.events[0].uri == "notes.txt"
    assert result.cursor == {"files": {}}


def test_given_nested_file_when_pulling_then_uri_is_relative_posix_path(tmp_path: Path) -> None:
    source_id = uuid.uuid4()
    nested = tmp_path / "docs" / "sub"
    nested.mkdir(parents=True)
    (nested / "guide.md").write_text("body", encoding="utf-8")

    result = _connector().pull(source_id, {"path": str(tmp_path)}, None)

    assert [event.uri for event in result.events] == ["docs/sub/guide.md"]


def test_given_symlink_file_when_pulling_then_file_is_skipped(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.md"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "linked.md"
    link.symlink_to(outside)

    result = _connector().pull(uuid.uuid4(), {"path": str(tmp_path)}, None)

    assert result.events == []


def test_given_unsupported_extension_when_pulling_then_file_skipped(tmp_path: Path) -> None:
    source_id = uuid.uuid4()
    (tmp_path / "notes.md").write_text("hello", encoding="utf-8")
    (tmp_path / "malware.exe").write_bytes(b"MZ\x90\x00")

    result = _connector().pull(source_id, {"path": str(tmp_path)}, None)

    assert [event.uri for event in result.events] == ["notes.md"]
    assert set(result.cursor["files"]) == {"notes.md"}


def test_given_empty_folder_when_pulling_then_no_events_and_empty_cursor(tmp_path: Path) -> None:
    result = _connector().pull(uuid.uuid4(), {"path": str(tmp_path)}, None)

    assert result.events == []
    assert result.cursor == {"files": {}}


def test_given_event_limit_when_pulling_then_continuation_preserves_all_changes(
    tmp_path: Path,
) -> None:
    source_id = uuid.uuid4()
    for name in ("one.md", "two.md", "three.md"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    connector = _connector()
    config = {"path": str(tmp_path), "_max_events": 1}

    first = connector.pull(source_id, config, None)
    second = connector.pull(source_id, config, first.cursor)
    third = connector.pull(source_id, config, second.cursor)

    assert [event.uri for event in first.events + second.events + third.events] == [
        "one.md",
        "three.md",
        "two.md",
    ]
    assert third.cursor == {
        "files": {
            name: {
                "hash": hashlib.sha256(name.encode()).hexdigest(),
                "doc_id": str(uuid.uuid5(connector._DOC_NAMESPACE, f"{source_id}:{name}")),
            }
            for name in ("one.md", "two.md", "three.md")
        }
    }


def test_given_missing_folder_when_pulling_then_no_events(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"

    result = _connector().pull(uuid.uuid4(), {"path": str(missing)}, None)

    assert result.events == []


def test_given_path_outside_configured_sources_root_when_pulling_then_rejected(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="outside"):
        _connector().pull(
            uuid.uuid4(),
            {"path": str(outside), "_allowed_root": str(tmp_path / "allowed")},
            None,
        )


def test_given_file_over_bound_when_pulling_then_file_is_rejected_before_reading(
    tmp_path: Path,
) -> None:
    (tmp_path / "notes.md").write_bytes(b"hello")

    with pytest.raises(ValueError, match="exceeds"):
        _connector().pull(
            uuid.uuid4(), {"path": str(tmp_path), "_max_file_bytes": 4}, None
        )


def test_given_different_sources_when_pulling_same_file_then_doc_ids_differ(
    tmp_path: Path,
) -> None:
    (tmp_path / "notes.md").write_text("hello", encoding="utf-8")

    first = _connector().pull(uuid.uuid4(), {"path": str(tmp_path)}, None)
    second = _connector().pull(uuid.uuid4(), {"path": str(tmp_path)}, None)

    assert first.events[0].doc_id != second.events[0].doc_id


def test_given_renamed_file_when_pulling_then_old_uri_deleted_and_new_uri_upserted(
    tmp_path: Path,
) -> None:
    source_id = uuid.uuid4()
    (tmp_path / "before.md").write_text("hello", encoding="utf-8")
    first = _connector().pull(source_id, {"path": str(tmp_path)}, None)
    (tmp_path / "before.md").rename(tmp_path / "after.md")

    result = _connector().pull(source_id, {"path": str(tmp_path)}, first.cursor)

    ops = {(type(event).__name__, event.uri) for event in result.events}
    assert ops == {("DeleteEvent", "before.md"), ("UpsertEvent", "after.md")}


def test_given_root_alias_when_pulling_then_config_root_accepted(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("hello", encoding="utf-8")

    result = _connector().pull(uuid.uuid4(), {"root": str(tmp_path)}, None)

    assert [event.uri for event in result.events] == ["notes.md"]


@pytest.mark.parametrize("config", [{}, {"path": ""}, {"path": 123}])
def test_given_missing_path_when_pulling_then_value_error(config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="path"):
        _connector().pull(uuid.uuid4(), config, None)


def test_given_registered_connector_when_resolving_type_then_registry_returns_it() -> None:
    connector = LocalFolderConnector()
    registry = SourceConnectorRegistry({"local_folder": connector})

    assert registry.get("local_folder") is connector


def test_given_unknown_type_when_resolving_then_value_error() -> None:
    registry = SourceConnectorRegistry()

    with pytest.raises(ValueError, match="unknown source type"):
        registry.get("git")


def test_given_default_registry_when_listing_types_then_local_folder_registered() -> None:
    registry = SourceConnectorRegistry()

    assert registry.known_types() == {"local_folder"}
    assert isinstance(registry.get("local_folder"), LocalFolderConnector)


def test_given_registered_type_when_registering_replacement_then_new_connector_wins() -> None:
    registry = SourceConnectorRegistry()
    replacement = LocalFolderConnector()

    registry.register("local_folder", replacement)

    assert registry.get("local_folder") is replacement
