"""Real SQLite migration round trips for schema-critical revisions."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from alembic import command


def _alembic_config(database: Path) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database}")
    return config


def test_given_truthful_0004_database_when_round_tripping_0005_then_primary_key_and_data_survive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "migration.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database}")
    from app.core.settings import get_settings

    get_settings.cache_clear()
    config = _alembic_config(database)
    command.upgrade(config, "0004_raw_file_cleanup")

    kb_id = uuid.uuid4().hex
    engine = create_engine(f"sqlite:///{database}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO knowledge_bases "
                "(id, name, description, overrides, status, config_version, created_at) "
                "VALUES (:id, :name, :description, :overrides, :status, :version, :created_at)"
            ),
            {
                "id": kb_id,
                "name": "Before 0005",
                "description": "",
                "overrides": "{}",
                "status": "active",
                "version": 0,
                "created_at": "2026-09-23 00:00:00",
            },
        )
    engine.dispose()

    command.upgrade(config, "0005_reindex_recovery")
    engine = create_engine(f"sqlite:///{database}")
    inspector = inspect(engine)
    columns = {column["name"]: column for column in inspector.get_columns("knowledge_bases")}
    assert inspector.get_pk_constraint("knowledge_bases")["constrained_columns"] == ["id"]
    assert columns["reindex_generation"]["nullable"] is False
    assert columns["reindex_generation"]["default"] is None
    assert columns["reindex_started_at"]["nullable"] is True
    with engine.begin() as connection:
        row = connection.execute(
            text(
                "SELECT name, reindex_generation, reindex_started_at "
                "FROM knowledge_bases WHERE id = :id"
            ),
            {"id": kb_id},
        ).one()
        assert tuple(row) == ("Before 0005", 0, None)
    engine.dispose()

    command.downgrade(config, "0004_raw_file_cleanup")
    engine = create_engine(f"sqlite:///{database}")
    inspector = inspect(engine)
    assert inspector.get_pk_constraint("knowledge_bases")["constrained_columns"] == ["id"]
    assert {column["name"] for column in inspector.get_columns("knowledge_bases")} == {
        "id",
        "name",
        "description",
        "owner_id",
        "overrides",
        "status",
        "config_version",
        "created_at",
    }
    with engine.connect() as connection:
        assert connection.scalar(
            text("SELECT name FROM knowledge_bases WHERE id = :id"), {"id": kb_id}
        ) == "Before 0005"
    engine.dispose()