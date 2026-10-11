"""Registry for interaction-layer source connectors."""

from __future__ import annotations

from collections.abc import Mapping

from app.domain.events import DeleteEvent, SourceEvent, UpsertEvent
from app.interaction.local_folder import LocalFolderConnector
from app.interaction.ports import PullResult, SourceConnector

__all__ = ["DeleteEvent", "PullResult", "SourceConnector", "SourceEvent", "UpsertEvent"]


class SourceConnectorRegistry:
    """Resolve a persisted source type to its connector implementation."""

    def __init__(self, connectors: Mapping[str, SourceConnector] | None = None) -> None:
        self._connectors = dict(connectors or {"local_folder": LocalFolderConnector()})

    def register(self, type_name: str, connector: SourceConnector) -> None:
        """Register or replace a connector type."""
        self._connectors[type_name] = connector

    def get(self, type_name: str) -> SourceConnector:
        """Return a connector or raise a clear validation error."""
        try:
            return self._connectors[type_name]
        except KeyError as exc:
            raise ValueError(f"unknown source type {type_name!r}") from exc

    def known_types(self) -> set[str]:
        """Return registered source type names."""
        return set(self._connectors)


_registry = SourceConnectorRegistry()


def get_source_registry() -> SourceConnectorRegistry:
    """Return the process-wide connector registry."""
    return _registry