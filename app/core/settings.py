"""Application settings: ``config.yaml`` plus environment overrides.

Nested keys are overridden with the ``__`` delimiter, e.g. ``DEFAULTS__CHAT_MODEL``
or ``QDRANT__HOST``. Environment variables take precedence over the YAML file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

CONFIG_FILE = Path(__file__).resolve().parents[2] / "config.yaml"


class ModelDefaults(BaseModel):
    """Global model + retrieval defaults; per-KB overrides fall back to these."""

    embed_model: str = "ollama/bge-m3"
    chat_model: str = "ollama/llama3.2"
    vision_model: str = "ollama/llama3.2-vision"
    chunk_size: int = 700
    chunk_overlap: int = 80
    top_k: int = 6
    max_ctx_chars: int = 3000


class OllamaProviderConfig(BaseModel):
    base_url: str = "http://localhost:11434"


class OpenAIProviderConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"


class ProvidersConfig(BaseModel):
    ollama: OllamaProviderConfig = Field(default_factory=OllamaProviderConfig)
    openai: OpenAIProviderConfig = Field(default_factory=OpenAIProviderConfig)


class OidcConfig(BaseModel):
    """Reserved for the multi-user roadmap; unused while ``auth.provider != oidc``."""

    issuer: str = ""
    audience: str = ""
    roles_claim: str = "groups"


class AuthConfig(BaseModel):
    provider: str = "none"
    api_key_env: str = "RAG_API_KEY"
    oidc: OidcConfig = Field(default_factory=OidcConfig)


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 50
    max_event_batch: int = Field(default=100, gt=0, le=10_000)


class StorageConfig(BaseModel):
    root: str = "./storage"
    sources_root: str | None = None


class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333


class JobsConfig(BaseModel):
    keep_result_s: int = 3600


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    defaults: ModelDefaults = Field(default_factory=ModelDefaults)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    database_url: str = "sqlite:///./rag.db"
    redis_url: str = "redis://localhost:6379"
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    jobs: JobsConfig = Field(default_factory=JobsConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]
        if CONFIG_FILE.exists():
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=CONFIG_FILE))
        sources.append(file_secret_settings)
        return tuple(sources)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings()
