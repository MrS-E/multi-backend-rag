# Multi-BackendRag

Headless RAG server (API only) serving multiple isolated knowledge bases, each
with its own vector index and model configuration.

Layered architecture: **API layer → broker (Redis/arq) → RAG core**, so the
interaction layer (where content comes from) and the RAG core (how content is
indexed/queried) evolve independently.

## Stack

- **FastAPI** API, **arq** (Redis) background jobs
- **Qdrant** — one collection per knowledge base
- **SQLite** (SQLAlchemy + Alembic) — KB/document/source metadata
- **Docling** for PDF conversion, **Ollama** / OpenAI-compatible model providers
- `uv` for packaging, `docker compose` for Redis + Qdrant

## Setup

```bash
uv sync
cp .env.example .env
docker compose up -d          # Qdrant + Redis
uv run alembic upgrade head   # create the SQLite schema
uv run uvicorn app.main:create_app --factory
```

Health check: `GET /health` → `{"status": "ok", "version": "0.1.0"}`.

## Configuration

Global defaults live in `config.yaml`; environment variables override them using
the `__` nested delimiter (e.g. `DEFAULTS__CHAT_MODEL`, `QDRANT__HOST`).
Per-KB model overrides are stored in SQLite and fall back to these defaults.
