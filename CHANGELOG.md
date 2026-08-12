# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - Unreleased

### Added

- OpenAI-compatible `/v1/chat/completions` endpoint with JSON and SSE streaming support
- `/v1/models` endpoint for model discovery
- FAISS vector index with hybrid retrieval (L2 semantic + keyword matching)
- Multi-backend LLM support: Ollama (local) and OpenAI-compatible API (cloud)
- Bilingual detection (EN/ZH) with language-filtered retrieval
- Auto-indexing via background file watcher
- Per-IP rate limiting middleware
- REST ingest API (`/api/ingest`, `/api/ingest/file`, `/api/ingest/stats`)
- Docker Compose deployment (backend + Ollama)
- Pluggable architecture with four extension points:
  - `ChunkingPlugin` — custom text chunking strategies
  - `RetrievalPlugin` — custom retrieval strategies
  - `ResponsePlugin` — custom prompt building and response formatting
  - `IngestionPlugin` — custom data source integration
