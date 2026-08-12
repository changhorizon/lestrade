# Lestrade

RAG knowledge engine with a pluggable architecture. Drop in documents, point at an LLM, and build domain-specific AI assistants.

## Features

- **OpenAI-compatible API** — `/v1/chat/completions` and `/v1/models` endpoints
- **RAG with FAISS** — Hybrid semantic + keyword search over your documents
- **Multi-backend LLM** — Local (Ollama) or cloud (OpenAI / DeepSeek / any OpenAI-compatible API)
- **Bilingual** — Auto-detects English / Chinese, with language-filtered retrieval
- **Auto-indexing** — Watches knowledge base directories, re-indexes on changes
- **Rate limiting** — Per-IP rate limit on chat completions
- **Streaming & JSON** — Both SSE streaming and standard JSON responses
- **Pluggable architecture** — Swap chunking, retrieval, response, and ingestion strategies via environment variables
- **Docker** — One `docker compose up` to run everything

## Quick Start

### Prerequisites

- [Ollama](https://ollama.com) installed and running (or use cloud API directly)
- Python 3.9+

### 1. Install

```bash
pip install lestrade
```

Or for development:

```bash
cd source
pip install -e ".[dev]"
```

### 2. Prepare Knowledge Base

Put your `.md` or `.txt` files in `data/kb/`. Organize by language:

```
data/kb/
├── en/
│   ├── getting-started.md
│   └── faq.md
└── zh/
    ├── getting-started.md
    └── faq.md
```

### 3. Configure

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

Key settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `CHAT_MODEL` | LLM model to use | `deepseek-chat` |
| `LLM_API_KEY` | API key for cloud LLM | (empty) |
| `LLM_BASE_URL` | OpenAI-compatible API base URL | `https://api.openai.com/v1` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://ollama:11434` |
| `EMBEDDING_MODEL` | Embedding model (Ollama) | `nomic-embed-text` |
| `KB_DIRS` | Comma-separated KB directories | (empty) |
| `RATE_LIMIT_MAX_REQUESTS` | Max requests per window | `10` |

### 4. Run

```bash
uvicorn lestrade.main:app --reload

# or Docker
docker compose up -d
```

## API

The service exposes an OpenAI-compatible API at `http://localhost:8000`:

### Chat Completions

```
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "deepseek-chat",
  "messages": [{"role": "user", "content": "What is your return policy?"}],
  "stream": false
}
```

### List Models

```
GET /v1/models
```

### Ingest Documents

```
POST /api/ingest
Content-Type: multipart/form-data

text=Document content here&source=my-doc.md
POST /api/ingest/file
Content-Type: multipart/form-data

file=@document.md
```

### Health Check

```
GET /health
```

## Pluggable Architecture

Lestrade provides four extension points, each swappable via environment variables.

| Plugin | Input | Output | Env Variable |
|--------|-------|--------|--------------|
| Chunking | text | chunks | `LESTRADE_CHUNKING` |
| Retrieval | query + index | ranked docs | `LESTRADE_RETRIEVAL` |
| Response | context + raw LLM output | prompt + formatted text | `LESTRADE_RESPONSE` |
| Ingestion | data source | texts | `LESTRADE_INGESTION`

### Creating a Custom Plugin

Inherit from the base class in your own Python package, then point lestrade at it.

```python
# my_legal_bot/plugins.py
from lestrade.plugins import ResponsePlugin
from lestrade.llm.base import ChatMessage

class LegalResponse(ResponsePlugin):
    def build_messages(self, contexts, user_message, lang):
        context_text = "\n\n".join(c[0] for c in contexts)
        prompt = (
            f"你是一位法律顾问。仅根据以下法律条文回答问题，"
            f"必须引用具体法条编号。回答末尾添加免责声明。\n\n"
            f"法律条文：\n{context_text}\n\n"
            f"问题：{user_message}"
        )
        return [ChatMessage(role="user", content=prompt)]

    def format_response(self, text):
        return text + "\n\n---\n以上回答仅供参考，不构成法律意见。"
```

```bash
# Install your plugin alongside lestrade
pip install lestrade my-legal-bot

# Point lestrade at your plugin
export LESTRADE_RESPONSE=my_legal_bot.plugins:LegalResponse
uvicorn lestrade.main:app
```

### Extension Points

| Plugin | Base Class | Method | Purpose |
|--------|-----------|--------|---------|
| Chunking | `ChunkingPlugin` | `chunk(text, max_chars) -> list[str]` | Split documents into searchable chunks |
| Retrieval | `RetrievalPlugin` | `search(query, k, lang, index, entries, embed_fn) -> list[tuple]` | Rank and return relevant chunks |
| Response | `ResponsePlugin` | `build_messages(contexts, user_msg, lang) -> list[ChatMessage]` | Build the LLM prompt from context |
| Response | `ResponsePlugin` | `format_response(text) -> str` | Post-process LLM output |
| Ingestion | `IngestionPlugin` | `start(on_content)` | Connect external data sources |

### Vertical Domain Examples

```
lestrade                     # open-source base
└── my-legal-bot/           # vertical: legal Q&A
    └── plugins.py           #   LegalChunking (split by clause)
                             #   LegalRetrieval (cite article numbers)
                             #   LegalResponse (attach disclaimers)

└── my-medical-bot/         # vertical: medical triage
    └── plugins.py           #   MedicalChunking (split by diagnosis)
                             #   MedicalResponse (symptom → triage suggestion)
```

Vertical packages only declare `lestrade` as a dependency — no code fork needed. Upgrade `lestrade` without touching vertical logic.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API Server | FastAPI + Uvicorn |
| Vector Store | FAISS (CPU) |
| Embeddings | Ollama (`nomic-embed-text` / `bge-m3`) |
| LLM | Ollama (local) or OpenAI-compatible API |
| Rate Limit | Custom Starlette middleware |

## License

MIT
