import httpx

from .. import config

_EXPECTED_DIMS = {
    "nomic-embed-text": 768,
    "bge-m3": 1024,
    "bge-large": 1024,
}


async def embed_text(text: str) -> list[float]:
    payload = {
        "model": config.EMBEDDING_MODEL,
        "input": text,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/embed",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    emb = data["embeddings"][0]
    expected = _EXPECTED_DIMS.get(config.EMBEDDING_MODEL)
    if expected and len(emb) != expected:
        raise ValueError(
            f"Embedding model '{config.EMBEDDING_MODEL}' returned dimension {len(emb)}, "
            f"expected {expected}. Clear FAISS index and reindex."
        )
    return emb
