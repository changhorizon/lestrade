import json
import logging
import os
from typing import List, Optional

import faiss
import numpy as np

from .. import config
from ..plugins import get_chunking, get_retrieval
from .embed import embed_text

logger = logging.getLogger(__name__)

META_PATH = config.FAISS_INDEX_PATH + ".meta.json"


class RAGEngine:
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.entries: List[dict] = []
        self.source_ids: dict[str, list[int]] = {}
        self.next_id: int = 0
        self._chunking = get_chunking()
        self._retrieval = get_retrieval()
        self._load()

    def _load(self):
        os.makedirs(os.path.dirname(config.FAISS_INDEX_PATH), exist_ok=True)
        rebuild = False
        if os.path.exists(META_PATH):
            with open(META_PATH, "r") as f:
                meta = json.load(f)
            stored_model = meta.get("embed_model", "")
            if stored_model and stored_model != config.EMBEDDING_MODEL:
                logger.warning(
                    "Embedding model changed: %s -> %s. Rebuilding index.",
                    stored_model, config.EMBEDDING_MODEL,
                )
                rebuild = True
                self.entries = []
                self.source_ids = {}
                self.next_id = 0
            else:
                self.next_id = meta.get("next_id", 0)
                self.entries = meta.get("entries", [])
                self.source_ids = meta.get("source_ids", {})

        index_path = config.FAISS_INDEX_PATH
        if rebuild:
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(META_PATH):
                os.remove(META_PATH)

        if os.path.exists(index_path):
            try:
                self.index = faiss.read_index(index_path)
            except Exception as e:
                logger.warning("FAISS index corrupted (%s), rebuilding.", e)
                os.remove(index_path)
                self.entries = []
                self.source_ids = {}
                self.next_id = 0
        else:
            dim = 1024
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))

    def _save(self):
        os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
        faiss.write_index(self.index, config.FAISS_INDEX_PATH)
        with open(META_PATH, "w") as f:
            json.dump({
                "next_id": self.next_id,
                "embed_model": config.EMBEDDING_MODEL,
                "entries": self.entries,
                "source_ids": self.source_ids,
            }, f, ensure_ascii=False, indent=2)

    async def add_text(self, text: str, source: Optional[str] = None) -> int:
        chunks = self._chunking.chunk(text)
        vectors = []
        texts = []
        ids = []

        for chunk in chunks:
            enriched = f"[{source}] {chunk}" if source else chunk
            emb = await embed_text(enriched)
            vec = np.array([emb], dtype=np.float32)

            if self.index.ntotal == 0:
                dim = vec.shape[1]
                self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))

            uid = self.next_id
            self.next_id += 1
            ids.append(uid)
            vectors.append(vec)
            texts.append(enriched)

        if vectors:
            all_vecs = np.vstack(vectors)
            id_array = np.array(ids, dtype=np.int64)
            self.index.add_with_ids(all_vecs, id_array)

            for uid, enriched in zip(ids, texts):
                self.entries.append({"id": uid, "text": enriched, "source": source or ""})
                src = source or ""
                self.source_ids.setdefault(src, []).append(uid)

            self._save()

        return len(chunks)

    def remove_by_source(self, source: str):
        if source not in self.source_ids:
            return
        old_ids = self.source_ids.pop(source)
        self.index.remove_ids(np.array(old_ids, dtype=np.int64))
        self.entries = [e for e in self.entries if e["id"] not in old_ids]
        self._save()

    async def search(self, query: str, k: int = 5, lang: str = '') -> List[tuple[str, str, float]]:
        if self.index is None or self.index.ntotal == 0 or not self.entries:
            return []

        return await self._retrieval.search(
            query=query,
            k=k,
            lang=lang,
            index=self.index,
            entries=self.entries,
            embed_fn=embed_text,
        )


engine = RAGEngine()
