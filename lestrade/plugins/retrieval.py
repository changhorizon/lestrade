import re
from abc import ABC, abstractmethod
from typing import Callable

import faiss


class RetrievalPlugin(ABC):
    """检索策略的抽象基类。

    垂直领域可继承此类实现自定义检索逻辑：
    - 法律领域按法条编号精确匹配
    - 电商领域按 SKU/类目加权
    - 医疗领域按症状→诊断关联推理
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        k: int,
        lang: str,
        index: faiss.Index,
        entries: list[dict],
        embed_fn: Callable,
    ) -> list[tuple[str, str, float]]:
        """返回 [(text, source, score), ...]，按 score 升序排列。"""
        ...


class HybridRetrieval(RetrievalPlugin):
    """默认混合检索策略：FAISS 向量检索 + 中文关键词匹配。"""

    async def search(
        self,
        query: str,
        k: int,
        lang: str,
        index: faiss.Index,
        entries: list[dict],
        embed_fn: Callable,
    ) -> list[tuple[str, str, float]]:

        if index is None or index.ntotal == 0 or not entries:
            return []

        candidates = entries
        if lang:
            prefix = f'{lang}/'
            candidates = [
                e for e in entries
                if e['source'].startswith(prefix) or not e['source'].startswith(('en/', 'zh/'))
            ]

        results = []

        emb = await embed_fn(query)
        import numpy as np
        vec = np.array([emb], dtype=np.float32)
        distances, indices = index.search(vec, min(k * 4, index.ntotal))
        id_map = {e["id"]: e for e in candidates}
        max_dist = max(distances[0]) if distances[0].size > 0 else 1
        for dist, idx in zip(distances[0], indices[0]):
            e = id_map.get(int(idx))
            if e:
                score = float(dist / max_dist) if max_dist > 0 else 0.0
                if score <= 0.85:
                    results.append((e["text"], e["source"], score, e["id"]))

        raw_terms = re.split(r"[\s,，。？?！!；;：:、()（）/\\\-]+", query)
        terms = set()
        for t in raw_terms:
            for sub in re.split(r"[的的地得与和或是及以及]", t):
                sub = sub.strip(".")
                if len(sub) >= 2:
                    terms.add(sub.lower())
            zh_chars = re.findall(r'[\u4e00-\u9fff]', t)
            for i in range(len(zh_chars) - 1):
                terms.add(zh_chars[i] + zh_chars[i + 1])

        if terms:
            for e in candidates:
                text = e["text"].lower()
                hit = sum(1 for t in terms if t in text)
                if hit >= 3 or (hit >= 2 and any(t in e["source"].lower() for t in terms)):
                    nav_links = text.count("](/")
                    content_score = 0.30 if hit >= 2 else 0.35
                    nav_penalty = 0.20 if nav_links >= 3 else 0.0
                    score = content_score + nav_penalty
                    results.append((e["text"], e["source"], score, e["id"]))

        best = {}
        for text, src, score, eid in results:
            if eid not in best or score < best[eid][2]:
                best[eid] = (text, src, score, eid)
        results = sorted(best.values(), key=lambda x: x[2])

        return [(t, s, d) for t, s, d, _ in results[:k]]
