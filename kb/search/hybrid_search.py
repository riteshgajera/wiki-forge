"""
Hybrid search: BM25 (keyword) + FAISS (semantic) fused with Reciprocal Rank Fusion.
Falls back gracefully if FAISS embeddings are not available.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kb.utils.logging import get_logger

logger = get_logger("search.hybrid")


@dataclass
class SearchHit:
    slug: str
    subdir: str
    title: str
    score: float
    snippet: str
    tags: list[str] = field(default_factory=list)
    bm25_rank: int = 0
    vector_rank: int = 0
    method: str = "hybrid"


class BM25Index:
    """In-memory BM25 index over wiki articles. Rebuilt on demand."""

    def __init__(self) -> None:
        self._corpus: list[dict] = []  # {slug, subdir, title, tokens, content}
        self._idf: dict[str, float] = {}
        self._k1 = 1.5
        self._b = 0.75
        self._avg_dl = 0.0

    def build(self, wiki_manager: Any) -> None:
        """Build index from all wiki articles."""
        self._corpus = []
        for path in wiki_manager.list_articles():
            content = path.read_text(encoding="utf-8")
            # Strip frontmatter for indexing
            body = re.sub(r"^---\n.*?---\n", "", content, flags=re.DOTALL)
            # Extract title from frontmatter
            title_m = re.search(r"^title:\s*(.+)$", content, re.MULTILINE)
            title = title_m.group(1).strip() if title_m else path.stem
            tags_m = re.search(r"^tags:\s*\[(.+)\]$", content, re.MULTILINE)
            tags = [t.strip().strip("'\"") for t in tags_m.group(1).split(",") if t.strip()] if tags_m else []
            tokens = self._tokenize(body + " " + title)
            self._corpus.append({
                "slug": path.stem,
                "subdir": path.parent.name,
                "title": title,
                "tokens": tokens,
                "content": content,
                "tags": tags,
            })

        if not self._corpus:
            return

        # IDF
        df: dict[str, int] = {}
        for doc in self._corpus:
            for term in set(doc["tokens"]):
                df[term] = df.get(term, 0) + 1
        N = len(self._corpus)
        self._idf = {t: math.log((N - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
        self._avg_dl = sum(len(d["tokens"]) for d in self._corpus) / N
        logger.info("bm25_built", docs=N, vocab=len(self._idf))

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Return (corpus_idx, bm25_score) pairs sorted by score."""
        if not self._corpus:
            return []
        q_tokens = self._tokenize(query)
        scores: list[tuple[int, float]] = []
        for i, doc in enumerate(self._corpus):
            score = 0.0
            dl = len(doc["tokens"])
            tf_map: dict[str, int] = {}
            for t in doc["tokens"]:
                tf_map[t] = tf_map.get(t, 0) + 1
            for qt in q_tokens:
                if qt not in self._idf:
                    continue
                tf = tf_map.get(qt, 0)
                idf = self._idf[qt]
                tf_norm = (tf * (self._k1 + 1)) / (
                    tf + self._k1 * (1 - self._b + self._b * dl / max(self._avg_dl, 1))
                )
                score += idf * tf_norm
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [w for w in text.split() if len(w) > 2]

    def get_doc(self, idx: int) -> dict:
        return self._corpus[idx]

    def get_snippet(self, idx: int, query: str, length: int = 200) -> str:
        content = self._corpus[idx]["content"]
        # Find first occurrence of any query term
        terms = self._tokenize(query)
        body = re.sub(r"^---\n.*?---\n", "", content, flags=re.DOTALL)
        pos = -1
        for t in terms:
            idx2 = body.lower().find(t)
            if idx2 >= 0 and (pos == -1 or idx2 < pos):
                pos = idx2
        if pos >= 0:
            start = max(0, pos - 80)
            return body[start:start + length].strip()
        return body[:length].strip()


def reciprocal_rank_fusion(
    bm25_results: list[tuple[int, float]],
    vector_results: list[Any],  # SearchResult from faiss_store
    bm25_corpus: list[dict],
    k_rrf: int = 60,
) -> list[dict]:
    """
    Fuse BM25 and vector rankings using Reciprocal Rank Fusion.
    RRF score = sum(1 / (k + rank)) across retrieval methods.
    """
    scores: dict[str, dict] = {}

    # BM25 contribution
    for rank, (corpus_idx, _) in enumerate(bm25_results):
        doc = bm25_corpus[corpus_idx]
        key = f"{doc['subdir']}/{doc['slug']}"
        if key not in scores:
            scores[key] = {
                "slug": doc["slug"], "subdir": doc["subdir"],
                "title": doc["title"], "tags": doc.get("tags", []),
                "rrf": 0.0, "bm25_rank": 9999, "vector_rank": 9999,
                "corpus_idx": corpus_idx,
            }
        scores[key]["rrf"] += 1.0 / (k_rrf + rank + 1)
        scores[key]["bm25_rank"] = rank + 1

    # Vector contribution
    for rank, vr in enumerate(vector_results):
        meta = vr.metadata
        slug = meta.get("slug", vr.doc_id)
        subdir = meta.get("subdir", "concepts")
        key = f"{subdir}/{slug}"
        if key not in scores:
            scores[key] = {
                "slug": slug, "subdir": subdir,
                "title": meta.get("title", slug), "tags": meta.get("tags", []),
                "rrf": 0.0, "bm25_rank": 9999, "vector_rank": 9999,
                "corpus_idx": -1,
            }
        scores[key]["rrf"] += 1.0 / (k_rrf + rank + 1)
        scores[key]["vector_rank"] = rank + 1

    return sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)


class HybridSearch:
    """Unified hybrid search: BM25 + FAISS + RRF."""

    def __init__(self, wiki_manager: Any, vector_pipeline: Any | None = None) -> None:
        self.wiki = wiki_manager
        self.vector = vector_pipeline
        self.bm25 = BM25Index()
        self._built = False

    def build(self) -> None:
        """Build BM25 index. Called after compile."""
        self.bm25.build(self.wiki)
        self._built = True

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[SearchHit]:
        if not self._built:
            self.build()

        bm25_raw: list[tuple[int, float]] = []
        vector_raw: list[Any] = []

        if mode in ("bm25", "hybrid"):
            bm25_raw = self.bm25.search(query, k=k * 2)

        if mode in ("vector", "hybrid") and self.vector:
            try:
                vector_raw = self.vector.search(query, k=k * 2)
            except Exception as e:
                logger.warning("vector_search_error", error=str(e))

        if mode == "bm25" or not vector_raw:
            # BM25 only
            hits = []
            for rank, (ci, score) in enumerate(bm25_raw[:k]):
                doc = self.bm25.get_doc(ci)
                hits.append(SearchHit(
                    slug=doc["slug"], subdir=doc["subdir"], title=doc["title"],
                    score=score, snippet=self.bm25.get_snippet(ci, query),
                    tags=doc.get("tags", []), bm25_rank=rank + 1, method="bm25",
                ))
            return hits

        if mode == "vector" or not bm25_raw:
            # Vector only
            return [SearchHit(
                slug=vr.metadata.get("slug", vr.doc_id),
                subdir=vr.metadata.get("subdir", "concepts"),
                title=vr.metadata.get("title", vr.doc_id),
                score=vr.score, snippet=vr.text[:200],
                tags=vr.metadata.get("tags", []),
                vector_rank=i + 1, method="vector",
            ) for i, vr in enumerate(vector_raw[:k])]

        # Hybrid: RRF fusion
        fused = reciprocal_rank_fusion(bm25_raw, vector_raw, self.bm25._corpus)
        hits = []
        for f in fused[:k]:
            ci = f.get("corpus_idx", -1)
            snippet = self.bm25.get_snippet(ci, query) if ci >= 0 else ""
            hits.append(SearchHit(
                slug=f["slug"], subdir=f["subdir"], title=f["title"],
                score=f["rrf"], snippet=snippet, tags=f.get("tags", []),
                bm25_rank=f["bm25_rank"], vector_rank=f["vector_rank"],
                method="hybrid",
            ))
        return hits
