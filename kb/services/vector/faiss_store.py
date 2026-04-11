"""
FAISS-backed vector store for semantic search.
Stores embeddings with metadata, supports incremental upsert and similarity search.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kb.utils.logging import get_logger

logger = get_logger("services.vector")


@dataclass
class VectorEntry:
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    doc_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class FAISSVectorStore:
    """
    Local FAISS vector store. Persists index + metadata to disk.
    Falls back gracefully if FAISS is unavailable.
    """

    def __init__(self, store_dir: str | Path, dimension: int = 768) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self.index_path = self.store_dir / "index.faiss"
        self.meta_path = self.store_dir / "metadata.pkl"
        self._index: Any = None
        self._metadata: list[VectorEntry] = []
        self._load()

    def _load(self) -> None:
        try:
            import faiss
            if self.index_path.exists() and self.meta_path.exists():
                self._index = faiss.read_index(str(self.index_path))
                with open(self.meta_path, "rb") as f:
                    self._metadata = pickle.load(f)
                logger.info("vector_store_loaded",
                           vectors=len(self._metadata), dim=self.dimension)
            else:
                self._index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine with normalized vecs)
                logger.info("vector_store_new", dim=self.dimension)
        except ImportError:
            logger.warning("faiss_unavailable", msg="Install faiss-cpu for vector search")
            self._index = None

    def _save(self) -> None:
        if self._index is None:
            return
        try:
            import faiss
            faiss.write_index(self._index, str(self.index_path))
            with open(self.meta_path, "wb") as f:
                pickle.dump(self._metadata, f)
        except Exception as e:
            logger.error("vector_store_save_error", error=str(e))

    def upsert(self, entry: VectorEntry, embedding: list[float]) -> None:
        if self._index is None:
            return

        vec = np.array([embedding], dtype=np.float32)
        # Normalize for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # Remove existing entry for same doc_id
        self._metadata = [m for m in self._metadata if m.doc_id != entry.doc_id]

        self._index.add(vec)
        self._metadata.append(entry)
        self._save()
        logger.debug("vector_upsert", doc_id=entry.doc_id)

    def search(self, query_embedding: list[float], k: int = 5) -> list[SearchResult]:
        if self._index is None or self._index.ntotal == 0:
            return []

        vec = np.array([query_embedding], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            entry = self._metadata[idx]
            results.append(SearchResult(
                doc_id=entry.doc_id,
                score=float(score),
                text=entry.text,
                metadata=entry.metadata,
            ))
        return results

    def delete(self, doc_id: str) -> bool:
        before = len(self._metadata)
        self._metadata = [m for m in self._metadata if m.doc_id != doc_id]
        if len(self._metadata) < before:
            # FAISS flat index doesn't support deletion; rebuild
            self._rebuild()
            return True
        return False

    def _rebuild(self) -> None:
        """Rebuild index from metadata (needed after deletions)."""
        logger.info("vector_store_rebuild", count=len(self._metadata))
        # We don't store raw embeddings, so we can't rebuild without re-embedding.
        # For now, just clear the index — caller should re-embed.
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)
        except ImportError:
            pass

    @property
    def count(self) -> int:
        return len(self._metadata)


class EmbeddingPipeline:
    """Orchestrates embedding generation and vector store upsert."""

    def __init__(self, vector_store: FAISSVectorStore, llm_provider: Any) -> None:
        self.store = vector_store
        self.llm = llm_provider

    def embed_and_store(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Embed text and upsert into vector store. Returns True on success."""
        try:
            response = self.llm.embed(text[:4096])
            if not response.embedding:
                logger.warning("empty_embedding", doc_id=doc_id)
                return False

            entry = VectorEntry(
                doc_id=doc_id,
                text=text[:500],  # Store snippet for result display
                metadata=metadata or {},
            )
            self.store.upsert(entry, response.embedding)
            return True
        except Exception as e:
            logger.error("embed_error", doc_id=doc_id, error=str(e))
            return False

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Embed query and search vector store."""
        try:
            response = self.llm.embed(query)
            if not response.embedding:
                return []
            return self.store.search(response.embedding, k=k)
        except Exception as e:
            logger.error("search_error", query=query[:50], error=str(e))
            return []
