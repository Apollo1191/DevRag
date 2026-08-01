"""FAISS-backed vector store for embeddings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Protocol

import faiss
import numpy as np

from .qdrant_store import QdrantVectorStore
from .models import SearchResult


class VectorStoreProtocol(Protocol):
    """Protocol for vector store implementations."""

    def add(self, embeddings: Iterable[Iterable[float]], texts: Iterable[str], metadatas: Iterable[dict]) -> None:
        ...

    def search(self, embedding: Iterable[float], top_k: int = 5) -> list[SearchResult]:
        ...

    def save(self, directory: Path) -> None:
        ...

    def get_samples(self, max_samples: int = 5) -> list[SearchResult]:
        ...

    def count(self) -> int:
        ...

    def clear(self) -> None:
        ...

    def get_all(self, batch_size: int = 256) -> list[SearchResult]:
        ...


DEFAULT_VECTOR_BACKEND = os.getenv("DEVRAG_VECTOR_BACKEND", "faiss").strip().lower()
DEFAULT_QDRANT_COLLECTION = os.getenv("DEVRAG_QDRANT_COLLECTION", "devrag_chunks")
DEFAULT_QDRANT_DIMENSION = int(os.getenv("DEVRAG_QDRANT_DIMENSION", "384"))


class VectorStore:
    """In-memory FAISS index with parallel metadata storage."""

    def __init__(self, dimension: int) -> None:
        """Create a cosine-similarity FAISS index.

        Args:
            dimension: Embedding dimensionality.
        """

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self._texts: list[str] = []
        self._metadatas: list[dict] = []

    def add(self, embeddings: Iterable[Iterable[float]], texts: Iterable[str], metadatas: Iterable[dict]) -> None:
        """Add a batch of embeddings with texts and metadata."""

        vectors = np.array(list(embeddings), dtype=np.float32)
        if vectors.size == 0:
            return

        texts_list = list(texts)
        metadatas_list = list(metadatas)
        if len(texts_list) != len(metadatas_list) or len(texts_list) != len(vectors):
            raise ValueError("Embeddings, texts, and metadata must have the same length")

        self._normalize(vectors)
        self.index.add(vectors)
        self._texts.extend(texts_list)
        self._metadatas.extend(metadatas_list)

    def search(self, embedding: Iterable[float], top_k: int = 5) -> list[SearchResult]:
        """Search the index and return the top-k results."""

        vector = np.array([list(embedding)], dtype=np.float32)
        self._normalize(vector)
        scores, indices = self.index.search(vector, top_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(SearchResult(text=self._texts[idx], metadata=self._metadatas[idx], score=float(score)))
        return results

    def save(self, directory: Path) -> None:
        """Persist the FAISS index and metadata to disk."""

        directory.mkdir(parents=True, exist_ok=True)
        index_path = directory / "index.faiss"
        meta_path = directory / "metadata.jsonl"

        faiss.write_index(self.index, str(index_path))
        with meta_path.open("w", encoding="utf8") as handle:
            for text, meta in zip(self._texts, self._metadatas):
                record = {"text": text, "metadata": meta}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        """Load a FAISS index and metadata from disk."""

        index_path = directory / "index.faiss"
        meta_path = directory / "metadata.jsonl"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError("Missing index.faiss or metadata.jsonl")

        index = faiss.read_index(str(index_path))
        store = cls(dimension=index.d)
        store.index = index

        with meta_path.open("r", encoding="utf8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                store._texts.append(record["text"])
                store._metadatas.append(record["metadata"])

        return store

    def get_samples(self, max_samples: int = 5) -> list[SearchResult]:
        """Return a small sample of stored items for introspection."""

        return [SearchResult(text=text, metadata=meta, score=0.0) for text, meta in zip(self._texts[:max_samples], self._metadatas[:max_samples])]

    def get_all(self, batch_size: int = 256) -> list[SearchResult]:
        """Return all stored items for rebuilding secondary indexes."""

        del batch_size
        return [SearchResult(text=text, metadata=meta, score=0.0) for text, meta in zip(self._texts, self._metadatas)]

    def count(self) -> int:
        """Return the number of items in the store."""

        return len(self._texts)

    def clear(self) -> None:
        """Remove all vectors and metadata from the in-memory store."""

        self.index = faiss.IndexFlatIP(self.dimension)
        self._texts.clear()
        self._metadatas.clear()

    @staticmethod
    def _normalize(vectors: np.ndarray) -> None:
        """Normalize vectors in-place for cosine similarity."""

        faiss.normalize_L2(vectors)


def create_vector_store(backend: str | None = None, dimension: int | None = None) -> VectorStoreProtocol:
    """Create a vector store instance for the configured backend."""

    backend_name = (backend or DEFAULT_VECTOR_BACKEND).strip().lower()
    if backend_name == "qdrant":
        if dimension is None:
            dimension = DEFAULT_QDRANT_DIMENSION
        return QdrantVectorStore(
            collection_name=DEFAULT_QDRANT_COLLECTION,
            dimension=dimension,
        )
    return VectorStore(dimension=dimension or DEFAULT_QDRANT_DIMENSION)


def load_vector_store(store_dir: Path, backend: str | None = None) -> VectorStoreProtocol:
    """Load a persisted vector store from disk or Qdrant depending on configuration."""

    backend_name = (backend or DEFAULT_VECTOR_BACKEND).strip().lower()

    qdrant_marker = store_dir / "qdrant.enabled"
    if qdrant_marker.exists() or backend_name == "qdrant":
        return QdrantVectorStore(
            collection_name=DEFAULT_QDRANT_COLLECTION,
            dimension=DEFAULT_QDRANT_DIMENSION,
        )

    return VectorStore.load(store_dir)
