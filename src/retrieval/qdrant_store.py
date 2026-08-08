"""Qdrant-backed vector store implementation."""

from __future__ import annotations

import logging
import os
import uuid
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from .models import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class QdrantVectorStore:
    """Vector store implementation backed by a Qdrant collection."""

    collection_name: str
    dimension: int
    url: str | None = None
    api_key: str | None = None
    client: QdrantClient | None = None

    def __post_init__(self) -> None:
        """Initialize the Qdrant client and ensure the collection exists."""

        self.url = self.url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = self.api_key or os.getenv("QDRANT_API_KEY")

        self.client = QdrantClient(url=self.url, api_key=self.api_key)

        try:
            # Check if the collection already exists.
            collections = self.client.get_collections().collections
            if self.collection_name not in [collection.name for collection in collections]:
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
                )
        except Exception as exc:
            logger.exception("Failed to initialize Qdrant collection %s", self.collection_name)
            raise

    def add(self, embeddings: Iterable[Iterable[float]], texts: Iterable[str], metadatas: Iterable[dict]) -> None:
        """Upsert vectors and payloads into Qdrant."""

        vectors = list(embeddings)
        texts_list = list(texts)
        metadatas_list = list(metadatas)

        if len(vectors) != len(texts_list) or len(vectors) != len(metadatas_list):
            raise ValueError("Embeddings, texts, and metadata must have the same length")

        points: list[PointStruct] = []
        for vector, text, metadata in zip(vectors, texts_list, metadatas_list):
            stable_key = "|".join(
                [
                    str(metadata.get("source_repo", "")),
                    str(metadata.get("relative_path", "")),
                    str(metadata.get("start_line", "")),
                    str(metadata.get("end_line", "")),
                    text,
                ]
            )
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, hashlib.sha256(stable_key.encode("utf8")).hexdigest()))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=list(vector),
                    payload={"text": text, "metadata": metadata},
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, embedding: Iterable[float], top_k: int = 5, repository: str | None = None) -> List[SearchResult]:
        """Search Qdrant and return top-k matching chunks."""

        vector = list(embedding)
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=top_k,
            query_filter=self._repository_filter(repository),
            with_payload=True,
        )

        results: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            text = payload.get("text", "")
            metadata = payload.get("metadata", {})
            results.append(SearchResult(text=text, metadata=metadata, score=float(hit.score or 0.0)))
        return results

    @staticmethod
    def _repository_filter(repository: str | None) -> Filter | None:
        if not repository:
            return None
        return Filter(must=[FieldCondition(key="metadata.source_repo", match=MatchValue(value=repository))])

    def save(self, directory: Path) -> None:
        """Qdrant persists data internally, so this is a no-op."""

        directory.mkdir(parents=True, exist_ok=True)
        marker_file = directory / "qdrant.enabled"
        marker_file.write_text(f"collection={self.collection_name}\n", encoding="utf8")

    def clear(self) -> None:
        """Delete all points while preserving the configured collection."""

        self.client.delete_collection(collection_name=self.collection_name)
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
        )

    def clear_repository(self, repository: str) -> None:
        """Delete all points belonging to one repository."""

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="metadata.source_repo", match=MatchValue(value=repository))]
            ),
        )

    def get_samples(self, max_samples: int = 5) -> list[SearchResult]:
        """Return a small sample of stored points for introspection."""

        results: list["SearchResult"] = []
        points, _ = self.client.scroll(collection_name=self.collection_name, limit=max_samples, with_payload=True)
        for point in points:
            payload = point.payload or {}
            text = payload.get("text", "")
            metadata = payload.get("metadata", {})
            results.append(SearchResult(text=text, metadata=metadata, score=0.0))
        return results

    def get_all(self, batch_size: int = 256) -> list[SearchResult]:
        """Load all stored chunks for rebuilding secondary indexes."""

        results: list[SearchResult] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
            )
            for point in points:
                payload = point.payload or {}
                results.append(
                    SearchResult(
                        text=payload.get("text", ""),
                        metadata=payload.get("metadata", {}),
                        score=0.0,
                    )
                )
            if offset is None:
                break
        return results

    def count(self) -> int:
        """Return the number of points in the collection."""

        info = self.client.get_collection(self.collection_name)
        return int(info.points_count or 0)
