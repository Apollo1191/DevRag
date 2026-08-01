"""Ingestion pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .chunker import chunk_text_by_language
from .embedder import EmbedderProtocol
from .loader import RepositoryLoader
from ..retrieval.vector_store import create_vector_store, load_vector_store, VectorStoreProtocol
from ..retrieval.hybrid_search import BM25Wrapper


@dataclass
class IngestionReport:
    """Summary of ingestion results."""

    total_files: int
    total_chunks: int
    total_embeddings: int


class IngestionPipeline:
    """Orchestrates repo loading, chunking, embedding, and indexing."""

    def __init__(
        self,
        workspace_root: Path,
        embedder: EmbedderProtocol,
        store: VectorStoreProtocol | None = None,
        store_dir: Path | None = None,
    ) -> None:
        self.loader = RepositoryLoader(workspace_root=workspace_root)
        self.embedder = embedder
        self.store = store
        self.store_dir = store_dir
        self.bm25_store = BM25Wrapper()

        if self.store is None and self.store_dir:
            try:
                self.store = load_vector_store(self.store_dir)
            except FileNotFoundError:
                self.store = None

        if self.store is not None:
            stored_chunks = self.store.get_all()
            if stored_chunks:
                self.bm25_store.add(
                    [chunk.text for chunk in stored_chunks],
                    [chunk.metadata for chunk in stored_chunks],
                )

    def clear(self) -> None:
        """Clear both the primary vector store and the secondary BM25 index."""

        if self.store is not None:
            self.store.clear()
        self.store = None
        self.bm25_store = BM25Wrapper()

    async def ingest_repo(
        self,
        repo_url: str,
        allowed_suffixes: Iterable[str] | None = None,
        progress_callback: Callable[[str, dict], None] | None = None,
    ) -> IngestionReport:
        """Clone and ingest a repository into the vector store."""

        if progress_callback is not None:
            progress_callback("cloning", {"message": "Cloning repository"})
        repo_path = self.loader.clone_repository(repo_url)
        return await self.ingest_local_path(
            repo_path,
            allowed_suffixes=allowed_suffixes,
            repo_name=repo_path.name,
            progress_callback=progress_callback,
        )

    async def ingest_local_path(
        self,
        repo_path: Path,
        allowed_suffixes: Iterable[str] | None = None,
        repo_name: str | None = None,
        progress_callback: Callable[[str, dict], None] | None = None,
    ) -> IngestionReport:
        """Ingest an already-cloned repository path."""

        if progress_callback is not None:
            progress_callback("walking", {"message": "Scanning source files"})
        source_files = self.loader.walk_source_files(repo_path, allowed_suffixes=allowed_suffixes)
        if progress_callback is not None:
            progress_callback("chunking", {"message": f"Chunking {len(source_files)} files", "total_files": len(source_files)})
        chunks = []
        for source in source_files:
            text = source.path.read_text(encoding="utf8", errors="ignore")
            chunks.extend(chunk_text_by_language(text, source.relative_path))

        texts = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "relative_path": chunk.metadata.get("relative_path"),
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "source_repo": repo_name or repo_path.name,
            }
            for chunk in chunks
        ]

        total_embeddings = 0
        if texts:
            if progress_callback is not None:
                progress_callback("embedding", {"message": f"Embedding {len(texts)} chunks", "total_chunks": len(texts)})
            try:
                embeddings = await self.embedder.embed_texts(texts)
                total_embeddings = len(embeddings)
            except Exception:
                embeddings = []

            if embeddings and self.store is None:
                self.store = create_vector_store(dimension=len(embeddings[0]))
            if self.store is not None and embeddings:
                if progress_callback is not None:
                    progress_callback("indexing", {"message": "Writing vectors to index"})
                self.store.add(embeddings, texts, metadatas)
                self.bm25_store.add(texts, metadatas)
                if self.store_dir is not None:
                    if progress_callback is not None:
                        progress_callback("saving", {"message": "Saving index to disk"})
                    self.store.save(self.store_dir)

        if progress_callback is not None:
            progress_callback(
                "done",
                {
                    "message": "Ingest complete",
                    "total_files": len(source_files),
                    "total_chunks": len(chunks),
                    "total_embeddings": total_embeddings,
                },
            )

        return IngestionReport(
            total_files=len(source_files),
            total_chunks=len(chunks),
            total_embeddings=total_embeddings,
        )