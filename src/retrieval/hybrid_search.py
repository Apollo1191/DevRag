"""Hybrid search combining BM25 keyword search with vector similarity."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable

import bm25s
import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """Result from hybrid search."""

    text: str
    metadata: dict
    vector_score: float
    bm25_score: float
    fused_score: float


def reciprocal_rank_fusion(
    vector_results: list[tuple[str, dict, float]],
    bm25_results: list[tuple[str, dict, float]],
    k: int = 60,
) -> list[tuple[str, dict, float, float]]:
    """Fuse vector and BM25 results using Reciprocal Rank Fusion (RRF).

    Args:
        vector_results: List of (text, metadata, score) from vector search.
        bm25_results: List of (text, metadata, score) from BM25 search.
        k: RRF parameter (typically 60).

    Returns:
        List of (text, metadata, vector_score, bm25_score) sorted by fused score.
    """

    # Build RRF scores: 1 / (k + rank)
    fused: dict[str, dict] = {}

    for rank, (text, meta, score) in enumerate(vector_results, start=1):
        key = _result_key(text, meta)
        if key not in fused:
            fused[key] = {"text": text, "meta": meta, "vector_score": score, "bm25_score": 0.0, "rrf": 0.0}
        fused[key]["rrf"] += 1.0 / (k + rank)

    for rank, (text, meta, score) in enumerate(bm25_results, start=1):
        key = _result_key(text, meta)
        if key not in fused:
            fused[key] = {"text": text, "meta": meta, "vector_score": 0.0, "bm25_score": score, "rrf": 0.0}
        else:
            fused[key]["bm25_score"] = score
        fused[key]["rrf"] += 1.0 / (k + rank)

    # Sort by fused score and return
    sorted_results = sorted(fused.items(), key=lambda x: x[1]["rrf"], reverse=True)
    return [(item["text"], item["meta"], item["vector_score"], item["bm25_score"]) for _, item in sorted_results]


def _result_key(text: str, metadata: dict) -> str:
    """Build a stable identity for one indexed chunk across search backends."""

    return "|".join(
        str(metadata.get(field, ""))
        for field in ("source_repo", "relative_path", "start_line", "end_line")
    ) + f"|{text}"


class BM25Wrapper:
    """BM25 keyword search wrapper using bm25s."""

    def __init__(self) -> None:
        self.corpus: list[str] = []
        self.corpus_metadata: list[dict] = []
        self.retriever = None

    def add(self, texts: Iterable[str], metadatas: Iterable[dict]) -> None:
        """Add documents to BM25 index."""

        texts_list = list(texts)
        metadatas_list = list(metadatas)

        self.corpus.extend(texts_list)
        self.corpus_metadata.extend(metadatas_list)

        # Rebuild the full index whenever we add
        self.retriever = bm25s.BM25()
        self.retriever.index(bm25s.tokenize(self.corpus))

    def remove_repository(self, repository: str) -> None:
        """Remove all documents belonging to one repository and rebuild BM25."""

        kept = [
            (text, metadata)
            for text, metadata in zip(self.corpus, self.corpus_metadata)
            if metadata.get("source_repo") != repository
        ]
        self.corpus = [text for text, _ in kept]
        self.corpus_metadata = [metadata for _, metadata in kept]
        self.retriever = bm25s.BM25() if self.corpus else None
        if self.corpus:
            self.retriever.index(bm25s.tokenize(self.corpus))

    def search(self, query: str, top_k: int = 5, repository: str | None = None) -> list[tuple[str, dict, float]]:
        """Search for top-k documents by BM25 score."""

        if self.retriever is None or not self.corpus:
            return []

        query_tokens = bm25s.tokenize(query)
        # Filtering after a global top-k query can discard all matching chunks
        # for a repository. Over-fetch the full corpus when a repository filter
        # is requested, then return the best matching filtered documents.
        retrieve_k = len(self.corpus) if repository else top_k
        results, scores = self.retriever.retrieve(query_tokens, corpus=self.corpus, k=retrieve_k)

        output = []
        for result_list, score_list in zip(results, scores):
            for doc_idx, score in zip(result_list, score_list):
                try:
                    index = int(doc_idx)
                except (TypeError, ValueError):
                    continue
                if 0 <= index < len(self.corpus):
                    if repository and self.corpus_metadata[index].get("source_repo") != repository:
                        continue
                    numeric_score = float(score)
                    if numeric_score > 0:
                        output.append((self.corpus[index], self.corpus_metadata[index], numeric_score))
                        if not repository and len(output) >= top_k:
                            return output
                        if repository and len(output) >= top_k:
                            return output
        return output


class RerankerWrapper:
    """Cross-encoder based reranker for refining search results."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, results: Iterable[tuple[str, dict, float, float]], top_k: int = 5
    ) -> list[tuple[str, dict, float]]:
        """Rerank results using cross-encoder.

        Args:
            query: The search query.
            results: List of (text, metadata, vector_score, bm25_score).
            top_k: Number of top results to return.

        Returns:
            List of (text, metadata, rerank_score) sorted by rerank score.
        """

        results_list = list(results)
        if not results_list:
            return []

        # Prepare pairs for cross-encoder: (query, text)
        pairs = [(query, text) for text, _, _, _ in results_list]

        # Score pairs
        scores = self.model.predict(pairs)

        # Create output with rerank scores
        reranked = []
        for (text, meta, _, _), score in zip(results_list, scores):
            reranked.append((text, meta, float(score)))

        # Prefer evidence from distinct chunks and files. A single long file
        # can otherwise occupy the whole context with near-duplicate sections.
        reranked.sort(key=lambda x: x[2], reverse=True)
        selected: list[tuple[str, dict, float]] = []
        selected_keys: set[tuple[str, str, int | None, int | None]] = set()
        path_counts: dict[str, int] = {}
        max_chunks_per_path = 2

        for item in reranked:
            text, metadata, _ = item
            chunk_key = (
                str(metadata.get("source_repo", "")),
                str(metadata.get("relative_path", "")),
                metadata.get("start_line"),
                metadata.get("end_line"),
            )
            path_key = f"{chunk_key[0]}|{chunk_key[1]}"
            if chunk_key in selected_keys or path_counts.get(path_key, 0) >= max_chunks_per_path:
                continue
            selected.append(item)
            selected_keys.add(chunk_key)
            path_counts[path_key] = path_counts.get(path_key, 0) + 1
            if len(selected) >= top_k:
                break

        # If one file is the only useful evidence, fill remaining slots with
        # lower-ranked chunks after the diversity pass.
        if len(selected) < top_k:
            selected_keys = {
                (
                    str(metadata.get("source_repo", "")),
                    str(metadata.get("relative_path", "")),
                    metadata.get("start_line"),
                    metadata.get("end_line"),
                )
                for _, metadata, _ in selected
            }
            for item in reranked:
                text, metadata, _ = item
                chunk_key = (
                    str(metadata.get("source_repo", "")),
                    str(metadata.get("relative_path", "")),
                    metadata.get("start_line"),
                    metadata.get("end_line"),
                )
                if chunk_key in selected_keys:
                    continue
                selected.append(item)
                selected_keys.add(chunk_key)
                if len(selected) >= top_k:
                    break

        return selected
