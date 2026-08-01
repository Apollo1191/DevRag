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
        key = f"{meta.get('relative_path')}:{meta.get('start_line')}"
        if key not in fused:
            fused[key] = {"text": text, "meta": meta, "vector_score": score, "bm25_score": 0.0, "rrf": 0.0}
        fused[key]["rrf"] += 1.0 / (k + rank)

    for rank, (text, meta, score) in enumerate(bm25_results, start=1):
        key = f"{meta.get('relative_path')}:{meta.get('start_line')}"
        if key not in fused:
            fused[key] = {"text": text, "meta": meta, "vector_score": 0.0, "bm25_score": score, "rrf": 0.0}
        else:
            fused[key]["bm25_score"] = score
        fused[key]["rrf"] += 1.0 / (k + rank)

    # Sort by fused score and return
    sorted_results = sorted(fused.items(), key=lambda x: x[1]["rrf"], reverse=True)
    return [(item["text"], item["meta"], item["vector_score"], item["bm25_score"]) for _, item in sorted_results]


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

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, dict, float]]:
        """Search for top-k documents by BM25 score."""

        if self.retriever is None or not self.corpus:
            return []

        query_tokens = bm25s.tokenize(query)
        results, scores = self.retriever.retrieve(query_tokens, corpus=self.corpus, k=top_k)

        output = []
        for result_list, score_list in zip(results, scores):
            for doc_idx, score in zip(result_list, score_list):
                try:
                    index = int(doc_idx)
                except (TypeError, ValueError):
                    continue
                if 0 <= index < len(self.corpus):
                    numeric_score = float(score)
                    if numeric_score > 0:
                        output.append((self.corpus[index], self.corpus_metadata[index], numeric_score))
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

        # Sort by rerank score and return top-k
        reranked.sort(key=lambda x: x[2], reverse=True)
        return reranked[:top_k]
