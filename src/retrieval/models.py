"""Shared retrieval data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchResult:
    """Single search result from a vector store."""

    text: str
    metadata: dict
    score: float
