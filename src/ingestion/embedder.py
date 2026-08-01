"""Embedding clients and provider factory.

This module provides an async interface for generating embeddings and supports
multiple providers. The default provider is a local sentence-transformers
model so the project can run with a free setup.
"""

from __future__ import annotations

import logging
import os
from asyncio import to_thread
from dataclasses import dataclass
from typing import Iterable, List, Protocol

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding clients.

    Attributes:
        model: Embedding model identifier.
        api_key_env: Environment variable name for the API key.
        batch_size: Batch size used for local embedding inference.
    """

    model: str = "all-MiniLM-L6-v2"
    api_key_env: str = "OPENAI_API_KEY"
    batch_size: int = 32


class EmbedderProtocol(Protocol):
    """Protocol for embedders used by the ingestion pipeline."""

    async def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""


class LocalEmbedder:
    """Local sentence-transformers embedder.

    This embedder does not require an external API key and is suitable for the
    project's free-tier setup.
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Initialize the local embedding model.

        Args:
            config: Optional embedding configuration.

        Raises:
            RuntimeError: If sentence-transformers cannot be loaded.
        """

        self.config = config or EmbeddingConfig()
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.config.model)
        except Exception as exc:
            logger.exception("Failed to initialize local embedding model")
            raise RuntimeError("Failed to initialize local embedding model") from exc

    async def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Generate embeddings locally for a batch of texts."""

        text_list = list(texts)
        if not text_list:
            return []

        try:
            vectors = await to_thread(
                self.model.encode,
                text_list,
                batch_size=self.config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.exception("Local embedding request failed")
            raise RuntimeError("Local embedding request failed") from exc

        return vectors.tolist()


class OpenAIEmbedder:
    """Async embedding client with minimal configuration and error handling."""

    def __init__(self, config: EmbeddingConfig | None = None, api_key: str | None = None) -> None:
        """Create an embedder using an optional config and API key.

        Args:
            config: Optional embedding configuration.
            api_key: Optional API key override (otherwise read from env).

        Raises:
            ValueError: If no API key can be found.
        """

        self.config = config or EmbeddingConfig(model="text-embedding-3-small")
        resolved_key = api_key or os.getenv(self.config.api_key_env)
        if not resolved_key:
            raise ValueError(
                f"Missing API key. Set {self.config.api_key_env} or pass api_key explicitly."
            )

        self.client = AsyncOpenAI(api_key=resolved_key)

    async def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: Iterable of input strings.

        Returns:
            A list of embedding vectors aligned with the input order.

        Raises:
            RuntimeError: If the OpenAI API call fails.
        """

        text_list = list(texts)
        if not text_list:
            return []

        try:
            response = await self.client.embeddings.create(
                model=self.config.model,
                input=text_list,
            )
        except Exception as exc:
            logger.exception("Embedding request failed")
            raise RuntimeError("Embedding request failed") from exc

        return [item.embedding for item in response.data]


def create_embedder(provider: str | None = None) -> EmbedderProtocol:
    """Create an embedder from a provider name.

    Args:
        provider: Provider name (`local` or `openai`). If omitted, reads
            `DEVRAG_EMBEDDING_PROVIDER` and defaults to `local`.

    Returns:
        An object implementing the embedding protocol.

    Raises:
        ValueError: If the provider name is unsupported.
    """

    resolved_provider = (provider or os.getenv("DEVRAG_EMBEDDING_PROVIDER", "local")).strip().lower()
    if resolved_provider == "local":
        return LocalEmbedder()
    if resolved_provider == "openai":
        return OpenAIEmbedder()
    raise ValueError("Unsupported embedding provider. Use 'local' or 'openai'.")
