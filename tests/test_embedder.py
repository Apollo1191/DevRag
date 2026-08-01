import os
import pytest

from src.ingestion.embedder import EmbeddingConfig, OpenAIEmbedder


def test_embedder_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenAIEmbedder()


def test_embedder_accepts_env_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    embedder = OpenAIEmbedder(EmbeddingConfig(model="text-embedding-3-small"))
    assert embedder.config.model == "text-embedding-3-small"
