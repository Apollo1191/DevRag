"""API request/response schemas."""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    repo_url: str
    allowed_suffixes: Optional[List[str]] = None


class IngestLocalRequest(BaseModel):
    repo_path: str
    allowed_suffixes: Optional[List[str]] = None


class IngestResponse(BaseModel):
    total_files: int
    total_chunks: int
    total_embeddings: int


class IngestAsyncResponse(BaseModel):
    job_id: str
    message: str


class ClearStorageResponse(BaseModel):
    message: str
    deleted_storage: bool


class IngestJobStatus(BaseModel):
    status: str
    message: str | None = None
    stage: str | None = None
    total_files: int | None = None
    total_chunks: int | None = None
    total_embeddings: int | None = None
    error: str | None = None


class IndexSampleItem(BaseModel):
    relative_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None


class IndexItem(BaseModel):
    store_path: str
    total_chunks: int
    sample: list[IndexSampleItem]
    label: str | None = None
    topics: list[str] = Field(default_factory=list)


class IndexInfoResponse(BaseModel):
    stores: list[IndexItem]


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    use_llm: bool = True
    repository: str | None = None


class ChunkResult(BaseModel):
    text: str
    metadata: dict
    score: float


class QueryResponse(BaseModel):
    answer: Optional[str]
    chunks: List[ChunkResult]
