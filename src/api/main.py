"""FastAPI app for ingestion and query endpoints."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
import re
from collections import Counter
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from ..ingestion.embedder import EmbedderProtocol, create_embedder
from ..ingestion.pipeline import IngestionPipeline
from ..generation.llm_client import LLMClient
from ..generation.prompt_builder import build_prompt, build_retrieval_fallback, load_prompt_template
from ..retrieval.vector_store import VectorStoreProtocol, load_vector_store
from ..retrieval.hybrid_search import RerankerWrapper, reciprocal_rank_fusion
import asyncio
import uuid
from pathlib import Path

from .schemas import (
    ChunkResult,
    IngestAsyncResponse,
    ClearStorageResponse,
    IngestLocalRequest,
    IngestRequest,
    IngestResponse,
    IndexInfoResponse,
    IndexItem,
    IndexSampleItem,
    IngestJobStatus,
    QueryRequest,
    QueryResponse,
)

app = FastAPI(title="DevRag")
logger = logging.getLogger(__name__)

# Load environment variables from .env in the project root.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

cors_origins = os.getenv("DEVRAG_CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins if origin.strip()],
    allow_methods=["*"] ,
    allow_headers=["*"] ,
)

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "v1.yaml"

_embedder: EmbedderProtocol | None = None
_pipeline: IngestionPipeline | None = None
_llm: LLMClient | None = None
_store: VectorStoreProtocol | None = None
_reranker: RerankerWrapper | None = None
_jobs: dict[str, asyncio.Task] = {}
_job_results: dict[str, dict] = {}
_MIN_VECTOR_RELEVANCE = float(os.getenv("DEVRAG_MIN_VECTOR_RELEVANCE", "0.55"))
_CASUAL_MESSAGES = {
    "hi",
    "hello",
    "hey",
    "สวัสดี",
    "หวัดดี",
    "ขอบคุณ",
    "thanks",
    "thank you",
    "bye",
    "ลาก่อน",
}


def _is_casual_message(question: str) -> bool:
    """Recognize short greetings, thanks, and goodbyes without retrieval."""

    normalized = re.sub(r"[^\w\u0E00-\u0E7F ]+", "", question.lower()).strip()
    return normalized in _CASUAL_MESSAGES or any(
        normalized.startswith(f"{prefix} ")
        for prefix in _CASUAL_MESSAGES
        if len(prefix) >= 4
    )


_TOPIC_STOPWORDS = {
    "docs",
    "doc",
    "tutorial",
    "tests",
    "test",
    "src",
    "readme",
    "example",
    "examples",
    "advanced",
    "basic",
    "guide",
    "zh",
    "zh-hant",
    "zh_hant",
    "en",
    "fr",
    "de",
    "ja",
    "pt",
    "py",
    "md",
    "api",
    "docs_src",
}

def _get_embedder() -> EmbedderProtocol:
    global _embedder
    if _embedder is None:
        _embedder = create_embedder()
    return _embedder


def _get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        store_dir = Path("./storage")
        _pipeline = IngestionPipeline(
            workspace_root=Path("./workspaces"),
            embedder=_get_embedder(),
            store_dir=store_dir,
        )
    return _pipeline


def _get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


def _get_reranker() -> RerankerWrapper:
    global _reranker
    if _reranker is None:
        _reranker = RerankerWrapper()
    return _reranker


def _extract_topics(texts: list[str], metadatas: list[dict], limit: int = 5) -> list[str]:
    """Derive human-friendly topic keywords from chunk metadata and sample text."""

    counter: Counter[str] = Counter()

    for meta, text in zip(metadatas, texts):
        relative_path = str(meta.get("relative_path") or "")
        heading = str(meta.get("heading") or "")

        candidates = []
        candidates.extend(re.split(r"[\\/._\-]+", relative_path))
        candidates.extend(re.split(r"[\\/._\-]+", heading))

        first_line = text.splitlines()[0] if text else ""
        candidates.extend(re.findall(r"[A-Za-z][A-Za-z0-9_]+", first_line))

        for token in candidates:
            normalized = token.strip().lower()
            if len(normalized) < 3:
                continue
            if normalized in _TOPIC_STOPWORDS:
                continue
            counter[normalized] += 1

    topics = [topic for topic, _ in counter.most_common(limit)]
    return topics


def _summarize_repo_names(metadatas: list[dict], limit: int = 5) -> list[str]:
    """Return the most common source repo names in a store."""

    counter: Counter[str] = Counter()
    for meta in metadatas:
        repo_name = str(meta.get("source_repo") or "").strip()
        if repo_name:
            counter[repo_name] += 1

    return [repo for repo, _ in counter.most_common(limit)]


def _remove_tree(path: Path) -> bool:
    """Remove directory contents while preserving bind-mounted root folders."""

    if not path.exists():
        return False

    def _handle_remove_readonly(func, target_path, exc_info):
        del exc_info
        os.chmod(target_path, 0o666)
        func(target_path)

    removed = False
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, onerror=_handle_remove_readonly)
        else:
            try:
                child.unlink()
            except PermissionError:
                os.chmod(child, 0o666)
                child.unlink()
        removed = True
    return removed


@app.post("/storage/clear", response_model=ClearStorageResponse)
async def clear_storage() -> ClearStorageResponse:
    """Delete the persisted vector store and reset in-memory index state."""

    global _pipeline

    removed = _remove_tree(Path("./storage"))
    if _pipeline is not None:
        _pipeline.clear()
    elif os.getenv("DEVRAG_VECTOR_BACKEND", "faiss").strip().lower() == "qdrant":
        qdrant_store = load_vector_store(Path("./storage"), backend="qdrant")
        had_points = qdrant_store.count() > 0
        qdrant_store.clear()
        removed = removed or had_points

    message = "Storage cleared" if removed else "No persisted storage found"
    return ClearStorageResponse(message=message, deleted_storage=removed)

@app.post("/ingest", response_model=IngestResponse)
async def ingest_repo(payload: IngestRequest) -> IngestResponse:
    # Run synchronous ingest directly (keeps compatibility). For large repos,
    # prefer /ingest_async which runs ingestion in background and returns a job id.
    pipeline = _get_pipeline()
    report = await pipeline.ingest_repo(payload.repo_url, allowed_suffixes=payload.allowed_suffixes)
    return IngestResponse(
        total_files=report.total_files,
        total_chunks=report.total_chunks,
        total_embeddings=report.total_embeddings,
    )


@app.post("/ingest_async", response_model=IngestAsyncResponse)
async def ingest_repo_async(payload: IngestRequest) -> IngestAsyncResponse:
    """Start an ingestion job in the background and return a job id."""

    pipeline = _get_pipeline()
    job_id = str(uuid.uuid4())

    def _set_job(stage: str, message: str | None = None, **extra: object) -> None:
        current = _job_results.get(job_id, {})
        current.update({"status": "running", "stage": stage})
        if message is not None:
            current["message"] = message
        current.update(extra)
        _job_results[job_id] = current

    async def _run_and_store():
        try:
            report = await pipeline.ingest_repo(
                payload.repo_url,
                allowed_suffixes=payload.allowed_suffixes,
                progress_callback=lambda stage, data: _set_job(
                    stage,
                    data.get("message"),
                    **{key: value for key, value in data.items() if key != "message"},
                ),
            )
            _job_results[job_id] = {
                "status": "completed",
                "stage": "done",
                "message": "Ingestion complete",
                "total_files": report.total_files,
                "total_chunks": report.total_chunks,
                "total_embeddings": report.total_embeddings,
            }
        except Exception as exc:
            _job_results[job_id] = {"status": "failed", "stage": "failed", "error": str(exc), "message": "Ingestion failed"}

    task = asyncio.create_task(_run_and_store())
    _jobs[job_id] = task
    _job_results[job_id] = {"status": "running", "stage": "queued", "message": "Ingestion queued"}
    return IngestAsyncResponse(job_id=job_id, message="Ingestion started")


@app.post("/ingest_local_async", response_model=IngestAsyncResponse)
async def ingest_local_async(payload: IngestLocalRequest) -> IngestAsyncResponse:
    """Start an ingestion job for a local repository path."""

    pipeline = _get_pipeline()
    job_id = str(uuid.uuid4())

    def _set_job(stage: str, message: str | None = None, **extra: object) -> None:
        current = _job_results.get(job_id, {})
        current.update({"status": "running", "stage": stage})
        if message is not None:
            current["message"] = message
        current.update(extra)
        _job_results[job_id] = current

    async def _run_and_store():
        try:
            report = await pipeline.ingest_local_path(
                Path(payload.repo_path),
                allowed_suffixes=payload.allowed_suffixes,
                progress_callback=lambda stage, data: _set_job(
                    stage,
                    data.get("message"),
                    **{key: value for key, value in data.items() if key != "message"},
                ),
            )
            _job_results[job_id] = {
                "status": "completed",
                "stage": "done",
                "message": "Ingestion complete",
                "total_files": report.total_files,
                "total_chunks": report.total_chunks,
                "total_embeddings": report.total_embeddings,
            }
        except Exception as exc:
            _job_results[job_id] = {"status": "failed", "stage": "failed", "error": str(exc), "message": "Ingestion failed"}

    task = asyncio.create_task(_run_and_store())
    _jobs[job_id] = task
    _job_results[job_id] = {"status": "running", "stage": "queued", "message": "Ingestion queued"}
    return IngestAsyncResponse(job_id=job_id, message="Ingestion started")


@app.get("/ingest_status/{job_id}", response_model=IngestJobStatus)
async def ingest_status(job_id: str):
    """Return the status of a background ingestion job."""
    if job_id not in _job_results:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_results[job_id]


@app.get("/index_info", response_model=IndexInfoResponse)
async def index_info():
    """Return information about available vector stores.

    This checks the configured storage folder and any in-memory store created
    by the pipeline and returns counts and a small sample of source metadata.
    """
    stores: list[IndexItem] = []

    storage_dir = Path("./storage")
    # If persisted index exists on disk, load summary from disk or Qdrant.
    if storage_dir.exists():
        try:
            disk_store = load_vector_store(storage_dir)
            all_results = disk_store.get_all()
            grouped: dict[str, list] = {}
            for result in all_results:
                repo = str(result.metadata.get("source_repo") or "unknown")
                grouped.setdefault(repo, []).append(result)
            for repo, repo_results in sorted(grouped.items()):
                sample = [
                    IndexSampleItem(
                        relative_path=result.metadata.get("relative_path"),
                        start_line=result.metadata.get("start_line"),
                        end_line=result.metadata.get("end_line"),
                    )
                    for result in repo_results[:5]
                ]
                stores.append(
                    IndexItem(
                        store_path=f"{storage_dir}/{repo}",
                        total_chunks=len(repo_results),
                        sample=sample,
                        label=repo,
                        topics=[repo],
                    )
                )
        except Exception:
            # ignore load errors and continue
            pass

    # Only fall back to in-memory store if we don't already have a persisted one.
    if not stores and _pipeline is not None and _pipeline.store is not None:
        mem = _pipeline.store
        all_results = mem.get_all()
        grouped: dict[str, list] = {}
        for result in all_results:
            repo = str(result.metadata.get("source_repo") or "unknown")
            grouped.setdefault(repo, []).append(result)
        for repo, repo_results in sorted(grouped.items()):
            sample = [
                IndexSampleItem(
                    relative_path=result.metadata.get("relative_path"),
                    start_line=result.metadata.get("start_line"),
                    end_line=result.metadata.get("end_line"),
                )
                for result in repo_results[:5]
            ]
            stores.append(
                IndexItem(
                    store_path=f"memory/{repo}",
                    total_chunks=len(repo_results),
                    sample=sample,
                    label=repo,
                    topics=[repo],
                )
            )

    return IndexInfoResponse(stores=stores)


@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest) -> QueryResponse:
    normalized_question = " ".join(payload.question.strip().lower().split())
    if _is_casual_message(normalized_question):
        return QueryResponse(
            answer=(
                "สวัสดีครับ ผมช่วยตอบคำถามเกี่ยวกับโค้ดและเอกสารที่อยู่ใน repository "
                "ที่ทำดัชนีไว้ได้ครับ มีอะไรให้ช่วยค้นหาไหม?"
            ),
            chunks=[],
        )

    pipeline = _get_pipeline()
    if pipeline.store is None:
        raise HTTPException(status_code=400, detail="No data indexed. Call /ingest first.")

    embedder = _get_embedder()
    query_embedding = (await embedder.embed_texts([payload.question]))[0]
    
    # Perform hybrid search: vector + BM25
    vector_results = pipeline.store.search(query_embedding, top_k=payload.top_k * 4)
    bm25_results = pipeline.bm25_store.search(payload.question, top_k=payload.top_k * 4)
    
    # Convert to tuples for RRF
    vector_tuples = [(r.text, r.metadata, r.score) for r in vector_results]
    bm25_tuples = [(text, meta, score) for text, meta, score in bm25_results]
    
    # Fuse results using RRF
    fused = reciprocal_rank_fusion(vector_tuples, bm25_tuples, k=60)

    # Do not send weak semantic matches to the LLM. Exact BM25 matches remain
    # eligible, while unrelated vector-only results are rejected.
    relevant = [
        item
        for item in fused
        if item[2] >= _MIN_VECTOR_RELEVANCE or item[3] > 0
    ]
    
    # Rerank fused results
    reranker = _get_reranker()
    reranked = reranker.rerank(payload.question, relevant, top_k=payload.top_k)
    
    # Convert to ChunkResult
    from ..retrieval.vector_store import SearchResult
    results = [SearchResult(text=text, metadata=meta, score=score) for text, meta, score in reranked]
    chunks = [ChunkResult(text=r.text, metadata=r.metadata, score=r.score) for r in results]

    answer = None
    if not results:
        answer = (
            "I could not find enough relevant information in the indexed sources "
            "to answer that reliably. ลองถามเกี่ยวกับเนื้อหาใน repository ที่ index ไว้ได้เลยครับ"
        )
    elif payload.use_llm:
        template = load_prompt_template(PROMPT_PATH)
        system_prompt, user_prompt = build_prompt(payload.question, results, template)
        try:
            answer = await _get_llm().generate(system_prompt, user_prompt)
        except Exception:
            # Never fail the retrieval path just because the LLM provider is unavailable.
            logger.warning("LLM generation failed; returning retrieved chunks only", exc_info=True)
            answer = None

    if answer is None and results:
        answer = build_retrieval_fallback(payload.question, results)

    return QueryResponse(answer=answer, chunks=chunks)
