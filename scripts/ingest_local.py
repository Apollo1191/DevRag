from __future__ import annotations

import asyncio
from pathlib import Path
from dotenv import load_dotenv

from src.ingestion.embedder import create_embedder
from src.ingestion.pipeline import IngestionPipeline

load_dotenv(Path(".env"))

async def main():
    embedder = create_embedder()
    pipeline = IngestionPipeline(workspace_root=Path("./workspaces"), embedder=embedder)
    repo_path = Path("./workspaces/fastapi")
    if not repo_path.exists():
        print("Local path ./workspaces/fastapi does not exist. Exiting.")
        return
    print("Starting ingest of local workspace: ./workspaces/fastapi")
    report = await pipeline.ingest_local_path(repo_path)
    print(f"Ingest finished: files={report.total_files}, chunks={report.total_chunks}, embeddings={report.total_embeddings}")

if __name__ == '__main__':
    asyncio.run(main())
