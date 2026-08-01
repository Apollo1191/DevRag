"""Local CLI to ingest a repo and run a query."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from src.ingestion.embedder import create_embedder
from src.ingestion.pipeline import IngestionPipeline
from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import build_prompt, load_prompt_template

load_dotenv(Path(".env"))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_url", help="Repository URL to ingest")
    parser.add_argument("question", help="Question to query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    embedder = create_embedder()
    pipeline = IngestionPipeline(workspace_root=Path("./workspaces"), embedder=embedder)
    report = await pipeline.ingest_repo(args.repo_url)
    print(f"Ingested: {report.total_files} files, {report.total_chunks} chunks")

    query_embedding = (await embedder.embed_texts([args.question]))[0]
    results = pipeline.store.search(query_embedding, top_k=args.top_k)

    prompt_path = Path("./prompts/v1.yaml")
    template = load_prompt_template(prompt_path)
    system_prompt, user_prompt = build_prompt(args.question, results, template)

    llm = LLMClient()
    answer = await llm.generate(system_prompt, user_prompt)

    print("\nAnswer:\n")
    print(answer or "(No answer: missing API key)")
    print("\nTop sources:\n")
    for idx, result in enumerate(results, start=1):
        meta = result.metadata
        print(f"[{idx}] {meta.get('relative_path')}:{meta.get('start_line')}-{meta.get('end_line')}")


if __name__ == "__main__":
    asyncio.run(main())
