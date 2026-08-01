"""Run a low-cost, multi-phase RAG benchmark.

Retrieval and behavior checks are free. Generation checks are opt-in because
`--with-llm` calls the configured provider for every case.

Run from the repository root while the backend is running:
    python -m eval.run_rag_benchmark
    python -m eval.run_rag_benchmark --with-llm
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


DEFAULT_DATASET = Path(__file__).with_name("rag_cases.jsonl")
DEFAULT_REPORT = Path(__file__).with_name("rag_report.json")


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load and validate benchmark cases."""

    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf8").splitlines(), start=1):
        if not line.strip():
            continue
        case = json.loads(line)
        required = ("id", "category", "language", "question", "relevant_paths")
        missing = [field for field in required if field not in case]
        if missing:
            raise ValueError(f"Line {line_number} is missing: {', '.join(missing)}")
        cases.append(case)
    return cases


def path_matches(actual: str, expected: str) -> bool:
    """Match normalized file paths and directory prefixes."""

    actual_path = actual.replace("\\", "/").lower()
    expected_path = expected.replace("\\", "/").lower()
    return actual_path == expected_path or actual_path.startswith(expected_path.rstrip("/") + "/")


def evaluate_case(client: httpx.Client, base_url: str, case: dict[str, Any], top_k: int, with_llm: bool) -> dict[str, Any]:
    """Evaluate retrieval, behavior, and optionally answer generation."""

    started = time.perf_counter()
    response = client.post(
        f"{base_url.rstrip('/')}/query",
        json={"question": case["question"], "top_k": top_k, "use_llm": with_llm},
    )
    latency_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    body = response.json()
    chunks = body.get("chunks", [])
    expected_paths = case["relevant_paths"]

    matched_ranks: list[int] = []
    retrieved_paths: list[str] = []
    for rank, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        actual_path = str(metadata.get("relative_path") or "")
        source_repo = str(metadata.get("source_repo") or "")
        if actual_path:
            retrieved_paths.append(actual_path)
        if expected_paths and source_repo == case.get("repo") and any(path_matches(actual_path, expected) for expected in expected_paths):
            matched_ranks.append(rank)

    answer = str(body.get("answer") or "")
    citation_count = answer.count("[Source ")
    is_ood = case["category"] in {"casual", "out_of_domain"}
    behavior_pass = not is_ood or not chunks
    return {
        "id": case["id"],
        "category": case["category"],
        "language": case["language"],
        "repo": case.get("repo"),
        "question": case["question"],
        "retrieved_paths": retrieved_paths,
        "matched_ranks": matched_ranks,
        "hit_at_k": bool(matched_ranks),
        "recall_at_k": min(len(set(matched_ranks)) / len(expected_paths), 1.0) if expected_paths else None,
        "mrr": 1.0 / matched_ranks[0] if matched_ranks else 0.0,
        "behavior_pass": behavior_pass,
        "source_count": len(chunks),
        "answer_present": bool(answer),
        "citation_count": citation_count,
        "latency_ms": round(latency_ms, 2),
        "answer_preview": answer[:300] if with_llm else None,
    }


def mean(rows: list[dict[str, Any]], field: str) -> float:
    """Average a numeric or boolean field."""

    values = [float(row[field]) for row in rows if row[field] is not None]
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict[str, Any]], top_k: int, with_llm: bool) -> dict[str, Any]:
    """Build phase and per-category summaries."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)

    retrieval_rows = [row for row in rows if row["category"] == "in_domain"]
    behavior_rows = [row for row in rows if row["category"] != "in_domain"]
    latency_values = [row["latency_ms"] for row in rows]
    summary: dict[str, Any] = {
        "top_k": top_k,
        "with_llm": with_llm,
        "phases": {
            "retrieval": {
                "questions": len(retrieval_rows),
                f"hit_at_{top_k}": mean(retrieval_rows, "hit_at_k"),
                f"recall_at_{top_k}": mean(retrieval_rows, "recall_at_k"),
                "mrr": mean(retrieval_rows, "mrr"),
            },
            "behavior": {
                "questions": len(behavior_rows),
                "no_irrelevant_sources_rate": mean(behavior_rows, "behavior_pass"),
            },
            "performance": {
                "p50_latency_ms": statistics.median(latency_values) if latency_values else 0.0,
                "max_latency_ms": max(latency_values) if latency_values else 0.0,
            },
        },
        "by_category": {
            category: {
                "questions": len(category_rows),
                "hit_rate": mean(category_rows, "hit_at_k"),
                "behavior_pass_rate": mean(category_rows, "behavior_pass"),
                "answer_rate": mean(category_rows, "answer_present") if with_llm else None,
                "citation_rate": mean([{"value": row["citation_count"] > 0} for row in category_rows], "value") if with_llm else None,
            }
            for category, category_rows in sorted(groups.items())
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DevRag's multi-phase RAG benchmark")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--with-llm", action="store_true", help="Call the configured LLM for every case")
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    rows = []
    with httpx.Client(timeout=120.0) as client:
        for index, case in enumerate(cases, start=1):
            row = evaluate_case(client, args.base_url, case, args.top_k, args.with_llm)
            rows.append(row)
            status = "PASS" if row["hit_at_k"] or row["behavior_pass"] else "MISS"
            print(f"[{status}] {index:02d}/{len(cases)} {case['id']} latency={row['latency_ms']:.0f}ms")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "summary": summarize(rows, args.top_k, args.with_llm),
        "cases": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print("\nSummary")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
