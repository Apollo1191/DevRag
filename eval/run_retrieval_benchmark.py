"""Evaluate retrieval quality against a JSONL question set.

Run from the repository root while the backend is running:
    python -m eval.run_retrieval_benchmark
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


DEFAULT_DATASET = Path(__file__).with_name("questions.jsonl")
DEFAULT_REPORT = Path(__file__).with_name("retrieval_report.json")


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load benchmark cases from JSONL."""

    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
        for field in ("id", "repo", "question", "relevant_paths"):
            if field not in case:
                raise ValueError(f"Missing '{field}' on line {line_number} of {path}")
        cases.append(case)
    return cases


def path_matches(actual: str, expected: str) -> bool:
    """Match slash-normalized repository paths and directory prefixes."""

    actual_path = actual.replace("\\", "/").lower()
    expected_path = expected.replace("\\", "/").lower()
    return actual_path == expected_path or actual_path.startswith(expected_path.rstrip("/") + "/")


def evaluate_case(client: httpx.Client, base_url: str, case: dict[str, Any], top_k: int) -> dict[str, Any]:
    """Run one retrieval-only query and calculate ranking metrics."""

    response = client.post(
        f"{base_url.rstrip('/')}/query",
        json={"question": case["question"], "top_k": top_k, "use_llm": False},
    )
    response.raise_for_status()
    body = response.json()
    chunks = body.get("chunks", [])
    expected_paths = case["relevant_paths"]

    matched_ranks: list[int] = []
    retrieved_paths: list[str] = []
    for rank, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        actual_path = str(metadata.get("relative_path") or "")
        repo_name = str(metadata.get("source_repo") or "")
        if actual_path:
            retrieved_paths.append(actual_path)
        if repo_name == case["repo"] and any(path_matches(actual_path, expected) for expected in expected_paths):
            matched_ranks.append(rank)

    relevant_count = len(set(matched_ranks))
    reciprocal_rank = 1.0 / matched_ranks[0] if matched_ranks else 0.0
    return {
        "id": case["id"],
        "repo": case["repo"],
        "question": case["question"],
        "expected_paths": expected_paths,
        "retrieved_paths": retrieved_paths,
        "matched_ranks": matched_ranks,
        "hit_at_k": bool(matched_ranks),
        "recall_at_k": min(relevant_count / len(expected_paths), 1.0),
        "mrr": reciprocal_rank,
    }


def average(rows: list[dict[str, Any]], field: str) -> float:
    """Average one numeric result field."""

    return sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0


def build_summary(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    """Build overall and per-repository benchmark metrics."""

    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_repo[row["repo"]].append(row)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "questions": len(group),
            f"hit_at_{top_k}": average([{"value": float(row["hit_at_k"])} for row in group], "value"),
            f"recall_at_{top_k}": average(group, "recall_at_k"),
            "mrr": average(group, "mrr"),
        }

    return {
        "top_k": top_k,
        "overall": summarize(rows),
        "by_repo": {repo: summarize(repo_rows) for repo, repo_rows in sorted(by_repo.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DevRag retrieval benchmark")
    parser.add_argument("--base-url", default="http://localhost:8000", help="DevRag API base URL")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cases = load_questions(args.dataset)
    rows = []
    with httpx.Client(timeout=120.0) as client:
        for index, case in enumerate(cases, start=1):
            row = evaluate_case(client, args.base_url, case, args.top_k)
            rows.append(row)
            status = "PASS" if row["hit_at_k"] else "MISS"
            print(f"[{status}] {index:02d}/{len(cases)} {case['id']} MRR={row['mrr']:.3f}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "summary": build_summary(rows, args.top_k),
        "cases": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf8")

    summary = report["summary"]["overall"]
    print("\nSummary")
    print(json.dumps(summary, indent=2))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
