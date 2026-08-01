"""Prompt builder using versioned YAML templates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from ..retrieval.vector_store import SearchResult


@dataclass
class PromptTemplate:
    """Represents a prompt template with system and user sections."""

    system: str
    user: str


def load_prompt_template(path: Path) -> PromptTemplate:
    """Load a prompt template from YAML."""

    data = yaml.safe_load(path.read_text(encoding="utf8"))
    return PromptTemplate(system=data["system"], user=data["user"])


def build_prompt(
    question: str,
    results: Iterable[SearchResult],
    template: PromptTemplate,
    max_source_chars: int = 6000,
    max_total_source_chars: int = 18000,
) -> tuple[str, str]:
    """Build a system and user prompt string from results."""

    sources = []
    total_chars = 0
    for idx, result in enumerate(results, start=1):
        meta = result.metadata
        ref = f"{meta.get('relative_path')}:{meta.get('start_line')}-{meta.get('end_line')}"
        remaining_chars = max_total_source_chars - total_chars
        if remaining_chars <= 0:
            break
        source_text = result.text[: min(max_source_chars, remaining_chars)]
        if len(source_text) < len(result.text):
            source_text = f"{source_text}\n[Source truncated for context budget]"
        sources.append(f"[Source {idx}] {ref}\n{source_text}")
        total_chars += len(source_text)

    user_prompt = template.user.format(question=question, sources="\n\n".join(sources))
    return template.system, user_prompt


def build_retrieval_fallback(question: str, results: Iterable[SearchResult], limit: int = 3) -> str | None:
    """Build a transparent source-based answer when an LLM is unavailable."""

    result_list = list(results)[:limit]
    if not result_list:
        return None

    lines = [
        "Gemini is unavailable right now, so here are the most relevant indexed sources for your question:",
        "",
        f"Question: {question}",
    ]
    for index, result in enumerate(result_list, start=1):
        metadata = result.metadata
        reference = f"{metadata.get('relative_path')}:{metadata.get('start_line')}-{metadata.get('end_line')}"
        excerpt = " ".join(result.text.split())
        if len(excerpt) > 500:
            excerpt = f"{excerpt[:497]}..."
        lines.extend(["", f"[Source {index}] {reference}", excerpt])

    return "\n".join(lines)
