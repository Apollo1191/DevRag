"""Chunk source files into retrieval-friendly pieces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


@dataclass(frozen=True)
class Chunk:
    """A chunk extracted from a source file."""

    text: str
    start_line: int
    end_line: int
    metadata: dict = field(default_factory=dict)


def chunk_text_by_language(source: str, relative_path: Path) -> list[Chunk]:
    """Chunk a file based on its extension."""

    if relative_path.suffix.lower() == ".md":
        return _split_markdown(source, relative_path)
    if relative_path.suffix.lower() == ".py":
        return _split_python(source, relative_path)
    return [
        Chunk(
            text=source,
            start_line=1,
            end_line=max(source.count("\n") + 1, 1),
            metadata={"relative_path": str(relative_path)},
        )
    ]


def _split_python(source: str, relative_path: Path) -> list[Chunk]:
    """Split Python files into module header plus top-level defs/classes."""

    lines = source.splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    start_line = 1
    current_start = 1
    seen_top_level = False

    for line_number, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        is_top_level = line == stripped
        is_block_start = is_top_level and re.match(r"^(def|class)\s+", stripped)

        if is_block_start:
            if current_start < line_number:
                text = "\n".join(lines[current_start - 1 : line_number - 1]).strip()
                if text:
                    chunks.append(
                        Chunk(
                            text=text,
                            start_line=current_start,
                            end_line=line_number - 1,
                            metadata={"relative_path": str(relative_path)},
                        )
                    )
            current_start = line_number
            seen_top_level = True

        if seen_top_level and is_top_level and line_number > current_start and stripped:
            # When another top-level statement appears, end the previous block.
            if not is_block_start:
                text = "\n".join(lines[current_start - 1 : line_number - 1]).strip()
                if text:
                    chunks.append(
                        Chunk(
                            text=text,
                            start_line=current_start,
                            end_line=line_number - 1,
                            metadata={"relative_path": str(relative_path)},
                        )
                    )
                current_start = line_number

    tail = "\n".join(lines[current_start - 1 :]).strip()
    if tail:
        chunks.append(
            Chunk(
                text=tail,
                start_line=current_start,
                end_line=len(lines),
                metadata={"relative_path": str(relative_path)},
            )
        )

    if not chunks:
        return _split_python_heuristic(source, relative_path)

    return chunks


def _split_python_heuristic(source: str, relative_path: Path) -> list[Chunk]:
    """Fallback Python chunking when the simple splitter cannot separate blocks."""

    lines = source.splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    start = 1
    for line_number, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if line == stripped and re.match(r"^(def|class)\s+", stripped):
            header = "\n".join(lines[start - 1 : line_number - 1]).strip()
            if header:
                chunks.append(
                    Chunk(
                        text=header,
                        start_line=start,
                        end_line=line_number - 1,
                        metadata={"relative_path": str(relative_path)},
                    )
                )
            start = line_number

    body = "\n".join(lines[start - 1 :]).strip()
    if body:
        chunks.append(
            Chunk(
                text=body,
                start_line=start,
                end_line=len(lines),
                metadata={"relative_path": str(relative_path)},
            )
        )

    return chunks


def _split_markdown(source: str, relative_path: Path) -> list[Chunk]:
    """Split Markdown by headings."""

    lines = source.splitlines()
    if not lines:
        return []

    heading_re = re.compile(r"^(#{1,6})\s+.+")
    chunks: list[Chunk] = []
    start = 1
    current_heading = None

    for line_number, line in enumerate(lines, start=1):
        if heading_re.match(line):
            if line_number > start:
                section = "\n".join(lines[start - 1 : line_number - 1]).strip()
                if section:
                    chunks.append(
                        Chunk(
                            text=section,
                            start_line=start,
                            end_line=line_number - 1,
                            metadata={"relative_path": str(relative_path), "heading": current_heading},
                        )
                    )
            start = line_number
            current_heading = line.lstrip("# ").strip()

    tail = "\n".join(lines[start - 1 :]).strip()
    if tail:
        chunks.append(
            Chunk(
                text=tail,
                start_line=start,
                end_line=len(lines),
                metadata={"relative_path": str(relative_path), "heading": current_heading},
            )
        )

    return chunks
