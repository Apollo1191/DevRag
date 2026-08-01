"""Utilities for cloning a GitHub repository and walking source files.

This module is the first step of the ingestion pipeline. It keeps the
responsibility narrow: fetch code into a local workspace and enumerate the
files we care about for later chunking.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceFile:
    """A file discovered in a cloned repository.

    Attributes:
        path: Absolute path to the file on disk.
        relative_path: Repository-relative path, which is stable for metadata.
    """

    path: Path
    relative_path: Path


class RepositoryLoaderError(RuntimeError):
    """Raised when repository cloning or file discovery fails."""


class RepositoryLoader:
    """Clone a repository and enumerate files needed by the ingestion pipeline."""

    def __init__(self, workspace_root: Path) -> None:
        """Create a loader anchored to a local workspace directory.

        Args:
            workspace_root: Directory where repositories will be cloned.
        """

        self.workspace_root = workspace_root

    def clone_repository(self, repo_url: str, repo_name: str | None = None) -> Path:
        """Clone a GitHub repository into the workspace.

        Args:
            repo_url: HTTPS or SSH repository URL.
            repo_name: Optional override for the destination folder name.

        Returns:
            Path to the cloned repository root.

        Raises:
            RepositoryLoaderError: If the clone operation fails.
        """

        # Keep the clone destination stable so later ingestion steps can reuse it.
        target_name = repo_name or self._derive_repo_name(repo_url)
        target_path = self.workspace_root / target_name

        if target_path.exists():
            # Start from a clean checkout so stale files do not leak into indexing.
            logger.info("Removing existing repository clone at %s", target_path)
            shutil.rmtree(target_path)

        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Use the system git client for a shallow clone to keep the first pass simple.
        logger.info("Cloning %s into %s", repo_url, target_path)
        try:
            self._run_git(["git", "clone", "--depth", "1", repo_url, str(target_path)])
        except subprocess.CalledProcessError as exc:
            logger.exception("Failed to clone repository: %s", repo_url)
            raise RepositoryLoaderError(
                f"Failed to clone repository {repo_url}: {exc.stderr.strip()}"
            ) from exc

        return target_path

    def walk_source_files(
        self,
        repository_path: Path,
        allowed_suffixes: Iterable[str] | None = None,
    ) -> list[SourceFile]:
        """Collect the files that should be chunked later.

        Args:
            repository_path: Root of the cloned repository.
            allowed_suffixes: File suffixes to include, such as ".py" and ".md".

        Returns:
            A list of source files with absolute and repository-relative paths.
        """

        suffixes = tuple(allowed_suffixes or (".py", ".md"))
        collected_files: list[SourceFile] = []

        for file_path in repository_path.rglob("*"):
            # Skip directories and non-file entries such as symlinks to folders.
            if not file_path.is_file():
                continue

            # Limit the first version to source code and documentation files.
            if file_path.suffix.lower() not in suffixes:
                continue

            relative_path = file_path.relative_to(repository_path)
            collected_files.append(
                SourceFile(path=file_path, relative_path=relative_path)
            )

        collected_files.sort(key=lambda item: str(item.relative_path).lower())
        return collected_files

    @staticmethod
    def _derive_repo_name(repo_url: str) -> str:
        """Extract a folder name from a repository URL."""

        parsed_url = urlparse(repo_url)
        if parsed_url.scheme in {"http", "https"}:
            repo_name = Path(parsed_url.path).name
        elif "@" in repo_url and ":" in repo_url:
            # Handle SSH URLs like git@github.com:owner/repo.git.
            repo_name = repo_url.rsplit(":", 1)[-1].rstrip("/")
        else:
            repo_name = Path(repo_url.rstrip("/")).name

        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        if not repo_name:
            raise RepositoryLoaderError(f"Could not derive repository name from {repo_url}")

        return repo_name

    @staticmethod
    def _run_git(command: list[str]) -> subprocess.CompletedProcess[str]:
        """Run a git command with UTF-8 decoding to avoid locale issues on Windows."""

        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
