from pathlib import Path
import shutil

from src.ingestion.loader import RepositoryLoader
from src.ingestion.chunker import chunk_text_by_language


def main():
    workspace = Path(".").resolve() / "workspaces"
    sample = workspace / "sample_repo"
    if sample.exists():
        shutil.rmtree(sample)
    sample.mkdir(parents=True)

    (sample / "sample.py").write_text(
        '"""Sample module docstring.\nMore info.\n"""\n\n@decorator\ndef foo(a, b):\n    return a + b\n\nclass Bar:\n    def method(self):\n        pass\n'
    )

    (sample / "README.md").write_text("# Title\n\nIntro paragraph.\n\n## Section A\n\nContent A\n")

    loader = RepositoryLoader(workspace_root=workspace)
    # Use the local sample as a repository root for this demo.
    files = loader.walk_source_files(sample, allowed_suffixes=[".py", ".md"]) 

    for sf in files:
        text = sf.path.read_text(encoding="utf8")
        chunks = chunk_text_by_language(text, sf.relative_path)
        print(f"File: {sf.relative_path} -> {len(chunks)} chunks")
        for c in chunks:
            first_line = c.text.splitlines()[0] if c.text.splitlines() else ""
            print(f"  lines {c.start_line}-{c.end_line}: {first_line[:80]}")

    print("E2E local demo complete")


if __name__ == "__main__":
    main()
