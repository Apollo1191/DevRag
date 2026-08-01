import textwrap
from src.ingestion.chunker import chunk_text_by_language, Chunk
from pathlib import Path


def test_python_chunking():
    src = textwrap.dedent('''
    """Module docstring.
    More info.
    """

    @decorator
    def foo(a, b):
        """Function foo"""
        return a + b


    class Bar:
        def method(self):
            pass
    ''').strip()

    chunks = chunk_text_by_language(src, Path("example.py"))
    # Expect module header + foo + class Bar -> at least 3 chunks
    assert len(chunks) >= 3
    texts = [c.text for c in chunks]
    assert any('def foo' in t for t in texts)
    assert any('class Bar' in t for t in texts)
    assert any('Module docstring' in t for t in texts)


def test_markdown_chunking():
    md = textwrap.dedent('''
    # Title

    Intro paragraph.

    ## Section A

    Content A

    ## Section B

    Content B
    ''').strip()

    chunks = chunk_text_by_language(md, Path("README.md"))
    # Expect chunks for each header (Title, Section A, Section B)
    assert len(chunks) >= 3
    assert any(c.text.startswith('# Title') for c in chunks)
    assert any('## Section A' in c.text for c in chunks)
    assert any('## Section B' in c.text for c in chunks)
