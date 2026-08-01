import sys
import textwrap
from pathlib import Path
from src.ingestion.chunker import chunk_text_by_language


def run():
    failures = 0

    # Test 1: Python chunking
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
    try:
        assert len(chunks) >= 3
        texts = [c.text for c in chunks]
        assert any('def foo' in t for t in texts)
        assert any('class Bar' in t for t in texts)
        assert any('Module docstring' in t for t in texts)
        print("test_python_chunking: PASS")
    except AssertionError:
        print("test_python_chunking: FAIL")
        failures += 1

    # Test 2: Markdown chunking
    md = textwrap.dedent('''
    # Title

    Intro paragraph.

    ## Section A

    Content A

    ## Section B

    Content B
    ''').strip()

    chunks = chunk_text_by_language(md, Path("README.md"))
    try:
        assert len(chunks) >= 3
        assert any(c.text.startswith('# Title') for c in chunks)
        assert any('## Section A' in c.text for c in chunks)
        assert any('## Section B' in c.text for c in chunks)
        print("test_markdown_chunking: PASS")
    except AssertionError:
        print("test_markdown_chunking: FAIL")
        failures += 1

    if failures:
        print(f"{failures} tests failed")
        sys.exit(2)
    else:
        print("All tests passed")
        sys.exit(0)


if __name__ == '__main__':
    run()
