from lestrade.plugins.chunking import MarkdownChunking

chunker = MarkdownChunking()


def test_strip_frontmatter_removes_yaml():
    text = "---\ntitle: Test\nauthor: me\n---\n# Header\nContent"
    result = chunker._strip_frontmatter(text)
    assert "title" not in result
    assert "# Header" in result
    assert "Content" in result


def test_strip_frontmatter_no_frontmatter():
    text = "# Just a header\nContent"
    result = chunker._strip_frontmatter(text)
    assert result == text


def test_chunk_basic():
    text = "## Section A\nThis is some content about section A.\n\n## Section B\nThis is content for section B."
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    all_text = " ".join(chunks)
    assert "Section A" in all_text
    assert "Section B" in all_text


def test_chunk_skips_separators():
    text = "## Intro\nHello world.\n\n---\n\n## Main\nMain content here."
    chunks = chunker.chunk(text)
    for c in chunks:
        assert "---" not in c


def test_chunk_skips_comments():
    text = "## Intro\nHello.\n\n<!-- This is a comment -->\n\n## Next\nMore content."
    chunks = chunker.chunk(text)
    for c in chunks:
        assert "<!--" not in c


def test_chunk_preserves_code_blocks():
    text = "## Example\n```python\nprint('hello')\n```\n\nMore text."
    chunks = chunker.chunk(text)
    all_text = " ".join(chunks)
    assert "```python" in all_text


def test_chunk_merges_small():
    text = "## A\nHi.\n\n## B\nOk.\n\n## C\nShort."
    chunks = chunker.chunk(text)
    assert len(chunks) <= 3


def test_chunk_empty():
    chunks = chunker.chunk("")
    assert chunks == []


def test_chunk_frontmatter_stripped():
    text = "---\nkey: value\n---\n## Real Content\nThis is the content."
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    assert "key: value" not in chunks[0]
