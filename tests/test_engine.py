from unittest.mock import patch

import pytest

from lestrade.rag.engine import RAGEngine


@pytest.fixture
def engine():
    eng = RAGEngine()
    eng.index.reset()
    eng.entries = []
    eng.source_ids = {}
    eng.next_id = 0
    return eng


@pytest.fixture
def mock_embed():
    with patch("lestrade.rag.engine.embed_text") as mock:
        mock.return_value = [0.1] * 1024
        yield mock


@pytest.mark.asyncio
async def test_add_text(engine, mock_embed):
    count = await engine.add_text("Hello world. This is a test document.", source="test.md")
    assert count >= 1
    assert engine.index.ntotal >= 1
    assert len(engine.entries) >= 1
    assert engine.entries[0]["source"] == "test.md"
    assert "[test.md]" in engine.entries[0]["text"]


@pytest.mark.asyncio
async def test_add_multiple_docs(engine, mock_embed):
    await engine.add_text("Doc one content.", source="one.md")
    await engine.add_text("Doc two content.", source="two.md")
    assert engine.index.ntotal >= 2
    assert "one.md" in engine.source_ids
    assert "two.md" in engine.source_ids


@pytest.mark.asyncio
async def test_remove_by_source(engine, mock_embed):
    await engine.add_text("Doc one.", source="one.md")
    n_before = engine.index.ntotal
    engine.remove_by_source("one.md")
    assert engine.index.ntotal < n_before
    assert "one.md" not in engine.source_ids
    assert all(e["source"] != "one.md" for e in engine.entries)


@pytest.mark.asyncio
async def test_remove_nonexistent_source(engine, mock_embed):
    engine.remove_by_source("nonexistent.md")


@pytest.mark.asyncio
async def test_search_empty(engine):
    results = await engine.search("anything")
    assert results == []


@pytest.mark.asyncio
async def test_search_returns_results(engine, mock_embed):
    await engine.add_text("Python is a popular programming language.", source="en/python.md")
    await engine.add_text("Java is also widely used in enterprise.", source="en/java.md")

    assert engine.index.ntotal >= 2
    assert len(engine.entries) >= 2

    results = await engine.search("programming language", k=5)
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_search_language_filter(engine, mock_embed):
    await engine.add_text("English content here.", source="en/doc.md")
    await engine.add_text("Chinese content here.", source="zh/doc.md")

    assert engine.index.ntotal >= 2

    en_results = await engine.search("content", k=5, lang="en")
    zh_results = await engine.search("content", k=5, lang="zh")

    en_sources = [s for _, s, _ in en_results]
    zh_sources = [s for _, s, _ in zh_results]
    assert all(not s.startswith("zh/") for s in en_sources)
    assert all(not s.startswith("en/") for s in zh_sources)


@pytest.mark.asyncio
async def test_search_result_format(engine, mock_embed):
    await engine.add_text("Sample content for testing.", source="test.md")

    assert engine.index.ntotal >= 1

    results = await engine.search("sample", k=3)

    for item in results:
        assert len(item) == 3
        text, source, score = item
        assert isinstance(text, str)
        assert isinstance(source, str)
        assert isinstance(score, (int, float))
