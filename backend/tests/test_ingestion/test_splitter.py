def test_recursive_splitter_chunks_by_separators():
    from app.ingestion.splitter import RecursiveSplitter
    splitter = RecursiveSplitter(chunk_size=30, chunk_overlap=5)
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.\n\nFourth paragraph here."
    chunks = splitter.split(text)
    assert len(chunks) >= 2


def test_markdown_splitter_preserves_headings():
    from app.ingestion.splitter import MarkdownSplitter
    splitter = MarkdownSplitter(chunk_size=500, chunk_overlap=0)
    text = "## Section A\n\nContent for A.\n\n## Section B\n\nContent for B."
    chunks = splitter.split(text)
    assert any("Section A" in c for c in chunks)
    assert any("Section B" in c for c in chunks)


def test_get_splitter_raises_for_unknown():
    import pytest
    from app.ingestion.splitter import get_splitter
    with pytest.raises(ValueError, match="Unknown splitter"):
        get_splitter("nonexistent")
