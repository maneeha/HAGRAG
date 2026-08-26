from hagrag.data import PDFProcessor


def test_separator_aware_chunking_preserves_overlap():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = PDFProcessor._split_text(text, chunk_size=30, overlap=5, separators=(". ", " ", ""))
    assert len(chunks) >= 2
    assert all(len(chunk) <= 30 for chunk in chunks)
