
import pytest

from rag_app.ingestion.chunking import (
    fixed_size_chunks,
    paragraph_chunks,
    sentence_chunks,
)


def test_fixed_size_chunks():
    chunks = fixed_size_chunks(
        "one two three four five six",
        chunk_size=4,
        overlap=2,
    )

    assert chunks == [
        "one two three four",
        "three four five six",
    ]


def test_fixed_size_chunks_empty_text():
    assert fixed_size_chunks("") == []


def test_fixed_size_chunks_invalid_size():
    with pytest.raises(ValueError):
        fixed_size_chunks("some text", chunk_size=0)


def test_fixed_size_chunks_invalid_overlap():
    with pytest.raises(ValueError):
        fixed_size_chunks("some text", chunk_size=4, overlap=4)


def test_sentence_chunks():
    chunks = sentence_chunks(
        "Deep learning is useful. Neural networks learn patterns. "
        "Models need data. Training adjusts weights.",
        max_sentences=2,
    )

    assert len(chunks) == 2
    assert chunks[0] == (
        "Deep learning is useful. Neural networks learn patterns."
    )


def test_sentence_chunks_empty_text():
    assert sentence_chunks("") == []


def test_paragraph_chunks():
    chunks = paragraph_chunks("First paragraph.\n\nSecond paragraph.")

    assert chunks == ["First paragraph.", "Second paragraph."]


def test_paragraph_chunks_single_paragraph():
    assert paragraph_chunks("One paragraph.") == ["One paragraph."]


def test_paragraph_chunks_empty_text():
    assert paragraph_chunks("") == []
