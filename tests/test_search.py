
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from rag_app.retrieval.search import search_dense


class FakeEncoder:
    def encode(self, text, **kwargs):
        return np.ones(384, dtype=np.float32)


def test_search_dense_returns_source_documents():
    client = Mock()
    client.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(
                id="chunk-001",
                score=0.91,
                payload={
                    "title": "What Is Deep Learning?",
                    "content": "Full original document.",
                    "chunk": "Deep learning uses neural networks.",
                    "category": "Fundamentals",
                    "chunk_strategy": "fixed",
                },
            )
        ]
    )

    results = search_dense(
        query="What is deep learning?",
        top_k=3,
        strategy="fixed",
        client=client,
        encoder=FakeEncoder(),
    )

    assert len(results) == 1
    assert results[0].id == "chunk-001"
    assert results[0].title == "What Is Deep Learning?"
    assert results[0].content == "Deep learning uses neural networks."
    assert results[0].score == pytest.approx(0.91)
    assert results[0].metadata["category"] == "Fundamentals"

    client.query_points.assert_called_once()
    assert client.query_points.call_args.kwargs["using"] == "fixed"


def test_search_dense_rejects_empty_query():
    with pytest.raises(ValueError, match="empty"):
        search_dense("   ", client=Mock(), encoder=FakeEncoder())


def test_search_dense_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        search_dense(
            "Explain RAG",
            top_k=0,
            client=Mock(),
            encoder=FakeEncoder(),
        )


def test_search_dense_rejects_invalid_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        search_dense(
            "Explain RAG",
            strategy="unknown",
            client=Mock(),
            encoder=FakeEncoder(),
        )
