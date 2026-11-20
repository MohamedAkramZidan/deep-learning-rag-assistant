from unittest.mock import Mock, patch

import pytest
from pydantic import SecretStr

from rag_app.generation.client import create_gemini_client
from rag_app.retrieval.embeddings import load_dense_encoder
from rag_app.retrieval.reranker import load_reranker
from rag_app.retrieval.sparse import load_sparse_encoder
from rag_app.retrieval.vector_store import create_qdrant_client


@pytest.fixture(autouse=True)
def clear_cached_resources():
    yield
    create_gemini_client.cache_clear()
    create_qdrant_client.cache_clear()
    load_dense_encoder.cache_clear()
    load_sparse_encoder.cache_clear()
    load_reranker.cache_clear()


def test_create_gemini_client_is_cached(monkeypatch):
    monkeypatch.setattr(
        "rag_app.generation.client.settings.gemini_api_key",
        SecretStr("test-gemini-key"),
    )

    create_gemini_client.cache_clear()
    fake_client = Mock()

    with patch(
        "rag_app.generation.client.genai.Client",
        return_value=fake_client,
    ) as constructor:
        first = create_gemini_client()
        second = create_gemini_client()

    assert first is fake_client
    assert second is fake_client
    constructor.assert_called_once()


def test_create_qdrant_client_is_cached():
    create_qdrant_client.cache_clear()
    fake_client = Mock()

    with patch(
        "rag_app.retrieval.vector_store.QdrantClient",
        return_value=fake_client,
    ) as constructor:
        first = create_qdrant_client()
        second = create_qdrant_client()

    assert first is fake_client
    assert second is fake_client
    constructor.assert_called_once()


def test_dense_encoder_is_cached():
    load_dense_encoder.cache_clear()
    fake_encoder = Mock()

    with patch(
        "rag_app.retrieval.embeddings.SentenceTransformer",
        return_value=fake_encoder,
    ) as constructor:
        first = load_dense_encoder()
        second = load_dense_encoder()

    assert first is fake_encoder
    assert second is fake_encoder
    constructor.assert_called_once()


def test_sparse_encoder_is_cached():
    load_sparse_encoder.cache_clear()
    fake_encoder = Mock()

    with patch(
        "rag_app.retrieval.sparse.SparseTextEmbedding",
        return_value=fake_encoder,
    ) as constructor:
        first = load_sparse_encoder()
        second = load_sparse_encoder()

    assert first is fake_encoder
    assert second is fake_encoder
    constructor.assert_called_once()


def test_reranker_is_cached():
    load_reranker.cache_clear()
    fake_reranker = Mock()

    with patch(
        "rag_app.retrieval.reranker.CrossEncoder",
        return_value=fake_reranker,
    ) as constructor:
        first = load_reranker()
        second = load_reranker()

    assert first is fake_reranker
    assert second is fake_reranker
    constructor.assert_called_once()
