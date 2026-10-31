
from unittest.mock import Mock, patch

import pytest

from rag_app.retrieval.search import search_hybrid
from rag_app.schemas.response import SourceDocument


def doc(document_id: str) -> SourceDocument:
    return SourceDocument(
        id=document_id,
        title=f"Document {document_id}",
        content=f"Content {document_id}",
    )


@patch("rag_app.retrieval.search.search_sparse")
@patch("rag_app.retrieval.search.search_dense")
def test_hybrid_combines_dense_and_sparse_results(
    mock_dense,
    mock_sparse,
):
    mock_dense.return_value = [doc("A"), doc("B")]
    mock_sparse.return_value = [doc("B"), doc("C")]

    results = search_hybrid("deep learning", top_k=3, client=Mock())

    assert [item.id for item in results] == ["B", "A", "C"]
    mock_dense.assert_called_once()
    mock_sparse.assert_called_once()


@pytest.mark.parametrize("query", ["", "   "])
def test_hybrid_rejects_empty_query(query):
    with pytest.raises(ValueError):
        search_hybrid(query)


@pytest.mark.parametrize("top_k", [0, 21])
def test_hybrid_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError):
        search_hybrid("deep learning", top_k=top_k)


def test_hybrid_rejects_invalid_strategy():
    with pytest.raises(ValueError):
        search_hybrid("deep learning", strategy="invalid")
