
from unittest.mock import Mock, patch

from rag_app.schemas.request import RAGRequest
from rag_app.schemas.response import SourceDocument
from rag_app.services.rag_service import retrieve_documents


def make_document(document_id: str) -> SourceDocument:
    return SourceDocument(
        id=document_id,
        title=f"Document {document_id}",
        content=f"Content {document_id}",
    )


@patch("rag_app.services.rag_service.search_dense")
def test_dense_mode(mock_search):
    mock_search.return_value = [make_document("A")]

    result = retrieve_documents(
        RAGRequest(question="What is deep learning?", retrieval_mode="dense"),
        client=Mock(),
    )

    assert [document.id for document in result] == ["A"]
    mock_search.assert_called_once()


@patch("rag_app.services.rag_service.search_sparse")
def test_sparse_mode(mock_search):
    mock_search.return_value = [make_document("A")]

    result = retrieve_documents(
        RAGRequest(question="What is deep learning?", retrieval_mode="sparse"),
        client=Mock(),
    )

    assert [document.id for document in result] == ["A"]
    mock_search.assert_called_once()


@patch("rag_app.services.rag_service.search_hybrid")
def test_hybrid_mode(mock_search):
    mock_search.return_value = [make_document("A")]

    result = retrieve_documents(
        RAGRequest(question="What is deep learning?", retrieval_mode="hybrid"),
        client=Mock(),
    )

    assert [document.id for document in result] == ["A"]
    mock_search.assert_called_once()


@patch("rag_app.services.rag_service.rerank_documents")
@patch("rag_app.services.rag_service.search_hybrid")
def test_reranks_candidates(mock_search, mock_rerank):
    mock_search.return_value = [
        make_document("A"),
        make_document("B"),
    ]
    mock_rerank.return_value = [make_document("B")]

    result = retrieve_documents(
        RAGRequest(question="What is deep learning?", top_k=1),
        rerank=True,
        client=Mock(),
        reranker=Mock(),
    )

    assert [document.id for document in result] == ["B"]
    mock_rerank.assert_called_once()
    assert mock_search.call_args.kwargs["top_k"] == 10
