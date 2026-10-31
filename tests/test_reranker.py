
from unittest.mock import Mock

import pytest

from rag_app.retrieval.reranker import rerank_documents
from rag_app.schemas.response import SourceDocument


def make_document(document_id: str) -> SourceDocument:
    return SourceDocument(
        id=document_id,
        title=f"Document {document_id}",
        content=f"Content {document_id}",
    )


def test_reranks_documents_by_score():
    documents = [
        make_document("A"),
        make_document("B"),
        make_document("C"),
    ]

    model = Mock()
    model.predict.return_value = [0.2, 0.9, 0.5]

    results = rerank_documents(
        query="deep learning",
        documents=documents,
        top_k=2,
        reranker=model,
    )

    assert [document.id for document in results] == ["B", "C"]
    assert results[0].score == pytest.approx(0.9)
    assert results[1].score == pytest.approx(0.5)


def test_empty_documents_return_empty_list():
    assert rerank_documents("question", []) == []


def test_rejects_empty_query():
    with pytest.raises(ValueError):
        rerank_documents("  ", [make_document("A")])


def test_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        rerank_documents("question", [], top_k=0)


def test_does_not_modify_original_documents():
    document = make_document("A")
    model = Mock()
    model.predict.return_value = [0.8]

    rerank_documents("question", [document], reranker=model)

    assert document.score is None
