
import pytest
from pydantic import ValidationError

from rag_app.schemas.request import RAGRequest
from rag_app.schemas.response import RAGResponse, SourceDocument


def test_valid_rag_request():
    request = RAGRequest(
        question="What is deep learning?",
        top_k=5,
        retrieval_mode="hybrid",
    )

    assert request.question == "What is deep learning?"
    assert request.top_k == 5
    assert request.retrieval_mode == "hybrid"


def test_question_cannot_be_empty():
    with pytest.raises(ValidationError):
        RAGRequest(question="   ")


def test_top_k_must_be_valid():
    with pytest.raises(ValidationError):
        RAGRequest(question="Explain RAG", top_k=0)

    with pytest.raises(ValidationError):
        RAGRequest(question="Explain RAG", top_k=21)


def test_invalid_retrieval_mode():
    with pytest.raises(ValidationError):
        RAGRequest(
            question="Explain RAG",
            retrieval_mode="invalid",
        )


def test_rag_response_with_source():
    response = RAGResponse(
        answer="Deep learning uses multilayer neural networks.",
        sources=[
            SourceDocument(
                id="chunk-001",
                title="Deep Learning",
                content="Deep learning uses multilayer neural networks.",
                score=0.87,
                metadata={"category": "Fundamentals"},
            )
        ],
        model="gemini-2.5-flash",
    )

    assert len(response.sources) == 1
    assert response.sources[0].score == 0.87


def test_rag_response_without_sources():
    response = RAGResponse(answer="I could not find relevant information.")

    assert response.sources == []
