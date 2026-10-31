
from unittest.mock import patch

from rag_app.schemas.request import RAGRequest
from rag_app.schemas.response import RAGResponse, SourceDocument
from rag_app.services.rag_service import answer_question


def make_source() -> SourceDocument:
    return SourceDocument(
        id="source-1",
        title="Deep Learning",
        content="Deep learning uses multilayer neural networks.",
        score=0.9,
    )


@patch("rag_app.services.rag_service.generate_answer")
@patch("rag_app.services.rag_service.retrieve_documents")
def test_answer_question_returns_structured_response(
    mock_retrieve,
    mock_generate,
):
    mock_retrieve.return_value = [make_source()]
    mock_generate.return_value = (
        "Deep learning uses multilayer neural networks. "
        "[Source ID: source-1]"
    )

    request = RAGRequest(
        question="What is deep learning?",
        retrieval_mode="hybrid",
        top_k=3,
    )

    response = answer_question(
        request,
        model_name="test-model",
    )

    assert isinstance(response, RAGResponse)
    assert "multilayer neural networks" in response.answer
    assert len(response.sources) == 1
    assert response.sources[0].id == "source-1"
    assert response.model == "test-model"

    mock_retrieve.assert_called_once()
    mock_generate.assert_called_once()


@patch("rag_app.services.rag_service.generate_answer")
@patch("rag_app.services.rag_service.retrieve_documents")
def test_answer_question_passes_sources_to_generation(
    mock_retrieve,
    mock_generate,
):
    source = make_source()
    mock_retrieve.return_value = [source]
    mock_generate.return_value = "Answer based on the source."

    request = RAGRequest(question="Explain deep learning.")

    answer_question(request)

    kwargs = mock_generate.call_args.kwargs
    assert kwargs["sources"] == [source]
    assert kwargs["question"] == request.question
