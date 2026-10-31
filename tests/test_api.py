
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_app.main import app
from rag_app.schemas.response import RAGResponse, SourceDocument

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("rag_app.api.routes.answer_question")
def test_query_success(mock_answer_question) -> None:
    mock_answer_question.return_value = RAGResponse(
        answer="Gradient descent minimizes a loss function.",
        sources=[
            SourceDocument(
                id="doc-1",
                title="Gradient Descent",
                content="Gradient descent updates parameters to reduce loss.",
                score=0.92,
                metadata={"category": "deep-learning"},
            )
        ],
        model="gemini-2.5-flash",
    )

    response = client.post(
        "/api/v1/query",
        json={
            "question": "What is gradient descent?",
            "top_k": 5,
            "retrieval_mode": "hybrid",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Gradient descent minimizes a loss function."
    assert data["model"] == "gemini-2.5-flash"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["id"] == "doc-1"

    mock_answer_question.assert_called_once()


@patch("rag_app.api.routes.answer_question")
def test_query_service_error(mock_answer_question) -> None:
    mock_answer_question.side_effect = RuntimeError("Internal service error")

    response = client.post(
        "/api/v1/query",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to process the RAG request."
    assert "Internal service error" not in response.text


def test_query_rejects_invalid_request() -> None:
    response = client.post(
        "/api/v1/query",
        json={"question": "", "top_k": 5},
    )

    assert response.status_code == 422
