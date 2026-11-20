
import pytest
from unittest.mock import MagicMock

from google.genai import errors

from rag_app.generation.retry import generate_with_retry


@pytest.fixture
def client():
    return MagicMock()


def test_success_first_attempt(client):
    expected_response = MagicMock()
    client.models.generate_content.return_value = expected_response

    response = generate_with_retry(
        client=client,
        model="gemini-test",
        prompt="Test prompt",
    )

    assert response == expected_response
    assert client.models.generate_content.call_count == 1


def test_retry_after_503(client, monkeypatch):
    expected_response = MagicMock()

    client.models.generate_content.side_effect = [
        errors.ServerError(503, {"error": {"message": "Unavailable"}}),
        expected_response,
    ]

    monkeypatch.setattr("time.sleep", lambda _: None)

    response = generate_with_retry(
        client=client,
        model="gemini-test",
        prompt="Test prompt",
    )

    assert response == expected_response
    assert client.models.generate_content.call_count == 2


def test_raises_after_max_retries(client, monkeypatch):
    client.models.generate_content.side_effect = errors.ServerError(
        503,
        {"error": {"message": "Unavailable"}},
    )

    monkeypatch.setattr("time.sleep", lambda _: None)

    with pytest.raises(errors.ServerError):
        generate_with_retry(
            client=client,
            model="gemini-test",
            prompt="Test prompt",
            max_retries=3,
        )

    assert client.models.generate_content.call_count == 4


def test_does_not_retry_on_400(client):
    client.models.generate_content.side_effect = errors.ClientError(
        400,
        {"error": {"message": "Bad request"}},
    )

    with pytest.raises(errors.ClientError):
        generate_with_retry(
            client=client,
            model="gemini-test",
            prompt="Test prompt",
        )

    assert client.models.generate_content.call_count == 1
