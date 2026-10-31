
from unittest.mock import Mock

import pytest

from rag_app.generation.llm import generate_answer
from rag_app.generation.prompts import build_rag_prompt
from rag_app.schemas.response import SourceDocument


def make_source() -> SourceDocument:
    return SourceDocument(
        id="source-1",
        title="Introduction to Deep Learning",
        content="Deep learning uses neural networks with multiple layers.",
    )


def test_prompt_includes_question_and_source():
    prompt = build_rag_prompt(
        "What is deep learning?",
        [make_source()],
    )

    assert "What is deep learning?" in prompt
    assert "Deep learning uses neural networks" in prompt
    assert "[Source ID: source-1]" in prompt
    assert "Do not invent facts" in prompt


def test_prompt_handles_no_sources():
    prompt = build_rag_prompt("What is deep learning?", [])

    assert "No relevant source documents" in prompt


def test_generate_answer_uses_gemini_client():
    client = Mock()
    client.models.generate_content.return_value.text = (
        "Deep learning uses neural networks. [Source ID: source-1]"
    )

    answer = generate_answer(
        question="What is deep learning?",
        sources=[make_source()],
        client=client,
        model_name="test-model",
    )

    assert answer.startswith("Deep learning")
    client.models.generate_content.assert_called_once()

    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert "source-1" in kwargs["contents"]


def test_generate_answer_rejects_empty_question():
    with pytest.raises(ValueError):
        generate_answer("  ", [make_source()], client=Mock())


def test_generate_answer_rejects_empty_model_response():
    client = Mock()
    client.models.generate_content.return_value.text = "  "

    with pytest.raises(RuntimeError, match="empty answer"):
        generate_answer(
            "What is deep learning?",
            [make_source()],
            client=client,
        )
