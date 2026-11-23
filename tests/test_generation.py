
from unittest.mock import Mock

import pytest
from openai import APIConnectionError

from rag_app.generation.llm import generate_answer
from rag_app.generation.prompts import build_rag_prompt
from rag_app.schemas.response import SourceDocument


def make_source() -> SourceDocument:
    return SourceDocument(
        id="source-1",
        title="Introduction to Deep Learning",
        content="Deep learning uses neural networks with multiple layers.",
    )


def make_groq_response(content: str):
    return Mock(
        choices=[
            Mock(message=Mock(content=content))
        ]
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


def test_generate_answer_uses_groq_client(monkeypatch):
    client = Mock()
    client.chat.completions.create.return_value = make_groq_response(
        "Deep learning uses neural networks. [Source ID: source-1]"
    )

    monkeypatch.setattr(
        "rag_app.generation.llm.create_groq_client",
        lambda: client,
    )

    answer = generate_answer(
        question="What is deep learning?",
        sources=[make_source()],
    )

    assert answer.startswith("Deep learning")
    client.chat.completions.create.assert_called_once()

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"]
    assert "source-1" in kwargs["messages"][0]["content"]


def test_generate_answer_rejects_empty_question():
    with pytest.raises(ValueError):
        generate_answer("  ", [make_source()])


def test_generate_answer_rejects_empty_model_response(monkeypatch):
    client = Mock()
    client.chat.completions.create.return_value = make_groq_response("  ")

    monkeypatch.setattr(
        "rag_app.generation.llm.create_groq_client",
        lambda: client,
    )

    with pytest.raises(RuntimeError, match="empty answer"):
        generate_answer(
            "What is deep learning?",
            [make_source()],
        )
