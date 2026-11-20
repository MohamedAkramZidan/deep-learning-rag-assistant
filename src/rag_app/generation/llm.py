
from rag_app.core.config import settings
from rag_app.generation.client import create_gemini_client
from rag_app.generation.prompts import build_rag_prompt
from rag_app.schemas.response import SourceDocument
from rag_app.generation.retry import generate_with_retry


def generate_answer(
    question: str,
    sources: list[SourceDocument],
    client=None,
    model_name: str | None = None,
) -> str:
    """Generate a grounded answer using Gemini."""

    if not question.strip():
        raise ValueError("Question cannot be empty or whitespace.")

    if client is None:
        client = create_gemini_client()

    prompt = build_rag_prompt(question, sources)

    response = generate_with_retry(
        client=client,
        model=model_name or settings.gemini_model,
        prompt=prompt,
        config={
            "temperature": 0.1,
        },
    )

    answer = response.text

    if not answer or not answer.strip():
        raise RuntimeError("The language model returned an empty answer.")

    return answer.strip()