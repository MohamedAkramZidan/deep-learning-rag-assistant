
from google import genai

from rag_app.core.config import settings
from rag_app.generation.prompts import build_rag_prompt
from rag_app.schemas.response import SourceDocument


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
        if settings.gemini_api_key is None:
            raise ValueError(
                "GEMINI_API_KEY must be configured to generate answers."
            )

        client = genai.Client(
            api_key=settings.gemini_api_key.get_secret_value()
        )

    prompt = build_rag_prompt(question, sources)

    response = client.models.generate_content(
        model=model_name or settings.gemini_model,
        contents=prompt,
    )

    answer = response.text

    if not answer or not answer.strip():
        raise RuntimeError("The language model returned an empty answer.")

    return answer.strip()
