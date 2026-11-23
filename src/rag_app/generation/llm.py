
import logging

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from rag_app.core.config import settings
from rag_app.generation.client import (
    create_gemini_client,
    create_groq_client,
)
from rag_app.generation.prompts import build_rag_prompt
from rag_app.generation.retry import (
    generate_with_groq_retry,
    generate_with_retry,
)
from rag_app.schemas.response import SourceDocument


logger = logging.getLogger(__name__)

GROQ_RETRYABLE_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    APIStatusError,
)


def generate_answer(
    question: str,
    sources: list[SourceDocument],
    client=None,
    model_name: str | None = None,
) -> str:

    if not question or not question.strip():
        raise ValueError("Question cannot be empty or whitespace.")

    prompt = build_rag_prompt(question, sources)

    try:
        groq_client = client or create_groq_client()

        response = generate_with_groq_retry(
            client=groq_client,
            model=settings.groq_model,
            prompt=prompt,
        )

        answer = response.choices[0].message.content

    except GROQ_RETRYABLE_ERRORS as exc:
        if isinstance(exc, APIStatusError) and exc.status_code not in (
            429, 500, 502, 503, 504
        ):
            raise

        logger.warning(
            "Groq unavailable after retries. Switching to Gemini."
        )

        gemini_client = create_gemini_client()

        response = generate_with_retry(
            client=gemini_client,
            model=model_name or settings.gemini_model,
            prompt=prompt,
            config={
                "temperature": 0.1,
                "max_output_tokens": 1024,
            },
        )

        answer = response.text

    if not answer or not answer.strip():
        raise RuntimeError("The language model returned an empty answer.")

    return answer.strip()
