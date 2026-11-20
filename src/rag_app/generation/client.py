from functools import lru_cache

from google import genai

from rag_app.core.config import settings


@lru_cache(maxsize=1)
def create_gemini_client() -> genai.Client:
    if settings.gemini_api_key is None:
        raise ValueError(
            "GEMINI_API_KEY must be configured to generate answers."
        )

    return genai.Client(
        api_key=settings.gemini_api_key.get_secret_value()
    )