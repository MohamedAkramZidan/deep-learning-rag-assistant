from functools import lru_cache

from google import genai
from openai import OpenAI

from rag_app.core.config import settings


@lru_cache(maxsize=1)
def create_groq_client() -> OpenAI:
    if settings.groq_api_key is None:
        raise ValueError("GROQ_API_KEY must be configured.")

    return OpenAI(
        api_key=settings.groq_api_key.get_secret_value(),
        base_url="https://api.groq.com/openai/v1",
    )


@lru_cache(maxsize=1)
def create_gemini_client() -> genai.Client:
    if settings.gemini_api_key is None:
        raise ValueError("GEMINI_API_KEY must be configured.")

    return genai.Client(
        api_key=settings.gemini_api_key.get_secret_value()
    )