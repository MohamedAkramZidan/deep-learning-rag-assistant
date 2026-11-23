import random
import time

from google.genai import errors
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)


def generate_with_retry(
    client,
    model: str,
    prompt: str,
    config: dict | None = None,
    max_retries: int = 3,
):
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )

        except errors.ServerError as exc:
            if exc.code not in (500, 503) or attempt == max_retries:
                raise

            delay = (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(delay)


def generate_with_groq_retry(
    client,
    model: str,
    prompt: str,
    max_retries: int = 3,
):
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )

        except (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
        ):
            if attempt == max_retries:
                raise

            delay = (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(delay)

        except APIStatusError as exc:
            if exc.status_code not in (500, 502, 503, 504):
                raise

            if attempt == max_retries:
                raise

            delay = (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(delay)
