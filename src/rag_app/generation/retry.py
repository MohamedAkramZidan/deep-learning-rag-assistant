import random
import time

from google.genai import errors


def generate_with_retry(
    client,
    model: str,
    prompt: str,
    max_retries: int = 3,
):
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=prompt,
            )

        except errors.ServerError as exc:
            if exc.code not in (500, 503) or attempt == max_retries:
                raise

            delay = (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(delay)