from functools import lru_cache

from qdrant_client import QdrantClient

from rag_app.core.config import settings


@lru_cache(maxsize=1)
def create_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value(),
        timeout=100,
    )