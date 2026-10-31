from qdrant_client import QdrantClient

from rag_app.core.config import settings


def create_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value(),
        timeout=100,
    )