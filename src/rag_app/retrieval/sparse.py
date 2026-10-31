
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from rag_app.schemas.response import SourceDocument
from rag_app.retrieval.vector_store import create_qdrant_client


COLLECTION_NAME = "deep_learning_documents"
SPARSE_MODEL_NAME = "Qdrant/bm25"
SPARSE_VECTOR_NAME = "bm25"
DEFAULT_TOP_K = 5
MAX_TOP_K = 20


def load_sparse_encoder() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def search_sparse(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    client: QdrantClient | None = None,
    encoder: SparseTextEmbedding | None = None,
    collection_name: str = COLLECTION_NAME,
) -> list[SourceDocument]:
    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")

    qdrant = client or create_qdrant_client()
    sparse_encoder = encoder or load_sparse_encoder()

    query_embedding = next(sparse_encoder.query_embed(query))

    response = qdrant.query_points(
        collection_name=collection_name,
        query=models.SparseVector(
            indices=query_embedding.indices.tolist(),
            values=query_embedding.values.tolist(),
        ),
        using=SPARSE_VECTOR_NAME,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    results = []

    for point in response.points:
        payload = point.payload or {}

        results.append(
            SourceDocument(
                id=str(point.id),
                title=payload.get("title", "Untitled"),
                content=payload.get("chunk", ""),
                score=float(point.score),
                metadata={
                    key: value
                    for key, value in payload.items()
                    if key not in {"title", "content", "chunk"}
                },
            )
        )

    return results
