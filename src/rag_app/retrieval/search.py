
from qdrant_client import QdrantClient

from rag_app.schemas.response import SourceDocument
from rag_app.retrieval.embeddings import load_dense_encoder
from rag_app.retrieval.vector_store import create_qdrant_client

from rag_app.retrieval.fusion import reciprocal_rank_fusion
from rag_app.retrieval.sparse import search_sparse


COLLECTION_NAME = "deep_learning_documents"
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

DENSE_STRATEGIES = {"fixed", "sentence", "paragraph"}


def search_dense(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    strategy: str = "fixed",
    client: QdrantClient | None = None,
    encoder=None,
    collection_name: str = COLLECTION_NAME,
) -> list[SourceDocument]:
    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")

    if strategy not in DENSE_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose one of {sorted(DENSE_STRATEGIES)}."
        )

    qdrant = client or create_qdrant_client()
    dense_encoder = encoder or load_dense_encoder()

    query_vector = dense_encoder.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    response = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=strategy,
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

def search_hybrid(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    strategy: str = "fixed",
    client: QdrantClient | None = None,
    encoder=None,
    sparse_encoder=None,
    collection_name: str = COLLECTION_NAME,
) -> list[SourceDocument]:
    """Combine dense and BM25 retrieval using RRF."""

    if not query.strip():
        raise ValueError("Query cannot be empty or whitespace.")

    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")

    if strategy not in DENSE_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {sorted(DENSE_STRATEGIES)}."
        )

    qdrant = client or create_qdrant_client()

    dense_results = search_dense(
        query=query,
        top_k=top_k,
        strategy=strategy,
        client=qdrant,
        encoder=encoder,
        collection_name=collection_name,
    )

    sparse_results = search_sparse(
        query=query,
        top_k=top_k,
        client=qdrant,
        encoder=sparse_encoder,
        collection_name=collection_name,
    )

    return reciprocal_rank_fusion(
        ranked_lists=[dense_results, sparse_results],
        top_k=top_k,
    )
