
from sentence_transformers import CrossEncoder

from rag_app.schemas.response import SourceDocument


RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CANDIDATE_LIMIT = 20


def load_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL_NAME)


def rerank_documents(
    query: str,
    documents: list[SourceDocument],
    top_k: int = 5,
    reranker: CrossEncoder | None = None,
) -> list[SourceDocument]:
    """Rerank retrieved documents using query-document relevance scores."""

    if not query.strip():
        raise ValueError("Query cannot be empty or whitespace.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not documents:
        return []

    model = reranker or load_reranker()

    pairs = [
        (query, f"{document.title}\n{document.content}")
        for document in documents
    ]

    scores = model.predict(pairs)

    ranked_documents = []

    for document, score in zip(documents, scores, strict=True):
        ranked = document.model_copy(deep=True)
        ranked.score = float(score)
        ranked_documents.append(ranked)

    ranked_documents.sort(
        key=lambda document: document.score,
        reverse=True,
    )

    return ranked_documents[:top_k]
