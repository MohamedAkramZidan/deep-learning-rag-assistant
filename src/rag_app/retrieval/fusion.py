
from rag_app.schemas.response import SourceDocument


DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[SourceDocument]],
    top_k: int = 5,
    k: int = DEFAULT_RRF_K,
) -> list[SourceDocument]:
    """Combine ranked retrieval results using Reciprocal Rank Fusion."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if k < 1:
        raise ValueError("k must be at least 1")

    scores: dict[str, float] = {}
    documents: dict[str, SourceDocument] = {}

    for ranked_list in ranked_lists:
        seen_in_list: set[str] = set()

        for rank, document in enumerate(ranked_list, start=1):
            if document.id in seen_in_list:
                continue

            seen_in_list.add(document.id)
            scores[document.id] = (
                scores.get(document.id, 0.0)
                + 1.0 / (k + rank)
            )
            documents.setdefault(document.id, document)

    ranked_ids = sorted(
        scores,
        key=lambda document_id: scores[document_id],
        reverse=True,
    )

    results = []

    for document_id in ranked_ids[:top_k]:
        document = documents[document_id].model_copy(deep=True)
        document.score = scores[document_id]
        results.append(document)

    return results
