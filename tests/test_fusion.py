
import pytest

from rag_app.retrieval.fusion import reciprocal_rank_fusion
from rag_app.schemas.response import SourceDocument


def make_document(document_id: str) -> SourceDocument:
    return SourceDocument(
        id=document_id,
        title=f"Document {document_id}",
        content=f"Content for {document_id}",
    )


def test_fuses_rankings_and_promotes_shared_documents():
    dense = [
        make_document("A"),
        make_document("B"),
        make_document("C"),
    ]
    sparse = [
        make_document("B"),
        make_document("A"),
        make_document("D"),
    ]

    results = reciprocal_rank_fusion(
        ranked_lists=[dense, sparse],
        top_k=4,
    )

    assert [doc.id for doc in results] == ["A", "B", "C", "D"]
    assert results[0].score == pytest.approx(
        1 / 61 + 1 / 62
    )
    assert results[1].score == pytest.approx(
        1 / 62 + 1 / 61
    )


def test_returns_empty_for_empty_rankings():
    assert reciprocal_rank_fusion([[], []]) == []


def test_respects_top_k():
    results = reciprocal_rank_fusion(
        [[make_document("A"), make_document("B")]],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].id == "A"


def test_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], top_k=0)


def test_rejects_invalid_rrf_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], k=0)


def test_does_not_double_count_duplicate_in_one_list():
    results = reciprocal_rank_fusion(
        [[make_document("A"), make_document("A")]],
    )

    assert len(results) == 1
    assert results[0].score == pytest.approx(1 / 61)
