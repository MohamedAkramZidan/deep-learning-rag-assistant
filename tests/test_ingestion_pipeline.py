
from unittest.mock import Mock

import numpy as np
import pytest

from types import SimpleNamespace

from rag_app.ingestion.pipeline import (
    STRATEGIES,
    build_points,
    run_ingestion,
)




class FakeEncoder:
    def encode(
        self,
        texts,
        batch_size,
        normalize_embeddings,
        show_progress_bar,
    ):
        return np.ones((len(texts), 384), dtype=np.float32)

class FakeSparseEncoder:
    def embed(self, texts, batch_size=256, **kwargs):
        for text in texts:
            tokens = set(text.lower().split())
            indices = sorted(
                abs(hash(token)) % 1_000_000
                for token in tokens
            )

            yield SimpleNamespace(
                indices=np.asarray(indices, dtype=np.int64),
                values=np.ones(len(indices), dtype=np.float32),
            )

def test_build_points_creates_all_strategies():
    dataset = [
        {
            "title": "Deep Learning",
            "content": (
                "Deep learning uses neural networks. "
                "Models learn patterns from data."
            ),
            "category": "Fundamentals",
            "topic": "Introduction",
            "difficulty": "Beginner",
            "tags": ["deep learning"],
        }
    ]

    points = build_points(
        dataset,
        FakeEncoder(),
        sparse_encoder=FakeSparseEncoder(),
    )

    assert points
    assert {
        point.payload["chunk_strategy"] for point in points
    } == set(STRATEGIES)

    for point in points:
        print("Vector:", point.vector)
        print("Type:", type(point.vector))
    assert all(
        len(next(iter(point.vector.values()))) == 384
        for point in points
    )


def test_build_points_uses_deterministic_ids():
    dataset = [
        {
            "title": "Deep Learning",
            "content": "Deep learning uses neural networks.",
        }
    ]

    first = build_points(dataset, FakeEncoder())
    second = build_points(dataset, FakeEncoder())

    assert [point.id for point in first] == [
        point.id for point in second
    ]


def test_run_ingestion_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        run_ingestion(
            batch_size=0,
            client=Mock(),
            encoder=FakeEncoder(),
        )


def test_run_ingestion_uploads_points(tmp_path):
    dataset_path = tmp_path / "documents.json"
    dataset_path.write_text(
        '[{"title": "Deep Learning", '
        '"content": "Deep learning uses neural networks."}]',
        encoding="utf-8",
    )

    client = Mock()
    client.collection_exists.return_value = True

    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                sparse_vectors={}
            )
        )
    )
    client.scroll.return_value = ([], None)

    result = run_ingestion(
        dataset_path=dataset_path,
        collection_name="test_collection",
        batch_size=16,
        client=client,
        encoder=FakeEncoder(),
        sparse_encoder=FakeSparseEncoder(),
    )

    assert result["documents"] == 1
    assert result["points_uploaded"] > 0
    assert result["stale_points_removed"] == 0

    client.upload_points.assert_called_once()
    client.create_vector_name.assert_called_once()
    assert (
        client.upload_points.call_args.kwargs["collection_name"]
        == "test_collection"
    )