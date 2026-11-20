from statistics import median
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from rag_app.generation.llm import generate_answer
from rag_app.generation.prompts import build_rag_prompt
from rag_app.retrieval.reranker import rerank_documents
from rag_app.retrieval.search import search_dense
from rag_app.retrieval.sparse import search_sparse
from rag_app.schemas.response import SourceDocument


LATENCY_LIMIT_SECONDS = 0.5


def measure_median(operation, iterations: int = 20) -> float:
    timings = []
    for _ in range(iterations):
        started = perf_counter()
        operation()
        timings.append(perf_counter() - started)
    return median(timings)


def make_source() -> SourceDocument:
    return SourceDocument(
        id="source-1",
        title="Deep Learning",
        content="Deep learning uses multilayer neural networks.",
        score=0.9,
    )


def test_prompt_construction_latency():
    latency = measure_median(
        lambda: build_rag_prompt("What is deep learning?", [make_source()])
    )

    print(f"prompt construction median latency: {latency * 1000:.3f} ms")
    assert latency < LATENCY_LIMIT_SECONDS


def test_dense_search_component_latency():
    encoder = Mock()
    encoder.encode.return_value = np.ones(384, dtype=np.float32)

    client = Mock()
    client.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(
                id="source-1",
                score=0.9,
                payload={
                    "title": "Deep Learning",
                    "chunk": "Deep learning uses multilayer neural networks.",
                },
            )
        ]
    )

    latency = measure_median(
        lambda: search_dense(
            "What is deep learning?",
            top_k=5,
            client=client,
            encoder=encoder,
        )
    )

    print(f"dense search component median latency: {latency * 1000:.3f} ms")
    assert latency < LATENCY_LIMIT_SECONDS


def test_sparse_search_component_latency():
    query_embedding = SimpleNamespace(
        indices=np.array([1, 4], dtype=np.int32),
        values=np.array([0.8, 0.5], dtype=np.float32),
    )
    encoder = Mock()
    encoder.query_embed.side_effect = lambda _query: iter([query_embedding])

    client = Mock()
    client.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(
                id="source-1",
                score=0.8,
                payload={
                    "title": "Deep Learning",
                    "chunk": "Deep learning uses multilayer neural networks.",
                },
            )
        ]
    )

    latency = measure_median(
        lambda: search_sparse(
            "What is deep learning?",
            top_k=5,
            client=client,
            encoder=encoder,
        )
    )

    print(f"sparse search component median latency: {latency * 1000:.3f} ms")
    assert latency < LATENCY_LIMIT_SECONDS


def test_reranking_component_latency():
    reranker = Mock()
    reranker.predict.return_value = np.array([0.95], dtype=np.float32)

    latency = measure_median(
        lambda: rerank_documents(
            "What is deep learning?",
            [make_source()],
            top_k=5,
            reranker=reranker,
        )
    )

    print(f"reranking component median latency: {latency * 1000:.3f} ms")
    assert latency < LATENCY_LIMIT_SECONDS


def test_llm_generation_component_latency():
    llm_client = Mock()
    llm_client.models.generate_content.return_value.text = (
        "Deep learning uses neural networks. [Source ID: source-1]"
    )

    latency = measure_median(
        lambda: generate_answer(
            "What is deep learning?",
            [make_source()],
            client=llm_client,
            model_name="test-model",
        )
    )

    print(f"LLM generation component median latency: {latency * 1000:.3f} ms")
    assert latency < LATENCY_LIMIT_SECONDS
