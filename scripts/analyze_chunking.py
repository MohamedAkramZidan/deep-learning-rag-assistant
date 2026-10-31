
from qdrant_client import models

from rag_app.retrieval.vector_store import create_qdrant_client

COLLECTION_NAME = "deep_learning_documents"
STRATEGIES = ("fixed", "sentence", "paragraph")
PAGE_SIZE = 100


def get_chunks_for_strategy(client, strategy: str) -> list[dict]:
    chunks = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="chunk_strategy",
                        match=models.MatchValue(value=strategy),
                    )
                ]
            ),
            limit=PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        chunks.extend(
            point.payload
            for point in points
            if point.payload is not None
        )

        if next_offset is None:
            break

        offset = next_offset

    return chunks


def analyze_chunking_effectiveness() -> None:
    client = create_qdrant_client()

    print("CHUNKING STRATEGY ANALYSIS")
    print("=" * 40)

    for strategy in STRATEGIES:
        chunks = get_chunks_for_strategy(client, strategy)

        print(f"\n{strategy.upper()} STRATEGY:")
        print(f"  Total chunks: {len(chunks)}")

        if not chunks:
            print("  No chunks found. Run the ingestion script first.")
            continue

        texts = [
            payload.get("chunk", "")
            for payload in chunks
        ]

        char_sizes = [len(text) for text in texts]
        word_sizes = [len(text.split()) for text in texts]

        print(f"  Average characters: {sum(char_sizes) / len(char_sizes):.0f}")
        print(f"  Character range: {min(char_sizes)}-{max(char_sizes)}")
        print(f"  Average words: {sum(word_sizes) / len(word_sizes):.1f}")
        print(f"  Word range: {min(word_sizes)}-{max(word_sizes)}")


if __name__ == "__main__":
    analyze_chunking_effectiveness()
