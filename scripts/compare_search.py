
from rag_app.retrieval.embeddings import load_dense_encoder
from rag_app.retrieval.vector_store import create_qdrant_client


COLLECTION_NAME = "deep_learning_documents"
STRATEGIES = ("fixed", "sentence", "paragraph")
TOP_K = 3

client = create_qdrant_client()
encoder = load_dense_encoder()


def compare_search_results(query: str) -> None:
    query_vector = encoder.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    print(f"\nQuery: '{query}'")

    for strategy in STRATEGIES:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            using=strategy,
            limit=TOP_K,
            with_payload=True,
        )

        print(f"\n--- {strategy.upper()} CHUNKING ---")

        if not results.points:
            print("No results found.")
            continue

        for rank, point in enumerate(results.points, start=1):
            payload = point.payload or {}

            title = payload.get("title", "Untitled")
            chunk = payload.get("chunk", "")
            category = payload.get("category", "Unknown")

            print(f"{rank}. {title}")
            print(f"   Category: {category}")
            print(f"   Score: {point.score:.3f}")
            print(f"   Chunk: {chunk[:200]}")

            if len(chunk) > 200:
                print("   ...")


def main() -> None:
    test_queries = [
        "What are artificial neural networks?",
        "How does deep learning learn features automatically?",
        "Applications of deep learning in computer vision and NLP",
    ]

    for query in test_queries:
        compare_search_results(query)


if __name__ == "__main__":
    main()
