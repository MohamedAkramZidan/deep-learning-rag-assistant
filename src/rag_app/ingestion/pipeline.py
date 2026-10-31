from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from rag_app.core.logging import get_logger
from rag_app.ingestion.chunking import (
    fixed_size_chunks,
    paragraph_chunks,
    sentence_chunks,
)
from rag_app.ingestion.loaders import (
    DEFAULT_DATASET_PATH,
    load_documents,
)
from rag_app.retrieval.embeddings import load_dense_encoder
from rag_app.retrieval.sparse import (
    SPARSE_VECTOR_NAME,
    load_sparse_encoder,
)
from rag_app.retrieval.vector_store import create_qdrant_client


logger = get_logger(__name__)

COLLECTION_NAME = "deep_learning_documents"
BATCH_SIZE = 16
VECTOR_SIZE = 384

STRATEGIES = {
    "fixed": fixed_size_chunks,
    "sentence": sentence_chunks,
    "paragraph": paragraph_chunks,
}


def ensure_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Create the collection or add a missing sparse vector."""

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                name: models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                )
                for name in STRATEGIES
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )
        logger.info("Created collection: %s", collection_name)

    else:
        collection_info = client.get_collection(collection_name)
        sparse_vectors = (
            collection_info.config.params.sparse_vectors or {}
        )

        if SPARSE_VECTOR_NAME not in sparse_vectors:
            client.create_vector_name(
                collection_name=collection_name,
                vector_name=SPARSE_VECTOR_NAME,
                vector_name_config=models.SparseVectorNameConfig(
                    sparse=models.SparseVectorConfig(
                        modifier=models.Modifier.IDF,
                    ),
                ),
            )
            logger.info(
                "Added sparse vector '%s' to '%s'",
                SPARSE_VECTOR_NAME,
                collection_name,
            )
        else:
            logger.info(
                "Sparse vector '%s' already exists in '%s'",
                SPARSE_VECTOR_NAME,
                collection_name,
            )

    for field in (
        "chunk_strategy",
        "category",
        "topic",
        "difficulty",
    ):
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )


def build_points(
    dataset: list[dict],
    encoder,
    batch_size: int = BATCH_SIZE,
    sparse_encoder=None,
) -> list[models.PointStruct]:
    if sparse_encoder is None:
        sparse_encoder = load_sparse_encoder()

    points = []

    for document in dataset:
        for strategy_name, chunker in STRATEGIES.items():
            chunks = chunker(document["content"])

            if not chunks:
                continue

            texts = [
                f"Title: {document['title']}\nContent: {chunk}"
                for chunk in chunks
            ]

            dense_vectors = encoder.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            sparse_vectors = list(
                sparse_encoder.embed(
                    texts,
                    batch_size=batch_size,
                )
            )

            if len(dense_vectors) != len(chunks):
                raise ValueError(
                    f"Dense encoder returned {len(dense_vectors)} vectors "
                    f"for {len(chunks)} chunks."
                )

            if len(sparse_vectors) != len(chunks):
                raise ValueError(
                    f"Sparse encoder returned {len(sparse_vectors)} vectors "
                    f"for {len(chunks)} chunks."
                )

            for chunk_index, (chunk, dense, sparse) in enumerate(
                zip(
                    chunks,
                    dense_vectors,
                    sparse_vectors,
                    strict=True,
                )
            ):
                point_id = str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{document['title']}::{strategy_name}::{chunk_index}",
                    )
                )

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            strategy_name: dense.tolist(),
                            SPARSE_VECTOR_NAME: models.SparseVector(
                                indices=sparse.indices.tolist(),
                                values=sparse.values.tolist(),
                            ),
                        },
                        payload={
                            **document,
                            "chunk": chunk,
                            "chunk_strategy": strategy_name,
                            "chunk_index": chunk_index,
                        },
                    )
                )

    logger.info("Built %d points", len(points))
    return points


def remove_stale_points(
    client: QdrantClient,
    points: list[models.PointStruct],
    collection_name: str = COLLECTION_NAME,
    batch_size: int = BATCH_SIZE,
) -> int:

    desired_ids = {str(point.id) for point in points}
    stale_ids = []
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )

        stale_ids.extend(
            record.id
            for record in records
            if str(record.id) not in desired_ids
        )

        if next_offset is None:
            break

        offset = next_offset

    for start in range(0, len(stale_ids), batch_size):
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=stale_ids[start : start + batch_size]
            ),
            wait=True,
        )

    if stale_ids:
        logger.info("Removed %d stale points", len(stale_ids))

    return len(stale_ids)


def run_ingestion(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = BATCH_SIZE,
    client: QdrantClient | None = None,
    encoder=None,
    sparse_encoder=None,
) -> dict[str, int]:

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    dataset = load_documents(dataset_path)

    qdrant = client or create_qdrant_client()
    dense_encoder = encoder or load_dense_encoder()
    sparse_model = sparse_encoder or load_sparse_encoder()

    ensure_collection(qdrant, collection_name)

    points = build_points(
        dataset=dataset,
        encoder=dense_encoder,
        batch_size=batch_size,
        sparse_encoder=sparse_model,
    )

    if not points:
        logger.warning(
            "No chunks generated; existing points were preserved."
        )
        return {
            "documents": len(dataset),
            "points_uploaded": 0,
            "stale_points_removed": 0,
        }

    qdrant.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=batch_size,
    )

    stale_count = remove_stale_points(
        client=qdrant,
        points=points,
        collection_name=collection_name,
        batch_size=batch_size,
    )

    logger.info(
        "Ingestion complete: %d documents, %d points uploaded, "
        "%d stale points removed",
        len(dataset),
        len(points),
        stale_count,
    )

    return {
        "documents": len(dataset),
        "points_uploaded": len(points),
        "stale_points_removed": stale_count,
    }
