
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from rag_app.core.logging import get_logger
from rag_app.ingestion.chunking import (
    fixed_size_chunks,
    paragraph_chunks,
    sentence_chunks,
)
from rag_app.retrieval.embeddings import load_dense_encoder
from rag_app.retrieval.sparse import (
    SPARSE_VECTOR_NAME,
    load_sparse_encoder,
)
from rag_app.retrieval.vector_store import create_qdrant_client


logger = get_logger(__name__)

COLLECTION_NAME = "deep_learning_documents"
DEFAULT_DATASET_PATH = Path("data/cleaned_document.json")

BATCH_SIZE = 16
VECTOR_SIZE = 384

STRATEGIES = {
    "fixed": fixed_size_chunks,
    "sentence": sentence_chunks,
    "paragraph": paragraph_chunks,
}


def load_documents(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> dict:
    with Path(dataset_path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict) or not isinstance(
        data.get("pages"), list
    ):
        raise ValueError("Invalid dataset: expected a pages list.")

    pages = [
        page
        for page in data["pages"]
        if isinstance(page, dict)
        and isinstance(page.get("text"), str)
        and page["text"].strip()
    ]

    data["pages"] = pages

    logger.info("Loaded %d pages", len(pages))

    return data


def ensure_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> None:

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

            logger.info("Added sparse vector: %s", SPARSE_VECTOR_NAME)

    for field in (
        "document_id",
        "filename",
        "title",
        "author",
        "page_number",
        "chapter",
        "chapter_title",
        "chunk_strategy",
    ):
        schema = (
            models.PayloadSchemaType.INTEGER
            if field in ("page_number", "chapter")
            else models.PayloadSchemaType.KEYWORD
        )

        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=schema,
            wait=True,
        )


def build_points(
    dataset: dict,
    encoder,
    batch_size: int = BATCH_SIZE,
    sparse_encoder=None,
) -> list[models.PointStruct]:

    if sparse_encoder is None:
        sparse_encoder = load_sparse_encoder()

    document_id = dataset.get("document_id", "unknown")
    filename = dataset.get("filename")
    metadata = dataset.get("metadata", {})

    points = []

    for page in dataset["pages"]:

        page_number = page.get("page_number")

        for strategy_name, chunker in STRATEGIES.items():

            chunks = chunker(page["text"])

            if not chunks:
                continue

            dense_vectors = encoder.encode(
                chunks,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            sparse_vectors = list(
                sparse_encoder.embed(
                    chunks,
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
                        f"{document_id}::{page_number}::"
                        f"{strategy_name}::{chunk_index}",
                    )
                )

                payload = {
                    "document_id": document_id,
                    "filename": filename,
                    "title": metadata.get("title"),
                    "author": metadata.get("author"),
                    "page_number": page_number,
                    "chapter": page.get("chapter"),
                    "chapter_title": page.get("chapter_title"),
                    "chunk": chunk,
                    "chunk_strategy": strategy_name,
                    "chunk_index": chunk_index,
                }

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
                        payload=payload,
                    )
                )

    logger.info("Built %d points", len(points))

    return points



def remove_stale_points(
    client: QdrantClient,
    points: list[models.PointStruct],
    document_id: str,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = BATCH_SIZE,
) -> int:

    desired_ids = {str(point.id) for point in points}

    stale_ids = []
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
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
                points=stale_ids[start:start + batch_size]
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
            "pages": len(dataset["pages"]),
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
        document_id=dataset["document_id"],
        collection_name=collection_name,
        batch_size=batch_size,
    )

    logger.info(
        "Ingestion complete: %d pages, %d points uploaded, "
        "%d stale points removed",
        len(dataset["pages"]),
        len(points),
        stale_count,
    )

    return {
        "pages": len(dataset["pages"]),
        "points_uploaded": len(points),
        "stale_points_removed": stale_count,
    }
