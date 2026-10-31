from sentence_transformers import SentenceTransformer

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_dense_encoder() -> SentenceTransformer:
    return SentenceTransformer(DENSE_MODEL_NAME)