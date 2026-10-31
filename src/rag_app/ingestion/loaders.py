
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "documents.json"


def load_documents(
    file_path: str | Path = DEFAULT_DATASET_PATH,
) -> list[dict[str, Any]]:
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    if not isinstance(documents, list):
        raise ValueError("Dataset must contain a JSON array.")

    if not documents:
        raise ValueError("Dataset must contain at least one document.")

    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"Document {index} must be a JSON object.")

        for field in ("title", "content"):
            value = document.get(field)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Document {index} must have a non-empty '{field}'."
                )

    return documents
