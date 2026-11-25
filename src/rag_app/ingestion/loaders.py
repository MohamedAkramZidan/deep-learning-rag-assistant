import json
from pathlib import Path


def load_documents(file_path: str | Path) -> dict:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    with file_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, dict):
        raise ValueError("Dataset must be a JSON object.")

    if not dataset.get("document_id"):
        raise ValueError("Dataset must contain a document_id.")

    pages = dataset.get("pages")

    if not isinstance(pages, list) or not pages:
        raise ValueError("Dataset must contain at least one page in 'pages'.")

    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError(f"Page {index} must be a JSON object.")

        if not isinstance(page.get("text"), str) or not page["text"].strip():
            raise ValueError(f"Page {index} must have non-empty content.")

        if (
            not isinstance(page.get("chapter_title"), str)
            or not page["chapter_title"].strip()
        ):
            raise ValueError(
                f"Page {index} must have a non-empty 'chapter_title'."
            )

    return dataset