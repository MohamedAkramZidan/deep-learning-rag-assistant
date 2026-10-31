
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "data" / "documents.json"


def test_dataset_file_exists():
    assert DATASET_PATH.is_file()


def test_dataset_is_valid_json_array():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    assert isinstance(documents, list)
    assert documents, "Dataset must contain at least one document"


def test_documents_have_required_fields():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    for index, document in enumerate(documents):
        assert isinstance(document, dict), f"Document {index} must be an object"
        assert isinstance(document.get("title"), str) and document["title"].strip()
        assert isinstance(document.get("content"), str) and document["content"].strip()
