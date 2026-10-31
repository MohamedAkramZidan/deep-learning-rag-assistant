
import json

import pytest

from rag_app.ingestion.loaders import load_documents


def test_load_documents(tmp_path):
    dataset = [
        {
            "title": "Deep Learning",
            "content": "Deep learning uses neural networks.",
            "category": "Fundamentals",
        }
    ]

    file_path = tmp_path / "documents.json"
    file_path.write_text(json.dumps(dataset), encoding="utf-8")

    documents = load_documents(file_path)

    assert len(documents) == 1
    assert documents[0]["title"] == "Deep Learning"


def test_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing.json")


def test_empty_dataset(tmp_path):
    file_path = tmp_path / "documents.json"
    file_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="at least one"):
        load_documents(file_path)


def test_invalid_document_content(tmp_path):
    file_path = tmp_path / "documents.json"
    file_path.write_text(
        json.dumps([{"title": "Deep Learning", "content": " "}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="content"):
        load_documents(file_path)


def test_invalid_json(tmp_path):
    file_path = tmp_path / "documents.json"
    file_path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_documents(file_path)
