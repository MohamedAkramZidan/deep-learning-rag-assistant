import json

import pytest

from rag_app.ingestion.loaders import load_documents


def sample_dataset():
    return {
        "document_id": "deep-learning-book",
        "filename": "deep-learning.pdf",
        "metadata": {
            "title": "Deep Learning",
            "author": "Test Author",
            "total_pages": 1,
            "cleaned_pages": 1,
        },
        "pages": [
            {
                "page_number": 1,
                "chapter": 1,
                "chapter_title": "Introduction",
                "text": "Deep learning uses neural networks.",
            }
        ],
    }


def write_dataset(tmp_path, dataset):
    file_path = tmp_path / "documents.json"
    file_path.write_text(json.dumps(dataset), encoding="utf-8")
    return file_path


def test_load_documents(tmp_path):
    file_path = write_dataset(tmp_path, sample_dataset())

    dataset = load_documents(file_path)

    assert dataset["document_id"] == "deep-learning-book"
    assert dataset["filename"] == "deep-learning.pdf"
    assert dataset["metadata"]["title"] == "Deep Learning"
    assert len(dataset["pages"]) == 1
    assert dataset["pages"][0]["text"] == (
        "Deep learning uses neural networks."
    )


def test_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing.json")


def test_empty_dataset(tmp_path):
    dataset = sample_dataset()
    dataset["pages"] = []

    file_path = write_dataset(tmp_path, dataset)

    with pytest.raises(ValueError, match="pages"):
        load_documents(file_path)


def test_invalid_document_content(tmp_path):
    dataset = sample_dataset()
    dataset["pages"][0]["text"] = " "

    file_path = write_dataset(tmp_path, dataset)

    with pytest.raises(ValueError, match="content"):
        load_documents(file_path)


def test_invalid_chapter_title(tmp_path):
    dataset = sample_dataset()
    dataset["pages"][0]["chapter_title"] = ""

    file_path = write_dataset(tmp_path, dataset)

    with pytest.raises(ValueError, match="chapter_title"):
        load_documents(file_path)


def test_invalid_json(tmp_path):
    file_path = tmp_path / "documents.json"
    file_path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_documents(file_path)