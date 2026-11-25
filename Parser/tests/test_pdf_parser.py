
import pymupdf
import pytest

from parser.pdf_parser import PDFParser


@pytest.fixture
def sample_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as pdf:
        page1 = pdf.new_page()
        page1.insert_text((72, 72), "Introduction to Deep Learning")

        page2 = pdf.new_page()
        page2.insert_text((72, 72), "Neural Networks and CNNs")

        pdf.set_metadata({
            "title": "Deep Learning",
            "author": "Test Author",
        })

        pdf.save(pdf_path)

    return pdf_path


def test_parse_pdf_returns_document(sample_pdf):
    parser = PDFParser()

    document = parser.parse(sample_pdf)

    assert document.filename == "sample.pdf"
    assert document.metadata.title == "Deep Learning"
    assert document.metadata.author == "Test Author"
    assert document.metadata.total_pages == 2


def test_extracts_page_text(sample_pdf):
    parser = PDFParser()

    document = parser.parse(sample_pdf)

    assert len(document.pages) == 2
    assert document.pages[0].page_number == 1
    assert "Introduction to Deep Learning" in document.pages[0].text
    assert document.pages[1].page_number == 2
    assert "Neural Networks and CNNs" in document.pages[1].text


def test_document_id_is_sha256(sample_pdf):
    parser = PDFParser()

    document = parser.parse(sample_pdf)

    assert len(document.document_id) == 64
    assert all(
        char in "0123456789abcdef"
        for char in document.document_id
    )


def test_missing_file_raises_error(tmp_path):
    parser = PDFParser()

    with pytest.raises(FileNotFoundError):
        parser.parse(tmp_path / "missing.pdf")


def test_non_pdf_file_raises_error(tmp_path):
    parser = PDFParser()

    text_file = tmp_path / "document.txt"
    text_file.write_text("Not a PDF")

    with pytest.raises(ValueError, match="PDF"):
        parser.parse(text_file)


def test_corrupted_pdf_raises_error(tmp_path):
    parser = PDFParser()

    corrupted_pdf = tmp_path / "corrupted.pdf"
    corrupted_pdf.write_bytes(b"This is not a valid PDF")

    with pytest.raises(ValueError, match="Invalid or corrupted PDF"):
        parser.parse(corrupted_pdf)
