
import hashlib
from pathlib import Path

import pymupdf

from .base import BaseParser
from schemas import (
    DocumentMetadata,
    PageContent,
    ParsedDocument,
)


class PDFParser(BaseParser):

    def parse(self, file_path: Path) -> ParsedDocument:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Expected a file: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError("File must have a PDF extension.")

        try:
            with pymupdf.open(file_path) as pdf:
                if pdf.is_encrypted:
                    raise ValueError("Encrypted PDFs are not supported.")

                pages = [
                    PageContent(
                        page_number=page_number,
                        text=page.get_text("text").strip(),
                    )
                    for page_number, page in enumerate(pdf, start=1)
                    if page.get_text("text").strip()
                ]

                metadata = pdf.metadata or {}

                document_id = hashlib.sha256(
                    file_path.read_bytes()
                ).hexdigest()

                return ParsedDocument(
                    document_id=document_id,
                    filename=file_path.name,
                    metadata=DocumentMetadata(
                        title=metadata.get("title"),
                        author=metadata.get("author"),
                        total_pages=len(pdf),
                    ),
                    pages=pages,
                )

        except pymupdf.FileDataError as exc:
            raise ValueError("Invalid or corrupted PDF.") from exc
