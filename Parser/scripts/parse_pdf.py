
import json
from dataclasses import asdict
from pathlib import Path

from parser.pdf_parser import PDFParser


def main():
    parser = PDFParser()

    pdf_path = Path("data/Hands-On-Machine-Learning.pdf")
    output_path = Path("data/parsed_document.json")

    document = parser.parse(pdf_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            asdict(document),
            file,
            ensure_ascii=False,
            indent=4,
        )

    print(f"PDF parsed successfully: {document.filename}")
    print(f"Total pages: {document.metadata.total_pages}")
    print(f"Extracted pages: {len(document.pages)}")
    print(f"JSON saved to: {output_path}")


if __name__ == "__main__":
    main()
