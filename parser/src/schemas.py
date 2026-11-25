from dataclasses import dataclass, field


@dataclass
class PageContent:
    page_number: int
    text: str


@dataclass
class DocumentMetadata:
    title: str | None = None
    author: str | None = None
    total_pages: int = 0


@dataclass
class ParsedDocument:
    document_id: str
    filename: str
    metadata: DocumentMetadata
    pages: list[PageContent] = field(default_factory=list)