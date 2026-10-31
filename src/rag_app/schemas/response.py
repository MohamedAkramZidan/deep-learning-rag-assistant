
from pydantic import BaseModel, ConfigDict, Field


class SourceDocument(BaseModel):
    """A document retrieved from the knowledge base."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        description="Unique identifier of the retrieved document."
    )

    title: str = Field(
        description="Title of the source document."
    )

    content: str = Field(
        description="Content of the retrieved document."
    )

    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Metadata associated with the document."
    )

    score: float | None = Field(
        default=None,
        description="Retrieval relevance score, if available."
    )


class RAGResponse(BaseModel):
    """Response returned by the RAG API."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        description="Answer generated using retrieved context."
    )

    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Source documents supporting the answer."
    )

    model: str | None = Field(
        default=None,
        description="Name of the language model used to generate the answer."
    )
