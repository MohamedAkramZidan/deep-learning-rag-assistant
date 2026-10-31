
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RAGRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        min_length=1,
        max_length=4000,
        description="The question to answer using the knowledge base.",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of documents to retrieve.",
    )

    retrieval_mode: Literal["dense", "sparse", "hybrid"] = Field(
        default="hybrid",
        description="Retrieval strategy to use.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value:
            raise ValueError("Question cannot be empty or whitespace.")
        return value
