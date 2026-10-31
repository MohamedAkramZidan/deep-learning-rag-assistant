
import logging

from fastapi import APIRouter, HTTPException

from rag_app.schemas.request import RAGRequest
from rag_app.schemas.response import RAGResponse
from rag_app.services.rag_service import answer_question

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=RAGResponse)
def query_rag(request: RAGRequest) -> RAGResponse:
    try:
        return answer_question(request)

    except Exception as exc:
        logger.exception("Failed to process RAG query")
        raise HTTPException(
            status_code=500,
            detail="Failed to process the RAG request.",
        ) from exc
