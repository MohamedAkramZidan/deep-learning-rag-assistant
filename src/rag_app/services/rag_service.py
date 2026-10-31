
from rag_app.generation.llm import generate_answer
from rag_app.retrieval.reranker import rerank_documents
from rag_app.retrieval.search import (
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    search_dense,
    search_hybrid,
)
from rag_app.retrieval.sparse import search_sparse
from rag_app.schemas.request import RAGRequest
from rag_app.schemas.response import RAGResponse, SourceDocument


def retrieve_documents(
    request: RAGRequest,
    *,
    rerank: bool = False,
    strategy: str = "fixed",
    client=None,
    dense_encoder=None,
    sparse_encoder=None,
    reranker=None,
    collection_name: str = COLLECTION_NAME,
) -> list[SourceDocument]:
    """Retrieve relevant documents using the requested search mode."""

    candidate_limit = max(request.top_k, 10) if rerank else request.top_k

    if request.retrieval_mode == "dense":
        documents = search_dense(
            query=request.question,
            top_k=candidate_limit,
            strategy=strategy,
            client=client,
            encoder=dense_encoder,
            collection_name=collection_name,
        )
    elif request.retrieval_mode == "sparse":
        documents = search_sparse(
            query=request.question,
            top_k=candidate_limit,
            client=client,
            encoder=sparse_encoder,
            collection_name=collection_name,
        )
    else:
        documents = search_hybrid(
            query=request.question,
            top_k=candidate_limit,
            strategy=strategy,
            client=client,
            encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            collection_name=collection_name,
        )

    if rerank and documents:
        return rerank_documents(
            query=request.question,
            documents=documents,
            top_k=request.top_k,
            reranker=reranker,
        )

    return documents[:request.top_k]


def answer_question(
    request: RAGRequest,
    *,
    rerank: bool = False,
    strategy: str = "fixed",
    client=None,
    dense_encoder=None,
    sparse_encoder=None,
    reranker=None,
    llm_client=None,
    collection_name: str = COLLECTION_NAME,
    model_name: str | None = None,
) -> RAGResponse:
    """Retrieve evidence and generate a grounded answer."""

    sources = retrieve_documents(
        request,
        rerank=rerank,
        strategy=strategy,
        client=client,
        dense_encoder=dense_encoder,
        sparse_encoder=sparse_encoder,
        reranker=reranker,
        collection_name=collection_name,
    )

    answer = generate_answer(
        question=request.question,
        sources=sources,
        client=llm_client,
        model_name=model_name,
    )

    return RAGResponse(
        answer=answer,
        sources=sources,
        model=model_name,
    )
