
from fastapi import FastAPI

from rag_app.api.routes import router
from rag_app.core.config import settings
from rag_app.core.logging import configure_logging

configure_logging()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
)

app.include_router(
    router,
    prefix="/api/v1",
    tags=["RAG"],
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}
