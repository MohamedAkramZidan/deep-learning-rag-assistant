
from rag_app.core.logging import configure_logging
from rag_app.ingestion.pipeline import run_ingestion


def main() -> None:
    configure_logging()
    run_ingestion()


if __name__ == "__main__":
    main()
