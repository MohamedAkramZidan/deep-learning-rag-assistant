from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    app_name: str = "Deep learning RAG App"
    environment: str = "development"
    log_level: str = "INFO"

    # Qdrant Cloud
    qdrant_url: str
    qdrant_api_key: SecretStr

    # Google Gemini
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-20b"

    gemini_api_key: SecretStr | None = None
    gemini_model: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
