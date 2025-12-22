from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Required
    openai_api_key: str
    upstash_redis_url: str

    # Optional with defaults
    host: str = "0.0.0.0"
    port: int = 8000
    session_ttl: int = 300  # 5 minutes
    conversation_ttl: int = 3600  # 1 hour

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
