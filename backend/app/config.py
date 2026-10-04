from functools import lru_cache

from pydantic import SecretStr
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Secrets are read only from environment variables or backend/.env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: SecretStr
    model_name: str = "gpt-5"
    allowed_origins: str = "http://localhost:4200"
    enable_live_crewai: bool = True

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip().rstrip("/") for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        raise RuntimeError(
            "OPENAI_API_KEY is required. Set it in backend/.env locally or as a Render environment variable."
        ) from exc
