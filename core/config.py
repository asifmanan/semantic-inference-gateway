from functools import lru_cache

from pydantic import SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    """
    postgres_user: str
    postgres_password: SecretStr
    postgres_host: str
    postgres_port: int
    postgres_db: str

    # Redis L1 Cache Configuration
    redis_l1_host: str
    redis_l1_port: int
    redis_l1_db: int
    redis_l1_password: SecretStr | None = None

    # Upstream LLM (OpenAI-compatible) Configuration
    upstream_base_url: str = "http://localhost:11434/v1"
    upstream_model: str = "llama3.2:1b"
    upstream_timeout_seconds: float = 30.0
    upstream_max_connections: int = 50
    upstream_max_keepalive_connections: int = 20

    @computed_field
    @property
    def asyncpg_dsn(self) -> str:
        """
        Returns a DSN suitable for asyncpg connections.
        """
        password = self.postgres_password.get_secret_value()
        return (
            f"postgresql://{self.postgres_user}:{password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field
    @property
    def redis_l1_dsn(self) -> str:
        """
        Returns a DSN suitable for Redis connections.
        """
        password = self.redis_l1_password.get_secret_value() if self.redis_l1_password else None
        if password:
            return f"redis://:{password}@{self.redis_l1_host}:{self.redis_l1_port}/{self.redis_l1_db}"
        return f"redis://{self.redis_l1_host}:{self.redis_l1_port}/{self.redis_l1_db}"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings class.
    """
    return Settings()