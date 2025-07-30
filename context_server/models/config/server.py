"""Server configuration models."""

from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Server configuration."""

    host: str = "localhost"
    port: int = 8000
    database_url: str = (
        "postgresql://context_user:context_password@localhost:5432/context_server"
    )
    openai_api_key: str | None = None


__all__ = ["ServerConfig"]