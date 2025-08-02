"""Server configuration models."""

from pydantic import BaseModel, Field, validator


class ServerConfig(BaseModel):
    """Server configuration with validation."""

    host: str = "localhost"
    port: int = Field(default=8000, ge=1, le=65535)
    database_url: str = (
        "postgresql://context_user:context_password@localhost:5432/context_server"
    )
    openai_api_key: str | None = None

    @validator("host")
    def validate_host(cls, v):
        """Validate host is not empty."""
        if not v or not v.strip():
            raise ValueError("Host cannot be empty")
        return v.strip()

    @validator("database_url")
    def validate_database_url(cls, v):
        """Validate database URL format."""
        if not v.startswith(("postgresql://", "postgres://")):
            raise ValueError("Database URL must be a PostgreSQL connection string")
        return v

    class Config:
        """Pydantic config."""
        extra = "forbid"


__all__ = ["ServerConfig"]
