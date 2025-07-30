"""System and monitoring related API models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class LogLevel(str, Enum):
    """Log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LogEntry(BaseModel):
    """Log entry model."""

    timestamp: datetime
    level: LogLevel
    message: str
    context: dict = Field(default_factory=dict)


class LogsResponse(BaseModel):
    """Response model for system logs."""

    logs: list[LogEntry]
    total: int
    offset: int
    limit: int


class JobStatus(BaseModel):
    """Processing job status."""

    id: str
    type: str
    context: str
    progress: float = Field(ge=0.0, le=1.0)
    status: str
    started_at: datetime
    estimated_completion: datetime | None = None


class SystemStatus(BaseModel):
    """System status response."""

    active_jobs: list[JobStatus]
    total_contexts: int
    total_documents: int
    system_uptime: str
    memory_usage_mb: float
    cpu_usage_percent: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    timestamp: datetime
    version: str
    database_connected: bool
    embedding_service_available: bool


__all__ = [
    "LogLevel",
    "LogEntry", 
    "LogsResponse",
    "JobStatus", 
    "SystemStatus", 
    "HealthResponse"
]