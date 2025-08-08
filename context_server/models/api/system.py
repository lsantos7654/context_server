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


class JobCreateResponse(BaseModel):
    """Response for job creation (document ingestion)."""

    job_id: str
    status: str
    message: str


class JobCancelResponse(BaseModel):
    """Response for job cancellation."""

    message: str
    job_id: str
    status: str


class JobCleanupResponse(BaseModel):
    """Response for job cleanup operations."""

    message: str
    deleted_count: int
    days: int


class ActiveJobsResponse(BaseModel):
    """Response for listing active jobs."""

    active_jobs: list[JobStatus]
    total: int


class JobStatusResponse(BaseModel):
    """Enhanced job status response with full metadata."""

    id: str
    type: str
    context_id: str | None
    status: str
    progress: float = Field(ge=0.0, le=1.0)
    started_at: datetime
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    metadata: dict = Field(default_factory=dict)


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


class RootResponse(BaseModel):
    """Response for root endpoint."""

    message: str
    version: str
    docs: str


class DatabaseReinitializeResponse(BaseModel):
    """Response for database reinitialization."""

    message: str


__all__ = [
    "LogLevel",
    "LogEntry",
    "LogsResponse",
    "JobStatus",
    "JobCreateResponse",
    "JobCancelResponse",
    "JobCleanupResponse",
    "ActiveJobsResponse",
    "JobStatusResponse",
    "SystemStatus",
    "HealthResponse",
    "RootResponse",
    "DatabaseReinitializeResponse",
]
