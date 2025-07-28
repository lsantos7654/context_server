"""Pydantic models for the Context Server API."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Type of document source."""

    URL = "url"
    FILE = "file"


class SearchMode(str, Enum):
    """Search mode for document queries."""

    VECTOR = "vector"
    FULLTEXT = "fulltext"
    HYBRID = "hybrid"


class MergeMode(str, Enum):
    """Mode for merging contexts."""

    UNION = "union"
    INTERSECTION = "intersection"


# Context models
class ContextCreate(BaseModel):
    """Request model for creating a new context."""

    name: str = Field(..., min_length=1, max_length=100, description="Context name")
    description: str = Field("", max_length=500, description="Context description")
    embedding_model: str = Field(
        "text-embedding-3-large", description="Embedding model to use"
    )


class ContextResponse(BaseModel):
    """Response model for context information."""

    id: str
    name: str
    description: str
    created_at: datetime
    document_count: int
    size_mb: float
    last_updated: datetime
    embedding_model: str


class ContextMerge(BaseModel):
    """Request model for merging contexts."""

    source_contexts: list[str] = Field(
        ..., min_items=1, description="Source context IDs"
    )
    target_context: str = Field(..., description="Target context name")
    mode: MergeMode = Field(MergeMode.UNION, description="Merge mode")


# Document models
class DocumentIngest(BaseModel):
    """Request model for document ingestion."""

    source_type: SourceType
    source: str = Field(..., description="URL, file path, or git repository")
    options: dict = Field(
        default_factory=dict, description="Additional processing options"
    )


class DocumentResponse(BaseModel):
    """Response model for document information."""

    id: str
    url: str
    title: str
    indexed_at: datetime
    chunks: int
    metadata: dict


class DocumentsResponse(BaseModel):
    """Response model for document listing."""

    documents: list[DocumentResponse]
    total: int
    offset: int
    limit: int


class DocumentDelete(BaseModel):
    """Request model for deleting documents."""

    document_ids: list[str] = Field(
        ..., min_items=1, description="Document IDs to delete"
    )


# Search models
class SearchRequest(BaseModel):
    """Request model for searching documents."""

    query: str = Field(..., min_length=1, description="Search query")
    mode: SearchMode = Field(SearchMode.HYBRID, description="Search mode")
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")
    include_raw: bool = Field(False, description="Include raw document content")
    filters: dict = Field(default_factory=dict, description="Additional search filters")


class SearchResult(BaseModel):
    """Individual search result."""

    id: str
    document_id: str | None = None
    title: str
    content: str
    summary: str | None = None
    summary_model: str | None = None
    score: float
    metadata: dict
    url: str | None = None
    chunk_index: int | None = None
    content_type: str = "chunk"


class CodeSearchResult(BaseModel):
    """Individual code search result (no summary/chunk_index fields)."""

    id: str
    document_id: str | None = None
    title: str
    content: str
    snippet_type: str = "code_block"
    score: float
    line_count: int
    metadata: dict
    url: str | None = None
    content_type: str = "code_snippet"
    
    class Config:
        # Exclude fields that are None from the JSON output
        exclude_none = True


class SearchResponse(BaseModel):
    """Response model for search results."""

    results: list[SearchResult]
    total: int
    query: str
    mode: str
    execution_time_ms: int


class CodeSearchResponse(BaseModel):
    """Response model for code search results."""

    results: list[CodeSearchResult]
    total: int
    query: str
    mode: str
    execution_time_ms: int


# System models
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
