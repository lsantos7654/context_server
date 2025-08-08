"""Database layer response models.

These models represent the structure of data returned from database operations,
providing type safety and validation at the database boundary.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ContextDBResponse(BaseModel):
    """Database response for context operations."""

    id: str | UUID
    name: str
    description: str = ""
    embedding_model: str
    created_at: datetime
    document_count: int = 0
    size_mb: float = 0.0
    last_updated: datetime | None = None

    # Allow extra fields for compatibility with raw database results
    model_config = {"extra": "allow"}


class DocumentDBResponse(BaseModel):
    """Database response for document operations."""

    id: str | UUID
    context_id: str | UUID
    url: str
    title: str
    content: str
    indexed_at: datetime
    metadata: dict = Field(default_factory=dict)
    source_type: str = "url"
    chunks: int = 0

    # Allow extra fields for compatibility with raw database results
    model_config = {"extra": "allow"}


class ChunkDBResponse(BaseModel):
    """Database response for chunk operations."""

    id: str | UUID
    document_id: str | UUID
    context_id: str | UUID
    content: str
    chunk_index: int
    tokens: int = 0
    title: str | None = None
    summary: str | None = None
    summary_model: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict = Field(default_factory=dict)
    url: str | None = None

    # Allow extra fields for compatibility with raw database results
    model_config = {"extra": "allow"}


class CodeSnippetDBResponse(BaseModel):
    """Database response for code snippet operations."""

    id: str | UUID
    document_id: str | UUID
    context_id: str | UUID
    content: str
    start_line: int | None = None
    end_line: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    snippet_type: str = "code_block"
    preview: str | None = None
    metadata: dict = Field(default_factory=dict)
    document_title: str | None = None
    document_url: str | None = None

    # Allow extra fields for compatibility with raw database results
    model_config = {"extra": "allow"}


class JobDBResponse(BaseModel):
    """Database response for job operations."""

    id: str
    type: str
    context_id: str | UUID | None = None
    status: str
    progress: float = Field(ge=0.0, le=1.0, default=0.0)
    started_at: datetime
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    metadata: dict = Field(default_factory=dict)
    result_data: dict = Field(default_factory=dict)

    # Allow extra fields for compatibility with raw database results
    model_config = {"extra": "allow"}


class SearchResultDBResponse(BaseModel):
    """Database response for search operations."""

    id: str | UUID
    document_id: str | UUID
    title: str | None = None
    content: str
    summary: str | None = None
    score: float
    url: str | None = None
    metadata: dict = Field(default_factory=dict)
    chunk_index: int | None = None

    # Allow extra fields for compatibility with raw database results
    model_config = {"extra": "allow"}


class CodeSearchResultDBResponse(BaseModel):
    """Database response for code search operations."""

    id: str | UUID
    document_id: str | UUID
    content: str
    score: float
    url: str | None = None
    line_count: int | None = None
    metadata: dict = Field(default_factory=dict)

    # Allow extra fields for compatibility with raw database results
    model_config = {"extra": "allow"}


class DocumentsListDBResponse(BaseModel):
    """Database response for paginated document lists."""

    documents: list[DocumentDBResponse] = Field(default_factory=list)
    total: int = Field(ge=0)
    offset: int = Field(ge=0, default=0)
    limit: int = Field(ge=1, default=50)

    # Allow extra fields for compatibility with raw database results
    model_config = {"extra": "allow"}


class SearchResultsDBResponse(BaseModel):
    """Database response for search operations with metadata."""

    results: list[SearchResultDBResponse] = Field(default_factory=list)
    total: int = Field(ge=0)
    query: str = ""
    mode: str = "hybrid"
    execution_time_ms: int = Field(ge=0, default=0)

    # Allow extra fields for compatibility with raw database results
    model_config = {"extra": "allow"}


class CodeSearchResultsDBResponse(BaseModel):
    """Database response for code search operations with metadata."""

    results: list[CodeSearchResultDBResponse] = Field(default_factory=list)
    total: int = Field(ge=0)
    query: str = ""
    execution_time_ms: int = Field(ge=0, default=0)

    # Allow extra fields for compatibility with raw database results
    model_config = {"extra": "allow"}


class ChunkWithDocumentDBResponse(BaseModel):
    """Database response for chunk retrieval with document context."""

    # Chunk fields
    id: str | UUID
    document_id: str | UUID
    context_id: str | UUID
    content: str
    chunk_index: int
    tokens: int = 0
    title: str | None = None
    summary: str | None = None
    summary_model: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict = Field(default_factory=dict)
    code_snippet_ids: list[str] = Field(default_factory=list)
    created_at: datetime | None = None

    # Document context fields
    doc_title: str | None = None
    doc_url: str | None = None
    doc_metadata: dict = Field(default_factory=dict)
    parent_page_size: int | None = None
    parent_total_chunks: int | None = None

    # Allow extra fields for compatibility with raw database results
    model_config = {"extra": "allow"}


class CodeSnippetWithDocumentDBResponse(BaseModel):
    """Database response for code snippet retrieval with document context."""

    # Code snippet fields
    id: str | UUID
    document_id: str | UUID
    context_id: str | UUID
    content: str
    start_line: int | None = None
    end_line: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    snippet_type: str = "code_block"
    preview: str | None = None
    metadata: dict = Field(default_factory=dict)

    # Document context fields
    document_title: str | None = None
    document_url: str | None = None
    document_metadata: dict = Field(default_factory=dict)

    # Allow extra fields for compatibility with raw database results
    model_config = {"extra": "allow"}


class ExportDataDBResponse(BaseModel):
    """Database response for context export operations."""

    context: dict = Field(default_factory=dict)
    documents: list[dict] = Field(default_factory=list)
    chunks: list[dict] = Field(default_factory=list)
    code_snippets: list[dict] = Field(default_factory=list)
    export_timestamp: datetime
    total_documents: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    total_code_snippets: int = Field(ge=0)

    # Allow extra fields for compatibility with raw database results
    model_config = {"extra": "allow"}


class ImportResultDBResponse(BaseModel):
    """Database response for context import operations."""

    context_name: str
    context_id: str | UUID
    success: bool
    documents_imported: int = Field(ge=0, default=0)
    chunks_imported: int = Field(ge=0, default=0)
    code_snippets_imported: int = Field(ge=0, default=0)
    conflicts_resolved: int = Field(ge=0, default=0)
    import_timestamp: datetime
    error_message: str | None = None

    # Allow extra fields for compatibility with raw database results
    model_config = {"extra": "allow"}


class MergeResultDBResponse(BaseModel):
    """Database response for context merge operations."""

    target_context_name: str
    target_context_id: str | UUID
    source_contexts: list[str] = Field(default_factory=list)
    merge_mode: str
    success: bool
    documents_merged: int = Field(ge=0, default=0)
    chunks_merged: int = Field(ge=0, default=0)
    code_snippets_merged: int = Field(ge=0, default=0)
    duplicates_handled: int = Field(ge=0, default=0)
    merge_timestamp: datetime
    error_message: str | None = None

    # Allow extra fields for compatibility with raw database results
    model_config = {"extra": "allow"}


class FilteredMetadataDBResponse(BaseModel):
    """Database response for filtered metadata operations."""

    document: dict = Field(default_factory=dict)
    chunk: dict = Field(default_factory=dict)
    code_snippets: list = Field(default_factory=list)

    # Allow extra fields for compatibility with raw database results
    model_config = {"extra": "allow"}


__all__ = [
    "ContextDBResponse",
    "DocumentDBResponse",
    "ChunkDBResponse",
    "CodeSnippetDBResponse",
    "JobDBResponse",
    "SearchResultDBResponse",
    "CodeSearchResultDBResponse",
    "DocumentsListDBResponse",
    "SearchResultsDBResponse",
    "CodeSearchResultsDBResponse",
    "ChunkWithDocumentDBResponse",
    "CodeSnippetWithDocumentDBResponse",
    "ExportDataDBResponse",
    "ImportResultDBResponse",
    "MergeResultDBResponse",
    "FilteredMetadataDBResponse",
]
