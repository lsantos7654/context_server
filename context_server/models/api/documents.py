"""Document-related API models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Type of document source."""

    URL = "url"
    FILE = "file"


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


class DocumentContentResponse(BaseModel):
    """Response for individual document content retrieval with pagination."""

    id: str
    title: str
    url: str | None = None
    content: str
    document_type: str
    created_at: datetime
    full_content_length: int
    pagination: dict = Field(default_factory=dict)


class ChunkResponse(BaseModel):
    """Response for individual chunk retrieval."""

    id: str
    document_id: str
    chunk_index: int
    content: str
    summary: str | None = None
    summary_model: str | None = None
    tokens: int
    start_line: int | None = None
    end_line: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    title: str | None = None
    url: str | None = None
    metadata: dict = Field(default_factory=dict)


class CodeSnippetResponse(BaseModel):
    """Response for individual code snippet retrieval."""

    id: str
    document_id: str | None = None
    content: str
    start_line: int | None = None
    end_line: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    snippet_type: str = "code_block"
    document_title: str | None = None
    document_url: str | None = None


class CodeSnippetsResponse(BaseModel):
    """Response for listing code snippets from a document."""

    snippets: list[CodeSnippetResponse]
    total: int
    document_id: str


class FileProcessingResult(BaseModel):
    """Result for individual file processing in directory extraction."""

    file: str
    status: str  # "processed", "failed", "skipped"
    error: str | None = None
    job_id: str | None = None


class DirectoryExtractionResponse(BaseModel):
    """Response for local directory extraction."""

    success: bool
    message: str
    processed_files: int
    failed_files: int
    total_files: int
    directory_path: str
    files: list[FileProcessingResult]


class DocumentDeleteResponse(BaseModel):
    """Response for document deletion."""

    message: str
    deleted_count: int
    context_name: str


__all__ = [
    "SourceType",
    "DocumentIngest",
    "DocumentResponse",
    "DocumentsResponse",
    "DocumentDelete",
    "DocumentContentResponse",
    "ChunkResponse",
    "CodeSnippetResponse",
    "CodeSnippetsResponse",
    "FileProcessingResult",
    "DirectoryExtractionResponse",
    "DocumentDeleteResponse",
]
