"""Export and import API models for context data."""

from datetime import datetime

from pydantic import BaseModel, Field


class ExportedContext(BaseModel):
    """Exported context metadata."""

    id: str
    name: str
    description: str
    embedding_model: str
    created_at: datetime
    # Allow additional fields for raw export data
    model_config = {"extra": "allow"}


class ExportedDocument(BaseModel):
    """Exported document data."""

    id: str
    url: str
    title: str
    content: str
    indexed_at: datetime
    # Allow additional fields for raw export data
    model_config = {"extra": "allow"}


class ExportedChunk(BaseModel):
    """Exported chunk data."""

    id: str
    document_id: str
    content: str
    chunk_index: int
    # Allow additional fields including embeddings
    model_config = {"extra": "allow"}


class ExportedCodeSnippet(BaseModel):
    """Exported code snippet data."""

    id: str
    document_id: str
    content: str
    # Allow additional fields including embeddings
    model_config = {"extra": "allow"}


class ContextExport(BaseModel):
    """Complete context export data structure."""

    schema_version: str = Field(
        "1.0", description="Export schema version for compatibility"
    )
    context: ExportedContext = Field(
        ..., description="Context metadata and configuration"
    )
    documents: list[ExportedDocument] = Field(
        ..., description="All documents with metadata"
    )
    chunks: list[ExportedChunk] = Field(..., description="All chunks with embeddings")
    code_snippets: list[ExportedCodeSnippet] = Field(
        ..., description="All code snippets with embeddings"
    )
    exported_at: datetime = Field(..., description="Timestamp when export was created")
    total_documents: int = Field(..., description="Total number of documents exported")
    total_chunks: int = Field(..., description="Total number of chunks exported")
    total_code_snippets: int = Field(
        ..., description="Total number of code snippets exported"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ContextImportRequest(BaseModel):
    """Request for importing context data."""

    context_data: ContextExport = Field(
        ..., description="Context export data to import"
    )
    overwrite_existing: bool = Field(
        False, description="Whether to overwrite existing context with same name"
    )


class ContextImportResponse(BaseModel):
    """Response for context import operation."""

    success: bool = Field(..., description="Whether import was successful")
    context_id: str = Field(..., description="ID of imported/updated context")
    context_name: str = Field(..., description="Name of imported/updated context")
    imported_documents: int = Field(..., description="Number of documents imported")
    imported_chunks: int = Field(..., description="Number of chunks imported")
    imported_code_snippets: int = Field(
        ..., description="Number of code snippets imported"
    )
    message: str = Field(..., description="Human-readable status message")


class ContextMergeResponse(BaseModel):
    """Response for context merge operation."""

    success: bool = Field(..., description="Whether merge was successful")
    target_context_id: str = Field(..., description="ID of target context")
    target_context_name: str = Field(..., description="Name of target context")
    merged_documents: int = Field(
        ..., description="Number of documents in merged context"
    )
    merged_chunks: int = Field(..., description="Number of chunks in merged context")
    merged_code_snippets: int = Field(
        ..., description="Number of code snippets in merged context"
    )
    source_contexts_processed: int = Field(
        ..., description="Number of source contexts processed"
    )
    message: str = Field(..., description="Human-readable status message")


__all__ = [
    "ExportedContext",
    "ExportedDocument",
    "ExportedChunk",
    "ExportedCodeSnippet",
    "ContextExport",
    "ContextImportRequest",
    "ContextImportResponse",
    "ContextMergeResponse",
]
