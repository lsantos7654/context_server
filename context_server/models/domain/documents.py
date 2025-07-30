"""Document-related domain models."""

from pydantic import BaseModel, Field

from context_server.models.domain.chunks import ProcessedChunk
from context_server.models.domain.snippets import CodeSnippet


class ProcessedDocument(BaseModel):
    """A processed document with chunks and code snippets."""

    url: str = Field(..., description="Source URL of the document")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Full document content")
    chunks: list[ProcessedChunk] = Field(default_factory=list, description="Text chunks")
    code_snippets: list[CodeSnippet] = Field(default_factory=list, description="Code snippets")
    metadata: dict = Field(default_factory=dict, description="Document metadata")


class ProcessingResult(BaseModel):
    """Result of document processing operation."""

    documents: list[ProcessedDocument] = Field(default_factory=list, description="Processed documents")
    success: bool = Field(..., description="Whether processing was successful")
    error: str | None = Field(None, description="Error message if processing failed")


class DocumentStats(BaseModel):
    """Statistics about document processing."""

    total_documents: int = Field(..., ge=0, description="Total number of documents processed")
    total_chunks: int = Field(..., ge=0, description="Total number of text chunks created")
    total_code_snippets: int = Field(..., ge=0, description="Total number of code snippets extracted")
    total_tokens: int = Field(..., ge=0, description="Total number of tokens processed")
    processing_time_seconds: float = Field(..., ge=0, description="Total processing time in seconds")


__all__ = ["ProcessedDocument", "ProcessingResult", "DocumentStats"]