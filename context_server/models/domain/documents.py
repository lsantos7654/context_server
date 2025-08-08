"""Document-related domain models."""

from pydantic import BaseModel, Field

from context_server.models.domain.chunks import ProcessedChunk
from context_server.models.domain.snippets import CodeSnippet


class ProcessedDocument(BaseModel):
    """A processed document with chunks and code snippets."""

    url: str
    title: str
    content: str = Field(description="Original content")
    cleaned_content: str | None = Field(default=None, description="Content with code snippet placeholders")
    chunks: list[ProcessedChunk] = Field(default_factory=list)
    code_snippets: list[CodeSnippet] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ProcessingResult(BaseModel):
    """Result of document processing operation."""

    success: bool
    documents: list[ProcessedDocument] = Field(default_factory=list)
    error: str | None = None


class DocumentStats(BaseModel):
    """Statistics about document processing."""

    total_documents: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    total_code_snippets: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    processing_time_seconds: float = Field(ge=0.0)


__all__ = ["ProcessedDocument", "ProcessingResult", "DocumentStats"]
