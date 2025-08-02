"""Document-related domain models."""

from dataclasses import dataclass, field

from context_server.models.domain.chunks import ProcessedChunk
from context_server.models.domain.snippets import CodeSnippet
from ..validation import validate_range_fields


@dataclass
class ProcessedDocument:
    """A processed document with chunks and code snippets."""

    url: str
    title: str
    content: str  # Original content
    cleaned_content: str  # Content with code snippet placeholders
    chunks: list[ProcessedChunk] = field(default_factory=list)
    code_snippets: list[CodeSnippet] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Result of document processing operation."""

    success: bool
    documents: list[ProcessedDocument] = field(default_factory=list)
    error: str | None = None


@dataclass
class DocumentStats:
    """Statistics about document processing."""

    total_documents: int
    total_chunks: int
    total_code_snippets: int
    total_tokens: int
    processing_time_seconds: float

    def __post_init__(self):
        """Validate fields after initialization."""
        validate_range_fields(self, [
            "total_documents", "total_chunks", "total_code_snippets", 
            "total_tokens", "processing_time_seconds"
        ])


__all__ = ["ProcessedDocument", "ProcessingResult", "DocumentStats"]
