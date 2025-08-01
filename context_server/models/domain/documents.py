"""Document-related domain models."""

from dataclasses import dataclass, field

from context_server.models.domain.chunks import ProcessedChunk
from context_server.models.domain.snippets import CodeSnippet


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
        if self.total_documents < 0:
            raise ValueError("total_documents must be non-negative")
        if self.total_chunks < 0:
            raise ValueError("total_chunks must be non-negative")
        if self.total_code_snippets < 0:
            raise ValueError("total_code_snippets must be non-negative")
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        if self.processing_time_seconds < 0:
            raise ValueError("processing_time_seconds must be non-negative")


__all__ = ["ProcessedDocument", "ProcessingResult", "DocumentStats"]
