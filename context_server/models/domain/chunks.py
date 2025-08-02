"""Chunk-related domain models."""

from dataclasses import dataclass, field

from context_server.models.validation import validate_range_fields


@dataclass
class TextChunk:
    """A chunk of text with metadata and line tracking."""

    content: str
    tokens: int
    metadata: dict = field(default_factory=dict)
    start_line: int | None = None
    end_line: int | None = None
    char_start: int | None = None
    char_end: int | None = None

    def __post_init__(self):
        """Validate fields after initialization."""
        validate_range_fields(self, ["tokens", "start_line", "end_line", "char_start", "char_end"])


@dataclass
class ProcessedChunk:
    """A processed text chunk with embedding and line tracking."""

    content: str
    embedding: list[float]
    tokens: int
    metadata: dict = field(default_factory=dict)
    title: str | None = None
    summary: str | None = None
    summary_model: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    char_start: int | None = None
    char_end: int | None = None

    def __post_init__(self):
        """Validate fields after initialization."""
        validate_range_fields(self, ["tokens", "start_line", "end_line", "char_start", "char_end"])


__all__ = ["TextChunk", "ProcessedChunk"]
