"""Chunk-related domain models."""

from dataclasses import dataclass, field


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
        if self.tokens < 0:
            raise ValueError("tokens must be non-negative")
        if self.start_line is not None and self.start_line < 0:
            raise ValueError("start_line must be non-negative")
        if self.end_line is not None and self.end_line < 0:
            raise ValueError("end_line must be non-negative")
        if self.char_start is not None and self.char_start < 0:
            raise ValueError("char_start must be non-negative")
        if self.char_end is not None and self.char_end < 0:
            raise ValueError("char_end must be non-negative")


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
        if self.tokens < 0:
            raise ValueError("tokens must be non-negative")
        if self.start_line is not None and self.start_line < 0:
            raise ValueError("start_line must be non-negative")
        if self.end_line is not None and self.end_line < 0:
            raise ValueError("end_line must be non-negative")
        if self.char_start is not None and self.char_start < 0:
            raise ValueError("char_start must be non-negative")
        if self.char_end is not None and self.char_end < 0:
            raise ValueError("char_end must be non-negative")


__all__ = ["TextChunk", "ProcessedChunk"]