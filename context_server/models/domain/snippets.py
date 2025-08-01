"""Code snippet domain models."""

from dataclasses import dataclass, field


@dataclass
class CodeSnippet:
    """A code snippet extracted from content."""

    content: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)
    start_line: int | None = None
    end_line: int | None = None
    char_start: int | None = None
    char_end: int | None = None

    def __post_init__(self):
        """Validate fields after initialization."""
        if self.start_line is not None and self.start_line < 0:
            raise ValueError("start_line must be non-negative")
        if self.end_line is not None and self.end_line < 0:
            raise ValueError("end_line must be non-negative")
        if self.char_start is not None and self.char_start < 0:
            raise ValueError("char_start must be non-negative")
        if self.char_end is not None and self.char_end < 0:
            raise ValueError("char_end must be non-negative")


__all__ = ["CodeSnippet"]
