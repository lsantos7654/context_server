"""Code snippet domain models."""

from dataclasses import dataclass, field

from context_server.models.validation import validate_range_fields


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
        validate_range_fields(self, ["start_line", "end_line", "char_start", "char_end"])


__all__ = ["CodeSnippet"]
