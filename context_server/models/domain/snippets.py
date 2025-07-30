"""Code snippet domain models."""

from pydantic import BaseModel, Field


class CodeSnippet(BaseModel):
    """A code snippet extracted from content."""

    content: str = Field(..., description="Code content of the snippet")
    embedding: list[float] = Field(..., description="Vector embedding of the code snippet")
    metadata: dict = Field(default_factory=dict, description="Code snippet metadata")
    start_line: int | None = Field(None, ge=0, description="Starting line number")
    end_line: int | None = Field(None, ge=0, description="Ending line number")
    char_start: int | None = Field(None, ge=0, description="Starting character position")
    char_end: int | None = Field(None, ge=0, description="Ending character position")


__all__ = ["CodeSnippet"]