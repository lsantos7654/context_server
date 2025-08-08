"""Code snippet domain models."""

from pydantic import BaseModel, Field, field_validator


class ExtractedCodeSnippet(BaseModel):
    """A code snippet extracted from content, before embedding generation."""

    content: str
    type: str = Field(description="Type of code snippet (code_block, inline_code)")
    start_pos: int = Field(ge=0, description="Start position in original text")
    end_pos: int = Field(ge=0, description="End position in original text")
    char_count: int = Field(ge=0, description="Number of characters in snippet")
    line_count: int = Field(ge=1, description="Number of lines in snippet")
    original_match: str = Field(description="Original matched text for replacement")

    @field_validator("end_pos")
    @classmethod
    def validate_end_pos(cls, v: int, info) -> int:
        """Ensure end_pos >= start_pos."""
        if info.data.get("start_pos") is not None and v < info.data["start_pos"]:
            raise ValueError("end_pos must be >= start_pos")
        return v


class CodeSnippet(BaseModel):
    """A code snippet extracted from content."""

    content: str
    embedding: list[float]
    metadata: dict = Field(default_factory=dict)
    start_line: int | None = Field(default=None, ge=0)
    end_line: int | None = Field(default=None, ge=0)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)

    @field_validator("end_line")
    @classmethod
    def validate_end_line(cls, v: int | None, info) -> int | None:
        """Ensure end_line >= start_line if both are set."""
        if v is not None and info.data.get("start_line") is not None:
            if v < info.data["start_line"]:
                raise ValueError("end_line must be >= start_line")
        return v

    @field_validator("char_end")
    @classmethod
    def validate_char_end(cls, v: int | None, info) -> int | None:
        """Ensure char_end >= char_start if both are set."""
        if v is not None and info.data.get("char_start") is not None:
            if v < info.data["char_start"]:
                raise ValueError("char_end must be >= char_start")
        return v


__all__ = ["ExtractedCodeSnippet", "CodeSnippet"]
