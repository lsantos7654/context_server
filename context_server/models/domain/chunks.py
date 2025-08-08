"""Chunk-related domain models."""

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Self


class TextChunk(BaseModel):
    """A chunk of text with metadata and line tracking."""

    content: str
    tokens: int = Field(ge=0, description="Number of tokens in the chunk")
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


class ProcessedChunk(BaseModel):
    """A processed text chunk with embedding and line tracking."""

    content: str
    embedding: list[float]
    tokens: int = Field(ge=0, description="Number of tokens in the chunk")
    metadata: dict = Field(default_factory=dict)
    title: str | None = None
    summary: str | None = None
    summary_model: str | None = None
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


__all__ = ["TextChunk", "ProcessedChunk"]
