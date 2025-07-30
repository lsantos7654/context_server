"""Chunk-related domain models."""

from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    """A chunk of text with metadata and line tracking."""

    content: str = Field(..., description="Text content of the chunk")
    tokens: int = Field(..., ge=0, description="Number of tokens in the chunk")
    metadata: dict = Field(default_factory=dict, description="Chunk metadata")
    start_line: int | None = Field(None, ge=0, description="Starting line number")
    end_line: int | None = Field(None, ge=0, description="Ending line number")
    char_start: int | None = Field(None, ge=0, description="Starting character position")
    char_end: int | None = Field(None, ge=0, description="Ending character position")


class ProcessedChunk(BaseModel):
    """A processed text chunk with embedding and line tracking."""

    content: str = Field(..., description="Text content of the chunk")
    embedding: list[float] = Field(..., description="Vector embedding of the chunk")
    metadata: dict = Field(default_factory=dict, description="Chunk metadata")
    tokens: int = Field(..., ge=0, description="Number of tokens in the chunk")
    summary: str | None = Field(None, description="LLM-generated summary")
    summary_model: str | None = Field(None, description="Model used for summary generation")
    start_line: int | None = Field(None, ge=0, description="Starting line number")
    end_line: int | None = Field(None, ge=0, description="Ending line number")
    char_start: int | None = Field(None, ge=0, description="Starting character position")
    char_end: int | None = Field(None, ge=0, description="Ending character position")


__all__ = ["TextChunk", "ProcessedChunk"]