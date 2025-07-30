"""Context-related API models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class MergeMode(str, Enum):
    """Mode for merging contexts."""

    UNION = "union"
    INTERSECTION = "intersection"


class ContextCreate(BaseModel):
    """Request model for creating a new context."""

    name: str = Field(..., min_length=1, max_length=100, description="Context name")
    description: str = Field("", max_length=500, description="Context description")
    embedding_model: str = Field(
        "text-embedding-3-large", description="Embedding model to use"
    )


class ContextResponse(BaseModel):
    """Response model for context information."""

    id: str
    name: str
    description: str
    created_at: datetime
    document_count: int
    size_mb: float
    last_updated: datetime
    embedding_model: str


class ContextMerge(BaseModel):
    """Request model for merging contexts."""

    source_contexts: list[str] = Field(
        ..., min_items=1, description="Source context IDs"
    )
    target_context: str = Field(..., description="Target context name")
    mode: MergeMode = Field(MergeMode.UNION, description="Merge mode")


__all__ = ["MergeMode", "ContextCreate", "ContextResponse", "ContextMerge"]