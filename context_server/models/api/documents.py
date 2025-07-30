"""Document-related API models."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Type of document source."""

    URL = "url"
    FILE = "file"


class DocumentIngest(BaseModel):
    """Request model for document ingestion."""

    source_type: SourceType
    source: str = Field(..., description="URL, file path, or git repository")
    options: dict = Field(
        default_factory=dict, description="Additional processing options"
    )


class DocumentResponse(BaseModel):
    """Response model for document information."""

    id: str
    url: str
    title: str
    indexed_at: datetime
    chunks: int
    metadata: dict


class DocumentsResponse(BaseModel):
    """Response model for document listing."""

    documents: list[DocumentResponse]
    total: int
    offset: int
    limit: int


class DocumentDelete(BaseModel):
    """Request model for deleting documents."""

    document_ids: list[str] = Field(
        ..., min_items=1, description="Document IDs to delete"
    )


__all__ = [
    "SourceType", 
    "DocumentIngest", 
    "DocumentResponse", 
    "DocumentsResponse", 
    "DocumentDelete"
]