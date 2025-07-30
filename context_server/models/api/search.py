"""Search-related API models."""

from enum import Enum

from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    """Search mode for document queries."""

    VECTOR = "vector"
    FULLTEXT = "fulltext"
    HYBRID = "hybrid"


class SearchRequest(BaseModel):
    """Request model for searching documents."""

    query: str = Field(..., min_length=1, description="Search query")
    mode: SearchMode = Field(SearchMode.HYBRID, description="Search mode")
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")
    include_raw: bool = Field(False, description="Include raw document content")
    filters: dict = Field(default_factory=dict, description="Additional search filters")


class SearchResult(BaseModel):
    """Individual search result."""

    id: str
    document_id: str | None = None
    title: str
    content: str
    summary: str | None = None
    summary_model: str | None = None
    score: float
    metadata: dict
    url: str | None = None
    content_type: str = "chunk"


class CodeSearchResult(BaseModel):
    """Individual code search result (no summary/chunk_index fields)."""

    id: str
    document_id: str | None = None
    title: str
    content: str
    snippet_type: str = "code_block"
    score: float
    line_count: int
    metadata: dict
    url: str | None = None
    content_type: str = "code_snippet"
    
    class Config:
        # Exclude fields that are None from the JSON output
        exclude_none = True


class SearchResponse(BaseModel):
    """Response model for search results."""

    results: list[SearchResult]
    total: int
    query: str
    mode: str
    execution_time_ms: int


class CodeSearchResponse(BaseModel):
    """Response model for code search results."""

    results: list[CodeSearchResult]
    total: int
    query: str
    mode: str
    execution_time_ms: int


__all__ = [
    "SearchMode",
    "SearchRequest", 
    "SearchResult", 
    "CodeSearchResult",
    "SearchResponse", 
    "CodeSearchResponse"
]