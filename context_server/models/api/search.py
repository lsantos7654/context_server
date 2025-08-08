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


class CodeSearchResult(BaseModel):
    """Individual code search result (no summary/chunk_index fields)."""

    id: str
    document_id: str | None = None
    content: str
    score: float
    line_count: int
    metadata: dict
    url: str | None = None

    class Config:
        # Exclude fields that are None from the JSON output
        exclude_none = True


class CodeSnippetInfo(BaseModel):
    """Information about a code snippet in compact search results."""

    id: str
    lines: int
    chars: int
    preview: str


class CompactSearchResultItem(BaseModel):
    """Individual compact search result item."""

    id: str | None = None
    document_id: str | None = None
    title: str | None = None
    summary: str | None = None
    score: float | None = None
    url: str | None = None
    code_snippets_count: int = 0
    code_snippet_ids: list[CodeSnippetInfo] = Field(default_factory=list)


class CompactCodeSearchResultItem(BaseModel):
    """Individual compact code search result item."""

    id: str | None = None
    document_id: str | None = None
    content: str | None = None
    score: float | None = None
    url: str | None = None
    line_count: int = 0


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


class CompactSearchResponse(BaseModel):
    """Response model for compact search results (MCP format)."""

    results: list[CompactSearchResultItem]
    total: int
    query: str
    mode: str
    execution_time_ms: int
    note: str | None = None


class CompactCodeSearchResponse(BaseModel):
    """Response model for compact code search results (MCP format)."""

    results: list[CompactCodeSearchResultItem]
    total: int
    query: str
    execution_time_ms: int
    mode: str = "hybrid"
    language_filter: str | None = None
    note: str | None = None


__all__ = [
    "SearchMode",
    "SearchRequest",
    "SearchResult",
    "CodeSearchResult",
    "SearchResponse",
    "CodeSearchResponse",
    "CodeSnippetInfo",
    "CompactSearchResultItem",
    "CompactCodeSearchResultItem",
    "CompactSearchResponse",
    "CompactCodeSearchResponse",
]
