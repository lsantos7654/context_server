"""Centralized model definitions for Context Server.

This package contains all Pydantic models organized by domain:
- api/: API request/response models
- domain/: Core domain models
- config/: Configuration models
"""

# Export all models for convenient importing
from context_server.models.api.contexts import *
from context_server.models.api.documents import *
from context_server.models.api.search import *
from context_server.models.api.system import *
from context_server.models.config.cli import *
from context_server.models.config.server import *
from context_server.models.domain.chunks import *
from context_server.models.domain.documents import *
from context_server.models.domain.snippets import *

__all__ = [
    # API models
    "ContextCreate",
    "ContextResponse",
    "ContextMerge",
    "DocumentIngest",
    "DocumentResponse",
    "DocumentsResponse",
    "DocumentDelete",
    "SearchRequest",
    "SearchResult",
    "CodeSearchResult",
    "SearchResponse",
    "CodeSearchResponse",
    "LogEntry",
    "LogsResponse",
    "JobStatus",
    "SystemStatus",
    "HealthResponse",
    "SourceType",
    "SearchMode",
    "MergeMode",
    # Domain models
    "TextChunk",
    "ProcessedChunk",
    "CodeSnippet",
    "ProcessedDocument",
    "ProcessingResult",
    "DocumentStats",
    # Config models
    "ServerConfig",
    "CLIConfig",
]
