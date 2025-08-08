"""Operations manager for metadata filtering and result transformation."""

from context_server.core.services.transformation import get_transformation_service
from context_server.models.api.search import (
    CompactCodeSearchResponse,
    CompactSearchResponse,
)
from context_server.models.database.responses import (
    CodeSearchResultDBResponse,
    FilteredMetadataDBResponse,
    SearchResultDBResponse,
)


class OperationsManager:
    """Manages helper operations like metadata filtering and result transformation."""

    def __init__(self, summarization_service=None):
        self.pool = None
        self.summarization_service = summarization_service

    def filter_metadata_for_search(self, metadata: dict) -> FilteredMetadataDBResponse:
        """Filter and organize metadata into clean, grouped structure for search results."""
        # Return structured metadata using Pydantic model
        return FilteredMetadataDBResponse(
            document=metadata.get("document", {}),
            chunk=metadata.get("chunk", {}),
            code_snippets=metadata.get("code_snippets", []),
        )

    async def transform_to_compact_format(
        self,
        results: list[SearchResultDBResponse | dict],
        query: str = "",
        mode: str = "hybrid",
        execution_time_ms: int = 0,
    ) -> CompactSearchResponse:
        """Transform full search results to compact MCP format.

        This method now delegates to the centralized transformation service.
        """
        transformation_service = get_transformation_service(self.summarization_service)
        return transformation_service.transform_to_compact_format(
            results, query, mode, execution_time_ms
        )

    def transform_code_to_compact_format(
        self,
        results: list[CodeSearchResultDBResponse | dict],
        query: str = "",
        execution_time_ms: int = 0,
    ) -> CompactCodeSearchResponse:
        """Transform code search results to compact MCP format.

        This method now delegates to the centralized transformation service.
        """
        transformation_service = get_transformation_service(self.summarization_service)
        return transformation_service.transform_code_to_compact_format(
            results, query, execution_time_ms
        )


__all__ = ["OperationsManager"]
