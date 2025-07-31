"""Operations manager for metadata filtering and result transformation."""

from ...services.transformation import get_transformation_service

class OperationsManager:
    """Manages helper operations like metadata filtering and result transformation."""
    
    def __init__(self, summarization_service=None):
        self.pool = None
        self.summarization_service = summarization_service
    
    def filter_metadata_for_search(self, metadata: dict) -> dict:
        """Filter and organize metadata into clean, grouped structure for search results."""
        # For now, return simplified metadata to avoid breaking the system
        return {
            "document": metadata.get("document", {}),
            "chunk": metadata.get("chunk", {}),
            "code_snippets": metadata.get("code_snippets", []),
        }
    
    async def transform_to_compact_format(self, results: list[dict], query: str = "", mode: str = "hybrid", execution_time_ms: int = 0) -> dict:
        """Transform full search results to compact MCP format.
        
        This method now delegates to the centralized transformation service.
        """
        transformation_service = get_transformation_service(self.summarization_service)
        return transformation_service.transform_to_compact_format(
            results, query, mode, execution_time_ms
        )
    
    def transform_code_to_compact_format(self, results: list[dict], query: str = "", execution_time_ms: int = 0) -> dict:
        """Transform code search results to compact MCP format.
        
        This method now delegates to the centralized transformation service.
        """
        transformation_service = get_transformation_service(self.summarization_service)
        return transformation_service.transform_code_to_compact_format(
            results, query, execution_time_ms
        )


__all__ = ["OperationsManager"]