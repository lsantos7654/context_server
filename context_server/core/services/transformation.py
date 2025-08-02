"""Centralized transformation service for search results.

This service provides consistent transformation logic for search results
across all interfaces (API, CLI, MCP server). It serves as the single
source of truth for result formatting and field standardization.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TransformationService:
    """Centralized service for transforming search results across all interfaces."""

    def __init__(self, summarization_service=None):
        """Initialize the transformation service.

        Args:
            summarization_service: Optional service for generating summaries
        """
        self.summarization_service = summarization_service

    def transform_to_compact_format(
        self,
        results: list[dict[str, Any]],
        query: str = "",
        mode: str = "hybrid",
        execution_time_ms: int = 0,
    ) -> dict[str, Any]:
        """Transform full search results to compact MCP format.

        This is the single source of truth for transforming search results
        to the compact format used by both CLI and MCP server.

        Args:
            results: List of full search result dictionaries
            query: Original search query
            mode: Search mode used
            execution_time_ms: Time taken for search execution

        Returns:
            Dictionary with compact search results optimized for quick scanning
        """
        compact_results = []

        for result in results:
            # Use summary if available, otherwise truncate content
            display_content = result.get("summary", "")
            if not display_content:
                content = result.get("content", "")
                display_content = (
                    content[:150] + "..." if len(content) > 150 else content
                )
            # No truncation for summaries - they're already AI-generated and meaningful

            # Get code snippets count and IDs from metadata
            metadata = result.get("metadata", {})
            code_snippets = metadata.get("code_snippets", [])
            code_snippets_count = len(code_snippets) if code_snippets else 0

            # Extract code snippet metadata for direct access
            code_snippet_ids = []
            if code_snippets:
                for snippet in code_snippets:
                    if isinstance(snippet, dict) and "id" in snippet:
                        snippet_id = snippet["id"]

                        # Use stored preview from database instead of generating at runtime
                        snippet_content = snippet.get("content", "")
                        preview = snippet.get("preview", "")

                        # Calculate basic stats
                        if snippet_content:
                            lines = snippet_content.split("\n")
                            line_count = len([line for line in lines if line.strip()])
                            char_count = len(snippet_content)
                        else:
                            line_count = 0
                            char_count = 0

                        snippet_obj = {
                            "id": snippet_id,
                            "lines": line_count,
                            "chars": char_count,
                            "preview": preview,
                        }
                        code_snippet_ids.append(snippet_obj)

            compact_result = {
                "id": result.get("id"),
                "document_id": result.get("document_id"),
                "title": result.get("title"),
                "summary": display_content,
                "score": result.get("score"),
                "url": result.get("url"),
                "code_snippets_count": code_snippets_count,
                "code_snippet_ids": code_snippet_ids,
                # Note: content_type field removed as it was causing inconsistencies
            }
            compact_results.append(compact_result)

        return {
            "results": compact_results,
            "total": len(compact_results),
            "query": query,
            "mode": mode,
            "execution_time_ms": execution_time_ms,
            "note": "Content summarized for quick scanning. Use get_document for full content.",
        }

    def transform_code_to_compact_format(
        self, results: list[dict[str, Any]], query: str = "", execution_time_ms: int = 0
    ) -> dict[str, Any]:
        """Transform code search results to compact MCP format.

        This is the single source of truth for transforming code search results
        to the compact format used by both CLI and MCP server.

        Args:
            results: List of code search result dictionaries
            query: Original search query
            execution_time_ms: Time taken for search execution

        Returns:
            Dictionary with compact code search results
        """
        compact_results = []

        for result in results:
            # Use content directly since code snippets are already concise
            display_content = result.get("content", "")

            # Truncate very long code snippets
            if len(display_content) > 500:
                display_content = display_content[:500] + "..."

            # Calculate line count from content
            line_count = len(display_content.split("\n")) if display_content else 0

            compact_result = {
                "id": result.get("id"),
                "document_id": result.get("document_id"),
                "content": display_content,
                "score": result.get("score"),
                "url": result.get("url"),
                "line_count": line_count,
                # Note: All problematic fields (snippet_type, content_type, title) removed
            }
            compact_results.append(compact_result)

        return {
            "results": compact_results,
            "total": len(compact_results),
            "query": query,
            "mode": "hybrid",
            "execution_time_ms": execution_time_ms,
            "note": "Code search using voyage-code-3 embeddings for enhanced code understanding.",
        }

    def clean_search_result_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Clean and organize metadata into consistent structure for search results.

        Args:
            metadata: Raw metadata dictionary

        Returns:
            Cleaned and organized metadata dictionary
        """
        # For now, return simplified metadata to avoid breaking the system
        return {
            "document": metadata.get("document", {}),
            "chunk": metadata.get("chunk", {}),
            "code_snippets": metadata.get("code_snippets", []),
        }

    def standardize_search_result_fields(
        self, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Standardize field names and structures across all search results.

        Args:
            result: Raw search result dictionary

        Returns:
            Standardized search result dictionary
        """
        # Clean metadata
        metadata = result.get("metadata", {})
        clean_metadata = self.clean_search_result_metadata(metadata)

        # Remove problematic fields that cause inconsistencies
        problematic_fields = ["snippet_type", "content_type", "type"]

        standardized = {}
        for key, value in result.items():
            if key not in problematic_fields:
                standardized[key] = value

        # Ensure metadata is clean
        standardized["metadata"] = clean_metadata

        return standardized

    def standardize_code_search_result_fields(
        self, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Standardize field names and structures for code search results.

        Args:
            result: Raw code search result dictionary

        Returns:
            Standardized code search result dictionary
        """
        # Clean metadata for code snippets
        metadata = result.get("metadata", {})
        clean_metadata = {}
        for key, value in metadata.items():
            # Skip deprecated fields that were used for chunks but not code snippets
            if key not in [
                "language",
                "start_line",
                "end_line",
                "char_start",
                "char_end",
                "type",
                "chunk_index",
            ]:
                clean_metadata[key] = value

        # Remove problematic fields
        problematic_fields = ["snippet_type", "content_type", "title", "type"]

        standardized = {}
        for key, value in result.items():
            if key not in problematic_fields:
                standardized[key] = value

        # Ensure metadata is clean
        standardized["metadata"] = clean_metadata

        # Ensure line_count is present
        if "line_count" not in standardized and "content" in standardized:
            content = standardized["content"]
            standardized["line_count"] = len(content.split("\n")) if content else 0

        return standardized


# Global instance for easy access
_transformation_service = None


def get_transformation_service(summarization_service=None) -> TransformationService:
    """Get the global transformation service instance.

    Args:
        summarization_service: Optional service for generating summaries

    Returns:
        TransformationService instance
    """
    global _transformation_service
    if _transformation_service is None:
        _transformation_service = TransformationService(summarization_service)
    return _transformation_service


__all__ = ["TransformationService", "get_transformation_service"]
