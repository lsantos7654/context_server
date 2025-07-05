"""Line-based context expansion service using Redis cache and database."""

import logging
from typing import Optional

from .cache import DocumentCacheService
from .storage import DatabaseManager

logger = logging.getLogger(__name__)


class ContextExpansionService:
    """Service for expanding search results with line-based context."""

    def __init__(self, db: DatabaseManager, cache_service: DocumentCacheService):
        self.db = db
        self.cache_service = cache_service

    async def expand_search_result(
        self, result: dict, expand_lines: int, prefer_boundaries: bool = True
    ) -> dict:
        """
        Expand a search result with surrounding lines from the original document.

        Args:
            result: Search result dictionary with chunk metadata
            expand_lines: Number of lines to expand above and below
            prefer_boundaries: Try to respect paragraph/section boundaries

        Returns:
            Updated result dictionary with expanded content
        """
        if expand_lines <= 0:
            return result

        document_id = result.get("document_id")
        if not document_id:
            logger.warning("No document_id in search result, cannot expand context")
            return result

        # Get chunk line information from database if not in result
        start_line = result.get("start_line")
        end_line = result.get("end_line")

        if start_line is None or end_line is None:
            # Try to get line info from database
            chunk_info = await self._get_chunk_line_info(result.get("id"))
            if chunk_info:
                start_line = chunk_info.get("start_line")
                end_line = chunk_info.get("end_line")
            else:
                logger.warning(
                    f"No line information available for chunk {result.get('id')}"
                )
                return result

        # Calculate expansion range
        expand_start = max(0, start_line - expand_lines)
        expand_end = end_line + expand_lines

        # Try to get expanded content from cache first
        expanded_content = await self._get_expanded_content_cached(
            document_id, expand_start, expand_end
        )

        if expanded_content is None:
            # Fallback to database retrieval
            expanded_content = await self._get_expanded_content_database(
                document_id, expand_start, expand_end
            )

        if expanded_content is None:
            logger.warning(
                f"Could not retrieve expanded content for document {document_id}"
            )
            return result

        # Apply smart boundary detection if requested
        if prefer_boundaries:
            expanded_content = self._adjust_for_boundaries(
                expanded_content, result["content"]
            )

        # Update result with expanded content
        expanded_result = result.copy()
        expanded_result.update(
            {
                "content": expanded_content,
                "content_type": "expanded_chunk",
                "original_content": result["content"],
                "expansion_info": {
                    "original_lines": f"{start_line}-{end_line}",
                    "expanded_lines": f"{expand_start}-{expand_end}",
                    "lines_added": expand_lines * 2,
                    "method": "line_based",
                },
            }
        )

        return expanded_result

    async def _get_chunk_line_info(self, chunk_id: str) -> Optional[dict]:
        """Get line information for a chunk from the database."""
        try:
            import uuid

            async with self.db.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT start_line, end_line, char_start, char_end FROM chunks WHERE id = $1",
                    uuid.UUID(chunk_id),
                )
                if row:
                    return {
                        "start_line": row["start_line"],
                        "end_line": row["end_line"],
                        "char_start": row["char_start"],
                        "char_end": row["char_end"],
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get chunk line info for {chunk_id}: {e}")
            return None

    async def _get_expanded_content_cached(
        self, document_id: str, start_line: int, end_line: int
    ) -> Optional[str]:
        """Get expanded content from Redis cache."""
        try:
            return await self.cache_service.get_line_range(
                document_id, start_line, end_line
            )
        except Exception as e:
            logger.debug(f"Cache miss for document {document_id}: {e}")
            return None

    async def _get_expanded_content_database(
        self, document_id: str, start_line: int, end_line: int
    ) -> Optional[str]:
        """
        Get expanded content from database and cache it.

        This is the fallback when Redis cache is not available or cache miss occurs.
        """
        try:
            # Get full document content from database
            document = await self.db.get_document_content_by_id(document_id)
            if not document:
                return None

            full_content = document["content"]

            # Cache the document for future requests
            await self.cache_service.cache_document(document_id, full_content)

            # Extract the requested line range
            lines = full_content.splitlines()
            if start_line >= len(lines):
                return None

            end_line = min(end_line, len(lines) - 1)
            selected_lines = lines[start_line : end_line + 1]

            return "\n".join(selected_lines)

        except Exception as e:
            logger.error(
                f"Failed to get expanded content from database for document {document_id}: {e}"
            )
            return None

    def _adjust_for_boundaries(
        self, expanded_content: str, original_content: str
    ) -> str:
        """
        Adjust expanded content to respect natural boundaries when possible.

        This tries to avoid cutting off mid-sentence or mid-paragraph.
        """
        try:
            lines = expanded_content.split("\n")

            # Find the original content within the expanded content
            original_lines = original_content.split("\n")
            if not original_lines:
                return expanded_content

            # Look for the original content in the expanded content
            original_start = None
            for i in range(len(lines) - len(original_lines) + 1):
                if lines[i : i + len(original_lines)] == original_lines:
                    original_start = i
                    break

            if original_start is None:
                # Can't find original content, return as-is
                return expanded_content

            # Try to extend to natural boundaries
            start_idx = 0
            end_idx = len(lines) - 1

            # Look backwards for paragraph boundary (empty line or section header)
            for i in range(original_start - 1, -1, -1):
                line = lines[i].strip()
                if not line:  # Empty line indicates paragraph boundary
                    start_idx = i + 1
                    break
                elif line.startswith("#") or line.startswith("##"):  # Markdown headers
                    start_idx = i
                    break

            # Look forwards for paragraph boundary
            original_end = original_start + len(original_lines) - 1
            for i in range(original_end + 1, len(lines)):
                line = lines[i].strip()
                if not line:  # Empty line indicates paragraph boundary
                    end_idx = i - 1
                    break
                elif line.startswith("#") or line.startswith("##"):  # Markdown headers
                    end_idx = i - 1
                    break

            # Return content within natural boundaries
            return "\n".join(lines[start_idx : end_idx + 1])

        except Exception as e:
            logger.debug(
                f"Boundary adjustment failed, returning original expanded content: {e}"
            )
            return expanded_content

    async def expand_search_results(
        self, results: list[dict], expand_lines: int, prefer_boundaries: bool = True
    ) -> list[dict]:
        """
        Expand multiple search results with line-based context.

        Args:
            results: List of search result dictionaries
            expand_lines: Number of lines to expand above and below
            prefer_boundaries: Try to respect paragraph/section boundaries

        Returns:
            List of expanded search results
        """
        if expand_lines <= 0:
            return results

        expanded_results = []
        for result in results:
            try:
                expanded_result = await self.expand_search_result(
                    result, expand_lines, prefer_boundaries
                )
                expanded_results.append(expanded_result)
            except Exception as e:
                logger.error(f"Failed to expand search result: {e}")
                # Include original result if expansion fails
                expanded_results.append(result)

        return expanded_results

    async def get_document_preview(
        self, document_id: str, around_line: int, context_lines: int = 10
    ) -> Optional[dict]:
        """
        Get a preview of a document around a specific line.

        Args:
            document_id: Document identifier
            around_line: Line number to center the preview around
            context_lines: Number of lines above and below to include

        Returns:
            Preview information with content and metadata
        """
        start_line = max(0, around_line - context_lines)
        end_line = around_line + context_lines

        content = await self._get_expanded_content_cached(
            document_id, start_line, end_line
        )

        if content is None:
            content = await self._get_expanded_content_database(
                document_id, start_line, end_line
            )

        if content is None:
            return None

        total_lines = await self.cache_service.get_line_count(document_id)
        if total_lines is None:
            # Estimate from content
            total_lines = len(content.split("\n")) + start_line

        return {
            "content": content,
            "start_line": start_line,
            "end_line": end_line,
            "center_line": around_line,
            "total_lines": total_lines,
            "preview_info": {
                "lines_shown": end_line - start_line + 1,
                "context_lines": context_lines,
                "document_id": document_id,
            },
        }
