"""Redis-based document caching service."""

import json
import logging
import os
from typing import Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class DocumentCacheService:
    """Service for caching full documents in Redis."""

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis: Optional[redis.Redis] = None
        self.default_ttl = 3600  # 1 hour default TTL

    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis = redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            await self.redis.ping()
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")

    async def is_healthy(self) -> bool:
        """Check if Redis is healthy."""
        if not self.redis:
            return False
        try:
            await self.redis.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    async def cache_document(
        self, document_id: str, content: str, ttl: int = None
    ) -> bool:
        """
        Cache a document's full content.

        Args:
            document_id: Unique document identifier
            content: Full document content
            ttl: Time to live in seconds (default: 1 hour)

        Returns:
            True if cached successfully, False otherwise
        """
        if not self.redis:
            logger.warning("Redis not initialized, cannot cache document")
            return False

        try:
            # Split content into lines for efficient line-based lookups
            lines = content.splitlines()

            # Cache the full content
            content_key = f"doc:{document_id}:content"
            await self.redis.set(content_key, content, ex=ttl or self.default_ttl)

            # Cache line count metadata
            lines_key = f"doc:{document_id}:lines"
            lines_data = {"total_lines": len(lines), "lines": lines}
            await self.redis.set(
                lines_key, json.dumps(lines_data), ex=ttl or self.default_ttl
            )

            logger.debug(f"Cached document {document_id} with {len(lines)} lines")
            return True

        except Exception as e:
            logger.error(f"Failed to cache document {document_id}: {e}")
            return False

    async def get_document_content(self, document_id: str) -> Optional[str]:
        """
        Get full document content from cache.

        Args:
            document_id: Unique document identifier

        Returns:
            Full document content or None if not cached
        """
        if not self.redis:
            return None

        try:
            content_key = f"doc:{document_id}:content"
            content = await self.redis.get(content_key)
            return content
        except Exception as e:
            logger.error(f"Failed to get document {document_id} from cache: {e}")
            return None

    async def get_document_lines(self, document_id: str) -> Optional[list[str]]:
        """
        Get document as list of lines from cache.

        Args:
            document_id: Unique document identifier

        Returns:
            List of lines or None if not cached
        """
        if not self.redis:
            return None

        try:
            lines_key = f"doc:{document_id}:lines"
            lines_data = await self.redis.get(lines_key)
            if lines_data:
                data = json.loads(lines_data)
                return data.get("lines", [])
            return None
        except Exception as e:
            logger.error(f"Failed to get lines for document {document_id}: {e}")
            return None

    async def get_line_range(
        self, document_id: str, start_line: int, end_line: int
    ) -> Optional[str]:
        """
        Get a specific range of lines from a cached document.

        Args:
            document_id: Unique document identifier
            start_line: Starting line number (0-based)
            end_line: Ending line number (inclusive, 0-based)

        Returns:
            Content of specified lines joined with newlines, or None if not cached
        """
        lines = await self.get_document_lines(document_id)
        if not lines:
            return None

        try:
            # Ensure bounds are valid
            start_line = max(0, start_line)
            end_line = min(len(lines) - 1, end_line)

            if start_line > end_line:
                return ""

            # Extract line range
            selected_lines = lines[start_line : end_line + 1]
            return "\n".join(selected_lines)

        except Exception as e:
            logger.error(f"Failed to get line range for document {document_id}: {e}")
            return None

    async def get_line_count(self, document_id: str) -> Optional[int]:
        """
        Get total line count for a cached document.

        Args:
            document_id: Unique document identifier

        Returns:
            Total number of lines or None if not cached
        """
        if not self.redis:
            return None

        try:
            lines_key = f"doc:{document_id}:lines"
            lines_data = await self.redis.get(lines_key)
            if lines_data:
                data = json.loads(lines_data)
                return data.get("total_lines", 0)
            return None
        except Exception as e:
            logger.error(f"Failed to get line count for document {document_id}: {e}")
            return None

    async def is_document_cached(self, document_id: str) -> bool:
        """
        Check if a document is cached.

        Args:
            document_id: Unique document identifier

        Returns:
            True if document is cached, False otherwise
        """
        if not self.redis:
            return False

        try:
            content_key = f"doc:{document_id}:content"
            exists = await self.redis.exists(content_key)
            return bool(exists)
        except Exception as e:
            logger.error(f"Failed to check if document {document_id} is cached: {e}")
            return False

    async def evict_document(self, document_id: str) -> bool:
        """
        Remove a document from cache.

        Args:
            document_id: Unique document identifier

        Returns:
            True if evicted successfully, False otherwise
        """
        if not self.redis:
            return False

        try:
            content_key = f"doc:{document_id}:content"
            lines_key = f"doc:{document_id}:lines"

            # Delete both keys
            deleted = await self.redis.delete(content_key, lines_key)
            logger.debug(f"Evicted document {document_id} from cache")
            return deleted > 0

        except Exception as e:
            logger.error(f"Failed to evict document {document_id}: {e}")
            return False

    async def clear_all(self) -> bool:
        """
        Clear all cached documents.

        Returns:
            True if cleared successfully, False otherwise
        """
        if not self.redis:
            return False

        try:
            # Find all document keys
            doc_keys = []
            async for key in self.redis.scan_iter(match="doc:*"):
                doc_keys.append(key)

            if doc_keys:
                await self.redis.delete(*doc_keys)
                logger.info(f"Cleared {len(doc_keys)} document cache entries")

            return True

        except Exception as e:
            logger.error(f"Failed to clear document cache: {e}")
            return False

    async def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.redis:
            return {"error": "Redis not initialized"}

        try:
            # Count document cache entries
            doc_count = 0
            async for _ in self.redis.scan_iter(match="doc:*:content"):
                doc_count += 1

            # Get Redis info
            info = await self.redis.info()

            return {
                "cached_documents": doc_count,
                "redis_memory_used": info.get("used_memory_human"),
                "redis_connected_clients": info.get("connected_clients"),
                "redis_uptime": info.get("uptime_in_seconds"),
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}
