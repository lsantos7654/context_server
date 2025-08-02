"""Base database manager with common patterns."""

import logging
from abc import ABC
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class DatabaseManagerBase(ABC):
    """Base class for database managers with common patterns."""

    def __init__(self):
        """Initialize the manager with no pool (set by connection manager)."""
        self.pool = None

    @asynccontextmanager
    async def connection(self):
        """Context manager for database connections with error handling."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.pool.acquire() as conn:
            try:
                yield conn
            except Exception as e:
                logger.error(f"Database operation failed: {e}")
                raise

    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions with error handling."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                try:
                    yield conn
                except Exception as e:
                    logger.error(f"Database transaction failed: {e}")
                    raise

    async def execute_query(self, query: str, *args):
        """Execute a query with connection management."""
        async with self.connection() as conn:
            return await conn.execute(query, *args)

    async def fetch_one(self, query: str, *args) -> dict | None:
        """Fetch a single row with connection management."""
        async with self.connection() as conn:
            return await conn.fetchrow(query, *args)

    async def fetch_many(self, query: str, *args) -> list[dict]:
        """Fetch multiple rows with connection management."""
        async with self.connection() as conn:
            return await conn.fetch(query, *args)

    async def fetch_value(self, query: str, *args):
        """Fetch a single value with connection management."""
        async with self.connection() as conn:
            return await conn.fetchval(query, *args)


__all__ = ["DatabaseManagerBase"]