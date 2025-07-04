"""PostgreSQL + pgvector storage manager for Context Server."""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Optional

import asyncpg
import numpy as np
from asyncpg import Pool

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL connection with pgvector for vector storage."""

    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", "postgresql://user:password@localhost:5432/context_server"
        )
        self.pool: Pool | None = None

    async def initialize(self):
        """Initialize database connection and create required tables."""
        max_retries = 30
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                self.pool = await asyncpg.create_pool(
                    self.database_url, min_size=2, max_size=10, command_timeout=30
                )

                # Create vector extension and base tables
                await self._create_base_schema()
                logger.info("Database initialized successfully")
                return

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.info(
                        f"Database connection attempt {attempt + 1} failed, retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"Failed to initialize database after {max_retries} attempts: {e}"
                    )
                    raise

    async def close(self):
        """Close database connection."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection closed")

    async def is_healthy(self) -> bool:
        """Check if database is healthy."""
        if not self.pool:
            return False

        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def _create_base_schema(self):
        """Create base database schema."""
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create contexts table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS contexts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(100) UNIQUE NOT NULL,
                    description TEXT DEFAULT '',
                    embedding_model VARCHAR(100) DEFAULT 'text-embedding-3-small',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    document_count INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 0
                )
            """
            )

            # Create documents table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    context_id UUID NOT NULL REFERENCES contexts(id) ON DELETE CASCADE,
                    url TEXT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    chunk_count INTEGER DEFAULT 0,
                    source_type VARCHAR(20) NOT NULL,
                    UNIQUE(context_id, url)
                )
            """
            )

            # Create chunks table with vector embeddings
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    context_id UUID NOT NULL REFERENCES contexts(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    embedding vector(1536),  -- OpenAI embedding dimension
                    chunk_index INTEGER NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    tokens INTEGER,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(document_id, chunk_index)
                )
            """
            )

            # Create indexes for performance
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_context_id ON chunks(context_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_context_id ON documents(context_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url)"
            )

            # Create full-text search indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_content_fts ON chunks USING gin(to_tsvector('english', content))"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_content_fts ON documents USING gin(to_tsvector('english', content))"
            )

            logger.info("Database schema created successfully")

    # Context management methods
    async def create_context(
        self,
        name: str,
        description: str = "",
        embedding_model: str = "text-embedding-3-small",
    ) -> dict:
        """Create a new context."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO contexts (name, description, embedding_model)
                VALUES ($1, $2, $3)
                RETURNING id, name, description, embedding_model, created_at, updated_at
            """,
                name,
                description,
                embedding_model,
            )

            return {
                "id": str(row["id"]),
                "name": row["name"],
                "description": row["description"],
                "embedding_model": row["embedding_model"],
                "created_at": row["created_at"],
                "document_count": 0,
                "size_mb": 0.0,
                "last_updated": row["updated_at"],
            }

    async def get_contexts(self) -> list[dict]:
        """Get all contexts."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    c.id, c.name, c.description, c.embedding_model,
                    c.created_at, c.updated_at, c.document_count,
                    COALESCE(pg_size_pretty(pg_total_relation_size('chunks'))::text, '0 bytes') as size_mb
                FROM contexts c
                ORDER BY c.created_at DESC
            """
            )

            return [
                {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "description": row["description"],
                    "embedding_model": row["embedding_model"],
                    "created_at": row["created_at"],
                    "document_count": row["document_count"],
                    "size_mb": 0.0,  # TODO: Calculate actual size
                    "last_updated": row["updated_at"],
                }
                for row in rows
            ]

    async def get_context_by_name(self, name: str) -> dict | None:
        """Get context by name."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, name, description, embedding_model, created_at, updated_at, document_count
                FROM contexts WHERE name = $1
            """,
                name,
            )

            if not row:
                return None

            return {
                "id": str(row["id"]),
                "name": row["name"],
                "description": row["description"],
                "embedding_model": row["embedding_model"],
                "created_at": row["created_at"],
                "document_count": row["document_count"],
                "size_mb": 0.0,
                "last_updated": row["updated_at"],
            }

    async def delete_context(self, context_id: str) -> bool:
        """Delete a context and all its data."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM contexts WHERE id = $1", uuid.UUID(context_id)
            )
            return result == "DELETE 1"

    # Document management methods
    async def create_document(
        self,
        context_id: str,
        url: str,
        title: str,
        content: str,
        metadata: dict,
        source_type: str,
    ) -> str:
        """Create a new document."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Insert document
                doc_id = await conn.fetchval(
                    """
                    INSERT INTO documents (context_id, url, title, content, metadata, source_type)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (context_id, url) DO UPDATE SET
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        indexed_at = NOW()
                    RETURNING id
                """,
                    uuid.UUID(context_id),
                    url,
                    title,
                    content,
                    json.dumps(metadata),
                    source_type,
                )

                # Update context document count
                await conn.execute(
                    """
                    UPDATE contexts
                    SET document_count = (SELECT COUNT(*) FROM documents WHERE context_id = $1),
                        updated_at = NOW()
                    WHERE id = $1
                """,
                    uuid.UUID(context_id),
                )

                return str(doc_id)

    async def get_documents(
        self, context_id: str, offset: int = 0, limit: int = 50
    ) -> dict:
        """Get documents in a context."""
        async with self.pool.acquire() as conn:
            # Get total count
            total = await conn.fetchval(
                """
                SELECT COUNT(*) FROM documents WHERE context_id = $1
            """,
                uuid.UUID(context_id),
            )

            # Get documents
            rows = await conn.fetch(
                """
                SELECT id, url, title, indexed_at, chunk_count, metadata
                FROM documents
                WHERE context_id = $1
                ORDER BY indexed_at DESC
                OFFSET $2 LIMIT $3
            """,
                uuid.UUID(context_id),
                offset,
                limit,
            )

            documents = [
                {
                    "id": str(row["id"]),
                    "url": row["url"],
                    "title": row["title"],
                    "indexed_at": row["indexed_at"],
                    "chunks": row["chunk_count"] or 0,
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                }
                for row in rows
            ]

            return {
                "documents": documents,
                "total": total,
                "offset": offset,
                "limit": limit,
            }

    async def delete_documents(self, context_id: str, document_ids: list[str]) -> int:
        """Delete documents from a context."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Delete documents
                result = await conn.execute(
                    """
                    DELETE FROM documents
                    WHERE context_id = $1 AND id = ANY($2::uuid[])
                """,
                    uuid.UUID(context_id),
                    [uuid.UUID(doc_id) for doc_id in document_ids],
                )

                # Update context document count
                await conn.execute(
                    """
                    UPDATE contexts
                    SET document_count = (SELECT COUNT(*) FROM documents WHERE context_id = $1),
                        updated_at = NOW()
                    WHERE id = $1
                """,
                    uuid.UUID(context_id),
                )

                # Extract number of deleted rows
                deleted_count = int(result.split()[-1]) if result else 0
                return deleted_count

    # Chunk management methods
    async def create_chunk(
        self,
        document_id: str,
        context_id: str,
        content: str,
        embedding: list[float],
        chunk_index: int,
        metadata: dict = None,
        tokens: int = None,
    ) -> str:
        """Create a new chunk with embedding."""
        async with self.pool.acquire() as conn:
            # Convert embedding list to PostgreSQL vector format
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            chunk_id = await conn.fetchval(
                """
                INSERT INTO chunks (document_id, context_id, content, embedding, chunk_index, metadata, tokens)
                VALUES ($1, $2, $3, $4::vector, $5, $6, $7)
                ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    tokens = EXCLUDED.tokens
                RETURNING id
            """,
                uuid.UUID(document_id),
                uuid.UUID(context_id),
                content,
                embedding_str,
                chunk_index,
                json.dumps(metadata or {}),
                tokens,
            )

            return str(chunk_id)

    async def update_document_chunk_count(self, document_id: str):
        """Update document chunk count."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE documents
                SET chunk_count = (SELECT COUNT(*) FROM chunks WHERE document_id = $1)
                WHERE id = $1
            """,
                uuid.UUID(document_id),
            )

    # Search methods
    async def vector_search(
        self,
        context_id: str,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.7,
    ) -> list[dict]:
        """Perform vector similarity search."""
        async with self.pool.acquire() as conn:
            # Convert embedding list to PostgreSQL vector format
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            rows = await conn.fetch(
                """
                SELECT
                    c.id, c.content, d.title, d.url, d.metadata as doc_metadata,
                    c.metadata as chunk_metadata, c.chunk_index,
                    1 - (c.embedding <=> $2::vector) as similarity
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.context_id = $1
                    AND 1 - (c.embedding <=> $2::vector) > $3
                ORDER BY c.embedding <=> $2::vector
                LIMIT $4
            """,
                uuid.UUID(context_id),
                embedding_str,
                min_similarity,
                limit,
            )

            return [
                {
                    "id": str(row["id"]),
                    "content": row["content"],
                    "title": row["title"],
                    "url": row["url"],
                    "score": float(row["similarity"]),
                    "metadata": {
                        **(
                            json.loads(row["doc_metadata"])
                            if row["doc_metadata"]
                            else {}
                        ),
                        **(
                            json.loads(row["chunk_metadata"])
                            if row["chunk_metadata"]
                            else {}
                        ),
                    },
                    "chunk_index": row["chunk_index"],
                }
                for row in rows
            ]

    async def fulltext_search(
        self, context_id: str, query: str, limit: int = 10
    ) -> list[dict]:
        """Perform full-text search."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    c.id, c.content, d.title, d.url, d.metadata as doc_metadata,
                    c.metadata as chunk_metadata, c.chunk_index,
                    ts_rank(to_tsvector('english', c.content), plainto_tsquery('english', $2)) as score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.context_id = $1
                    AND to_tsvector('english', c.content) @@ plainto_tsquery('english', $2)
                ORDER BY score DESC
                LIMIT $3
            """,
                uuid.UUID(context_id),
                query,
                limit,
            )

            return [
                {
                    "id": str(row["id"]),
                    "content": row["content"],
                    "title": row["title"],
                    "url": row["url"],
                    "score": float(row["score"]),
                    "metadata": {
                        **(
                            json.loads(row["doc_metadata"])
                            if row["doc_metadata"]
                            else {}
                        ),
                        **(
                            json.loads(row["chunk_metadata"])
                            if row["chunk_metadata"]
                            else {}
                        ),
                    },
                    "chunk_index": row["chunk_index"],
                }
                for row in rows
            ]
