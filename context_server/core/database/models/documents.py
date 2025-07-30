"""Document CRUD operations."""

import json
import uuid
from ..utils import parse_metadata, format_uuid, parse_uuid


class DocumentManager:
    """Manages document-related database operations."""
    
    def __init__(self):
        self.pool = None
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL to prevent duplicates (strip trailing slashes, etc.)."""
        if not url:
            return url
        
        # Strip trailing slashes
        normalized = url.rstrip('/')
        
        # Ensure we have a valid URL
        if not normalized:
            return url
            
        return normalized
    
    async def create_document(
        self,
        context_id: str,
        url: str,
        title: str,
        content: str,
        metadata: dict,
        source_type: str,
        document_type: str = "original",
    ) -> str:
        """Create a new document."""
        # Normalize URL to prevent duplicates
        normalized_url = self._normalize_url(url)
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Insert document
                doc_id = await conn.fetchval(
                    """
                    INSERT INTO documents (context_id, url, title, content, metadata, source_type, document_type)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (context_id, url, document_type) DO UPDATE SET
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        indexed_at = NOW()
                    RETURNING id
                """,
                    uuid.UUID(context_id),
                    normalized_url,
                    title,
                    content,
                    json.dumps(metadata),
                    source_type,
                    document_type,
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
    
    async def get_documents(self, context_id: str, offset: int = 0, limit: int = 50) -> dict:
        """Get documents in a context with pagination."""
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
                    "id": format_uuid(row["id"]),
                    "url": row["url"],
                    "title": row["title"],
                    "indexed_at": row["indexed_at"],
                    "chunks": row["chunk_count"] or 0,
                    "metadata": parse_metadata(row["metadata"]),
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
        """Delete documents from a context with transaction safety."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Delete documents
                result = await conn.execute(
                    """
                    DELETE FROM documents
                    WHERE context_id = $1 AND id = ANY($2::uuid[])
                    """,
                    uuid.UUID(context_id),
                    [parse_uuid(doc_id) for doc_id in document_ids],
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
    
    async def get_document_by_id(
        self, context_id: str, document_id: str, document_type: str = "original"
    ) -> dict | None:
        """Get document content by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    d.id, d.title, d.url, d.content, d.metadata,
                    d.indexed_at, d.source_type, d.chunk_count, d.document_type
                FROM documents d
                WHERE d.context_id = $1 AND d.id = $2 AND d.document_type = $3
                """,
                uuid.UUID(context_id),
                uuid.UUID(document_id),
                document_type,
            )

            if not row:
                return None

            return {
                "id": format_uuid(row["id"]),
                "title": row["title"],
                "url": row["url"],
                "content": row["content"],
                "metadata": parse_metadata(row["metadata"]),
                "created_at": row["indexed_at"].isoformat()
                if row["indexed_at"]
                else None,
                "source_type": row["source_type"],
                "chunk_count": row["chunk_count"] or 0,
                "document_type": row["document_type"],
            }
    
    async def get_document_content_by_id(self, document_id: str) -> dict | None:
        """Get document content by ID only (for expansion service)."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    d.id, d.title, d.url, d.content, d.metadata,
                    d.indexed_at, d.source_type, d.chunk_count
                FROM documents d
                WHERE d.id = $1
                """,
                uuid.UUID(document_id),
            )

            if not row:
                return None

            return {
                "id": format_uuid(row["id"]),
                "title": row["title"],
                "url": row["url"],
                "content": row["content"],
                "metadata": parse_metadata(row["metadata"]),
                "created_at": row["indexed_at"].isoformat()
                if row["indexed_at"]
                else None,
                "source_type": row["source_type"],
                "chunk_count": row["chunk_count"] or 0,
            }


__all__ = ["DocumentManager"]