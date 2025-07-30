"""Document CRUD operations - placeholder for full implementation."""

import json
import uuid


class DocumentManager:
    """Manages document-related database operations."""
    
    def __init__(self):
        self.pool = None  # Will be injected by DatabaseManager
    
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
    ) -> str:
        """Create a new document."""
        # Normalize URL to prevent duplicates
        normalized_url = self._normalize_url(url)
        
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
                    normalized_url,
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
    
    async def get_documents(self, *args, **kwargs):
        """Placeholder - implement from storage.py"""
        raise NotImplementedError("DocumentManager methods need to be implemented")
    
    async def delete_documents(self, *args, **kwargs):
        """Placeholder - implement from storage.py"""
        raise NotImplementedError("DocumentManager methods need to be implemented")
    
    async def get_document_by_id(
        self, context_id: str, document_id: str
    ) -> dict | None:
        """Get document content by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    d.id, d.title, d.url, d.content, d.metadata,
                    d.indexed_at, d.source_type, d.chunk_count
                FROM documents d
                WHERE d.context_id = $1 AND d.id = $2
                """,
                uuid.UUID(context_id),
                uuid.UUID(document_id),
            )

            if not row:
                return None

            return {
                "id": str(row["id"]),
                "title": row["title"],
                "url": row["url"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "created_at": row["indexed_at"].isoformat()
                if row["indexed_at"]
                else None,
                "source_type": row["source_type"],
                "chunk_count": row["chunk_count"] or 0,
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
                "id": str(row["id"]),
                "title": row["title"],
                "url": row["url"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "created_at": row["indexed_at"].isoformat()
                if row["indexed_at"]
                else None,
                "source_type": row["source_type"],
                "chunk_count": row["chunk_count"] or 0,
            }


__all__ = ["DocumentManager"]