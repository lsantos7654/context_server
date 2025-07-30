"""Chunk CRUD operations."""

import json
import uuid
from ..utils import convert_embedding_to_postgres, parse_metadata, format_uuid, parse_uuid


class ChunkManager:
    """Manages chunk-related database operations."""
    
    def __init__(self):
        self.pool = None
    
    async def create_chunk(
        self,
        document_id: str,
        context_id: str,
        content: str,
        embedding: list[float],
        chunk_index: int,
        metadata: dict = None,
        tokens: int = None,
        summary: str = None,
        summary_model: str = None,
        start_line: int = None,
        end_line: int = None,
        char_start: int = None,
        char_end: int = None,
        is_code: bool = False,
    ) -> str:
        """Create a new chunk with embedding and line tracking."""
        async with self.pool.acquire() as conn:
            # Convert embedding list to PostgreSQL vector format
            embedding_str = convert_embedding_to_postgres(embedding)

            # Choose the appropriate embedding column
            if is_code:
                chunk_id = await conn.fetchval(
                    """
                    INSERT INTO chunks (document_id, context_id, content, summary, summary_model, code_embedding, chunk_index, metadata, tokens, start_line, end_line, char_start, char_end)
                    VALUES ($1, $2, $3, $4, $5, $6::halfvec, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                        content = EXCLUDED.content,
                        summary = EXCLUDED.summary,
                        summary_model = EXCLUDED.summary_model,
                        code_embedding = EXCLUDED.code_embedding,
                        metadata = EXCLUDED.metadata,
                        tokens = EXCLUDED.tokens,
                        start_line = EXCLUDED.start_line,
                        end_line = EXCLUDED.end_line,
                        char_start = EXCLUDED.char_start,
                        char_end = EXCLUDED.char_end
                    RETURNING id
                """,
                uuid.UUID(document_id),
                uuid.UUID(context_id),
                content,
                summary,
                summary_model,
                embedding_str,
                chunk_index,
                json.dumps(metadata or {}),
                tokens,
                start_line,
                end_line,
                char_start,
                char_end,
            )
            else:
                chunk_id = await conn.fetchval(
                    """
                    INSERT INTO chunks (document_id, context_id, content, summary, summary_model, text_embedding, chunk_index, metadata, tokens, start_line, end_line, char_start, char_end)
                    VALUES ($1, $2, $3, $4, $5, $6::halfvec, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                        content = EXCLUDED.content,
                        summary = EXCLUDED.summary,
                        summary_model = EXCLUDED.summary_model,
                        text_embedding = EXCLUDED.text_embedding,
                        metadata = EXCLUDED.metadata,
                        tokens = EXCLUDED.tokens,
                        start_line = EXCLUDED.start_line,
                        end_line = EXCLUDED.end_line,
                        char_start = EXCLUDED.char_start,
                        char_end = EXCLUDED.char_end
                    RETURNING id
                """,
                uuid.UUID(document_id),
                uuid.UUID(context_id),
                content,
                summary,
                summary_model,
                embedding_str,
                chunk_index,
                json.dumps(metadata or {}),
                tokens,
                start_line,
                end_line,
                char_start,
                char_end,
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
    
    async def get_chunk_by_id(
        self, chunk_id: str, context_id: str = None
    ) -> dict | None:
        """Get a specific chunk by ID with full content and metadata."""
        async with self.pool.acquire() as conn:
            query = """
                SELECT
                    c.id, c.content, c.summary, c.summary_model, d.title, d.url, d.metadata as doc_metadata,
                    c.metadata as chunk_metadata, c.chunk_index, d.id as document_id,
                    c.start_line, c.end_line, c.char_start, c.char_end, c.tokens,
                    LENGTH(d.content) as parent_page_size,
                    d.chunk_count as parent_total_chunks,
                    c.created_at, c.context_id
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.id = $1
            """
            params = [uuid.UUID(chunk_id)]

            if context_id:
                query += " AND c.context_id = $2"
                params.append(uuid.UUID(context_id))

            row = await conn.fetchrow(query, *params)

            if not row:
                return None

            return {
                "id": format_uuid(row["id"]),
                "document_id": format_uuid(row["document_id"]),
                "content": row["content"],
                "summary": row["summary"],
                "summary_model": row["summary_model"],
                "title": row["title"],
                "url": row["url"],
                "chunk_index": row["chunk_index"],
                "tokens": row["tokens"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "char_start": row["char_start"],
                "char_end": row["char_end"],
                "created_at": row["created_at"],
                "metadata": parse_metadata(row["chunk_metadata"]),
                "doc_metadata": parse_metadata(row["doc_metadata"]),
                "parent_page_size": row["parent_page_size"],
                "parent_total_chunks": row["parent_total_chunks"],
            }


__all__ = ["ChunkManager"]