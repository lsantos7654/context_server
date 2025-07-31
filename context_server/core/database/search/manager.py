"""Search operations manager."""

import json
import uuid
from ..utils import parse_metadata, format_uuid, parse_uuid


class SearchManager:
    """Manages all search-related database operations."""
    
    def __init__(self, summarization_service=None):
        self.pool = None
        self.summarization_service = summarization_service
    
    async def vector_search(
        self,
        context_id: str,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.7,
        embedding_type: str = "text",
    ) -> list[dict]:
        """Perform vector similarity search on text or code embeddings."""
        async with self.pool.acquire() as conn:
            # Convert embedding list to PostgreSQL vector format
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            # Determine which embedding column to use based on embedding type
            if embedding_type == "code":
                embedding_column = "c.code_embedding"
                # Use halfvec for code embeddings (2048 dimensions)
                vector_type = "halfvec"
            else:
                embedding_column = "c.text_embedding"
                # Use halfvec for text embeddings (3072 dimensions)
                vector_type = "halfvec"

            rows = await conn.fetch(
                f"""
                SELECT
                    c.id, c.content, c.summary, c.summary_model, d.title, d.url, d.metadata as doc_metadata,
                    c.metadata as chunk_metadata, c.chunk_index, d.id as document_id,
                    c.start_line, c.end_line, c.char_start, c.char_end, c.code_snippet_ids,
                    LENGTH(d.content) as parent_page_size,
                    d.chunk_count as parent_total_chunks,
                    1 - ({embedding_column} <=> $2::{vector_type}) as similarity
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.context_id = $1
                    AND {embedding_column} IS NOT NULL
                    AND 1 - ({embedding_column} <=> $2::{vector_type}) > $3
                ORDER BY {embedding_column} <=> $2::{vector_type}
                LIMIT $4
            """,
                uuid.UUID(context_id),
                embedding_str,
                min_similarity,
                limit,
            )

            # Enhanced result formatting with code snippet information
            chunk_results = []
            for row in rows:
                # Process code_snippet_ids to get actual code snippet data
                code_snippet_ids = row.get("code_snippet_ids") or []
                code_snippets_data = []
                
                if code_snippet_ids:
                    # Fetch code snippet details for these IDs
                    snippet_rows = await conn.fetch(
                        """
                        SELECT id, content, snippet_type, preview
                        FROM code_snippets 
                        WHERE id = ANY($1::uuid[])
                        """,
                        code_snippet_ids
                    )
                    
                    for snippet_row in snippet_rows:
                        code_snippets_data.append({
                            "id": format_uuid(snippet_row["id"]),
                            "content": snippet_row["content"],
                            "snippet_type": snippet_row["snippet_type"],
                            "preview": snippet_row.get("preview", ""),
                        })
                
                # Parse metadata and add code snippets
                chunk_metadata = parse_metadata(row["chunk_metadata"])
                chunk_metadata["code_snippets"] = code_snippets_data
                
                chunk_results.append({
                    "id": format_uuid(row["id"]),
                    "document_id": str(row["document_id"]),
                    "content": row["content"],
                    "summary": row["summary"],
                    "summary_model": row["summary_model"],
                    "title": row["title"],
                    "url": row["url"],
                    "score": float(row["similarity"]),
                    "metadata": chunk_metadata,
                    "start_line": row.get("start_line"),
                    "end_line": row.get("end_line"),
                    "char_start": row.get("char_start"),
                    "char_end": row.get("char_end"),
                })

            return chunk_results
    
    async def fulltext_search(
        self, context_id: str, query: str, limit: int = 10
    ) -> list[dict]:
        """Perform full-text search."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    c.id, c.content, c.summary, c.summary_model, d.title, d.url, d.metadata as doc_metadata,
                    c.metadata as chunk_metadata, c.chunk_index, d.id as document_id,
                    c.start_line, c.end_line, c.char_start, c.char_end, c.code_snippet_ids,
                    LENGTH(d.content) as parent_page_size,
                    d.chunk_count as parent_total_chunks,
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

            # Enhanced result formatting with code snippet information  
            chunk_results = []
            for row in rows:
                # Process code_snippet_ids to get actual code snippet data
                code_snippet_ids = row.get("code_snippet_ids") or []
                code_snippets_data = []
                
                if code_snippet_ids:
                    # Fetch code snippet details for these IDs
                    snippet_rows = await conn.fetch(
                        """
                        SELECT id, content, snippet_type, preview
                        FROM code_snippets 
                        WHERE id = ANY($1::uuid[])
                        """,
                        code_snippet_ids
                    )
                    
                    for snippet_row in snippet_rows:
                        code_snippets_data.append({
                            "id": format_uuid(snippet_row["id"]),
                            "content": snippet_row["content"],
                            "snippet_type": snippet_row["snippet_type"],
                            "preview": snippet_row.get("preview", ""),
                        })
                
                # Parse metadata and add code snippets
                chunk_metadata = parse_metadata(row["chunk_metadata"])
                chunk_metadata["code_snippets"] = code_snippets_data
                
                chunk_results.append({
                    "id": format_uuid(row["id"]),
                    "document_id": str(row["document_id"]),
                    "content": row["content"],
                    "summary": row["summary"],
                    "summary_model": row["summary_model"],
                    "title": row["title"],
                    "url": row["url"],
                    "score": float(row["score"]),
                    "metadata": chunk_metadata,
                    "start_line": row.get("start_line"),
                    "end_line": row.get("end_line"),
                    "char_start": row.get("char_start"),
                    "char_end": row.get("char_end"),
                })

            return chunk_results
    
    async def vector_search_code_snippets(
        self,
        context_id: str,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.7,
    ) -> list[dict]:
        """Perform vector similarity search on code snippets."""
        async with self.pool.acquire() as conn:
            # Convert embedding list to PostgreSQL vector format
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            rows = await conn.fetch(
                """
                SELECT
                    cs.id, cs.content, cs.snippet_type, d.title, d.url, 
                    d.metadata as doc_metadata, cs.metadata as snippet_metadata, 
                    d.id as document_id,
                    1 - (cs.embedding <=> $2::halfvec) as similarity
                FROM code_snippets cs
                JOIN documents d ON cs.document_id = d.id
                WHERE cs.context_id = $1
                    AND 1 - (cs.embedding <=> $2::halfvec) > $3
                ORDER BY cs.embedding <=> $2::halfvec
                LIMIT $4
            """,
                uuid.UUID(context_id),
                embedding_str,
                min_similarity,
                limit,
            )

            return [
                {
                    "id": format_uuid(row["id"]),
                    "document_id": str(row["document_id"]),
                    "content": row["content"],
                    "snippet_type": row["snippet_type"],
                    "title": row["title"],
                    "url": row["url"],
                    "score": float(row["similarity"]),
                    "line_count": len(row["content"].split('\n')) if row["content"] else 0,
                    "metadata": parse_metadata(row["snippet_metadata"]),
                }
                for row in rows
            ]
    
    async def fulltext_search_code_snippets(
        self, context_id: str, query: str, limit: int = 10
    ) -> list[dict]:
        """Perform full-text search on code snippets."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    cs.id, cs.content, cs.snippet_type, d.title, d.url,
                    d.metadata as doc_metadata, cs.metadata as snippet_metadata,
                    d.id as document_id,
                    ts_rank(to_tsvector('english', cs.content), plainto_tsquery('english', $2)) as score
                FROM code_snippets cs
                JOIN documents d ON cs.document_id = d.id
                WHERE cs.context_id = $1
                    AND to_tsvector('english', cs.content) @@ plainto_tsquery('english', $2)
                ORDER BY score DESC
                LIMIT $3
            """,
                uuid.UUID(context_id),
                query,
                limit,
            )

            return [
                {
                    "id": format_uuid(row["id"]),
                    "document_id": str(row["document_id"]),
                    "content": row["content"],
                    "snippet_type": row["snippet_type"],
                    "title": row["title"],
                    "url": row["url"],
                    "score": float(row["score"]),
                    "line_count": len(row["content"].split('\n')) if row["content"] else 0,
                    "metadata": parse_metadata(row["snippet_metadata"]),
                }
                for row in rows
            ]


__all__ = ["SearchManager"]