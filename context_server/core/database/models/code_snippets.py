"""Code snippet CRUD operations."""

import json
import uuid
from ..utils import convert_embedding_to_postgres, parse_metadata, format_uuid, parse_uuid


class CodeSnippetManager:
    """Manages code snippet-related database operations."""
    
    def __init__(self):
        self.pool = None
    
    async def create_code_snippet(
        self,
        document_id: str,
        context_id: str,
        content: str,
        embedding: list[float],
        metadata: dict = None,
        start_line: int = None,
        end_line: int = None,
        char_start: int = None,
        char_end: int = None,
        snippet_type: str = "code_block",
        summary: str = None,
        summary_model: str = None,
    ) -> str:
        """Create a new code snippet with embedding and line tracking."""
        async with self.pool.acquire() as conn:
            # Convert embedding list to PostgreSQL vector format
            embedding_str = convert_embedding_to_postgres(embedding)

            snippet_id = await conn.fetchval(
                """
                INSERT INTO code_snippets (document_id, context_id, content, embedding, metadata, start_line, end_line, char_start, char_end, snippet_type, summary, summary_model)
                VALUES ($1, $2, $3, $4::halfvec, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING id
            """,
                uuid.UUID(document_id),
                uuid.UUID(context_id),
                content,
                embedding_str,
                json.dumps(metadata or {}),
                start_line,
                end_line,
                char_start,
                char_end,
                snippet_type,
                summary,
                summary_model,
            )

            return str(snippet_id)
    
    async def get_code_snippets_by_document(
        self, document_id: str, context_id: str = None
    ) -> list[dict]:
        """Get all code snippets for a document."""
        async with self.pool.acquire() as conn:
            query = """
                SELECT
                    cs.id, cs.content, cs.metadata,
                    cs.start_line, cs.end_line, cs.char_start, cs.char_end,
                    cs.snippet_type, cs.created_at
                FROM code_snippets cs
                WHERE cs.document_id = $1
            """
            params = [uuid.UUID(document_id)]

            if context_id:
                query += " AND cs.context_id = $2"
                params.append(uuid.UUID(context_id))

            query += " ORDER BY cs.start_line ASC"

            rows = await conn.fetch(query, *params)

            # Apply 8-line rule for each code snippet preview
            result = []
            for row in rows:
                content = row["content"]
                lines = content.split('\n')
                line_count = len([line for line in lines if line.strip()])
                
                if line_count <= 8:
                    preview = content.strip()
                else:
                    preview_lines = []
                    for line in lines[:8]:
                        preview_lines.append(line)
                    preview = '\n'.join(preview_lines).strip()
                
                result.append({
                    "id": format_uuid(row["id"]),
                    "content": row["content"],
                    "preview": preview,
                    "type": row["snippet_type"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "char_start": row["char_start"],
                    "char_end": row["char_end"],
                    "metadata": parse_metadata(row["metadata"]),
                    "created_at": row["created_at"],
                    "line_count": line_count,
                })

            return result
    
    async def get_code_snippet_by_id(
        self, snippet_id: str, context_id: str = None
    ) -> dict | None:
        """Get a specific code snippet by ID."""
        async with self.pool.acquire() as conn:
            query = """
                SELECT
                    cs.id, cs.content, cs.metadata,
                    cs.start_line, cs.end_line, cs.char_start, cs.char_end,
                    cs.snippet_type, cs.created_at, cs.document_id,
                    d.title as document_title, d.url as document_url
                FROM code_snippets cs
                JOIN documents d ON cs.document_id = d.id
                WHERE cs.id = $1
            """
            params = [uuid.UUID(snippet_id)]

            if context_id:
                query += " AND cs.context_id = $2"
                params.append(uuid.UUID(context_id))

            row = await conn.fetchrow(query, *params)

            if not row:
                return None

            return {
                "id": format_uuid(row["id"]),
                "document_id": format_uuid(row["document_id"]),
                "content": row["content"],
                "type": row["snippet_type"],
                "start_line": row["start_line"],
                "end_line": row["end_line"], 
                "char_start": row["char_start"],
                "char_end": row["char_end"],
                "metadata": parse_metadata(row["metadata"]),
                "created_at": row["created_at"],
                "document_title": row["document_title"],
                "document_url": row["document_url"],
            }


__all__ = ["CodeSnippetManager"]