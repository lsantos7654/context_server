"""Context CRUD operations."""

import json
import uuid
import logging
from datetime import datetime
from ..utils import convert_embedding_to_postgres, parse_metadata, format_uuid, parse_uuid

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages context-related database operations."""
    
    def __init__(self):
        self.pool = None
    
    async def create_context(self, name: str, description: str = "", embedding_model: str = "text-embedding-3-large") -> dict:
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
                "id": format_uuid(row["id"]),
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
                    "id": format_uuid(row["id"]),
                    "name": row["name"],
                    "description": row["description"],
                    "embedding_model": row["embedding_model"],
                    "created_at": row["created_at"],
                    "document_count": row["document_count"],
                    "size_mb": 0.0,
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
                "id": format_uuid(row["id"]),
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

    async def export_context_data(self, context_id: str) -> dict:
        """Export all context data for backup/migration."""
        async with self.pool.acquire() as conn:
            # Get context metadata
            context_row = await conn.fetchrow(
                """
                SELECT id, name, description, embedding_model, created_at, updated_at, document_count
                FROM contexts WHERE id = $1
                """,
                uuid.UUID(context_id)
            )
            
            if not context_row:
                raise ValueError(f"Context with ID {context_id} not found")
            
            context_data = {
                "id": str(context_row["id"]),
                "name": context_row["name"],
                "description": context_row["description"],
                "embedding_model": context_row["embedding_model"],
                "created_at": context_row["created_at"].isoformat(),
                "updated_at": context_row["updated_at"].isoformat(),
                "document_count": context_row["document_count"]
            }
            
            # Get all documents
            document_rows = await conn.fetch(
                """
                SELECT id, url, title, content, metadata, source_type, indexed_at, chunk_count
                FROM documents WHERE context_id = $1
                """,
                uuid.UUID(context_id)
            )
            
            documents = []
            for row in document_rows:
                documents.append({
                    "id": format_uuid(row["id"]),
                    "url": row["url"],
                    "title": row["title"],
                    "content": row["content"],
                    "metadata": parse_metadata(row["metadata"]),
                    "source_type": row["source_type"],
                    "indexed_at": row["indexed_at"].isoformat() if row["indexed_at"] else None,
                    "chunk_count": row["chunk_count"]
                })
            
            # Get all chunks
            chunk_rows = await conn.fetch(
                """
                SELECT c.id, c.document_id, c.content, c.summary, c.summary_model,
                       c.text_embedding, c.code_embedding, c.chunk_index, c.metadata,
                       c.tokens, c.start_line, c.end_line, c.char_start, c.char_end, c.created_at
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.context_id = $1
                """,
                uuid.UUID(context_id)
            )
            
            chunks = []
            for row in chunk_rows:
                # Convert embeddings from PostgreSQL format
                text_embedding = None
                code_embedding = None
                
                if row["text_embedding"]:
                    # Convert PostgreSQL vector to list
                    text_embedding = list(row["text_embedding"])
                    
                if row["code_embedding"]:
                    # Convert PostgreSQL vector to list
                    code_embedding = list(row["code_embedding"])
                
                chunks.append({
                    "id": format_uuid(row["id"]),
                    "document_id": str(row["document_id"]),
                    "content": row["content"],
                    "summary": row["summary"],
                    "summary_model": row["summary_model"],
                    "text_embedding": text_embedding,
                    "code_embedding": code_embedding,
                    "chunk_index": row["chunk_index"],
                    "metadata": parse_metadata(row["metadata"]),
                    "tokens": row["tokens"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "char_start": row["char_start"],
                    "char_end": row["char_end"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None
                })
            
            # Get all code snippets
            snippet_rows = await conn.fetch(
                """
                SELECT cs.id, cs.document_id, cs.content, cs.language, cs.embedding,
                       cs.metadata, cs.start_line, cs.end_line, cs.char_start, cs.char_end,
                       cs.snippet_type, cs.preview, cs.created_at
                FROM code_snippets cs
                JOIN documents d ON cs.document_id = d.id
                WHERE d.context_id = $1
                """,
                uuid.UUID(context_id)
            )
            
            code_snippets = []
            for row in snippet_rows:
                # Convert embedding from PostgreSQL format
                embedding = None
                if row["embedding"]:
                    embedding = list(row["embedding"])
                
                code_snippets.append({
                    "id": format_uuid(row["id"]),
                    "document_id": str(row["document_id"]),
                    "content": row["content"],
                    "language": row["language"],
                    "embedding": embedding,
                    "metadata": parse_metadata(row["metadata"]),
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "char_start": row["char_start"],
                    "char_end": row["char_end"],
                    "snippet_type": row["snippet_type"],
                    "preview": row["preview"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None
                })
            
            return {
                "schema_version": "1.0",
                "context": context_data,
                "documents": documents,
                "chunks": chunks,
                "code_snippets": code_snippets,
                "exported_at": datetime.utcnow(),
                "total_documents": len(documents),
                "total_chunks": len(chunks),
                "total_code_snippets": len(code_snippets)
            }

    async def import_context_data(self, export_data: dict, overwrite_existing: bool = False) -> dict:
        """Import context data from export."""
        context_data = export_data["context"]
        documents_data = export_data["documents"]
        chunks_data = export_data["chunks"]
        code_snippets_data = export_data["code_snippets"]
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Check if context with same name exists
                existing_context = await conn.fetchrow(
                    "SELECT id FROM contexts WHERE name = $1",
                    context_data["name"]
                )
                
                if existing_context and not overwrite_existing:
                    raise ValueError(f"Context '{context_data['name']}' already exists. Use overwrite_existing=True to replace it.")
                
                # Delete existing context if overwriting
                if existing_context and overwrite_existing:
                    await conn.execute(
                        "DELETE FROM contexts WHERE id = $1",
                        existing_context["id"]
                    )
                
                # Create new context
                context_id = await conn.fetchval(
                    """
                    INSERT INTO contexts (name, description, embedding_model)
                    VALUES ($1, $2, $3)
                    RETURNING id
                    """,
                    context_data["name"],
                    context_data["description"],
                    context_data.get("embedding_model", "text-embedding-3-large")
                )
                
                # Import documents
                document_id_mapping = {}  # Old ID -> New ID
                for doc_data in documents_data:
                    new_doc_id = await conn.fetchval(
                        """
                        INSERT INTO documents (context_id, url, title, content, metadata, source_type, chunk_count)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        RETURNING id
                        """,
                        context_id,
                        doc_data["url"],
                        doc_data["title"],
                        doc_data["content"],
                        json.dumps(doc_data["metadata"]),
                        doc_data["source_type"],
                        doc_data.get("chunk_count", 0)
                    )
                    document_id_mapping[doc_data["id"]] = str(new_doc_id)
                
                # Import chunks
                for chunk_data in chunks_data:
                    new_doc_id = document_id_mapping.get(chunk_data["document_id"])
                    if not new_doc_id:
                        continue
                    
                    # Convert embeddings back to PostgreSQL format
                    text_embedding_str = None
                    code_embedding_str = None
                    
                    if chunk_data.get("text_embedding"):
                        text_embedding_str = convert_embedding_to_postgres(chunk_data["text_embedding"])
                        
                    if chunk_data.get("code_embedding"):
                        code_embedding_str = convert_embedding_to_postgres(chunk_data["code_embedding"])
                    
                    await conn.execute(
                        """
                        INSERT INTO chunks (document_id, context_id, content, summary, summary_model,
                                          text_embedding, code_embedding, chunk_index, metadata, tokens,
                                          start_line, end_line, char_start, char_end)
                        VALUES ($1, $2, $3, $4, $5, $6::halfvec, $7::halfvec, $8, $9, $10, $11, $12, $13, $14)
                        """,
                        uuid.UUID(new_doc_id),
                        context_id,
                        chunk_data["content"],
                        chunk_data.get("summary"),
                        chunk_data.get("summary_model"),
                        text_embedding_str,
                        code_embedding_str,
                        chunk_data["chunk_index"],
                        json.dumps(chunk_data["metadata"]),
                        chunk_data.get("tokens"),
                        chunk_data.get("start_line"),
                        chunk_data.get("end_line"),
                        chunk_data.get("char_start"),
                        chunk_data.get("char_end")
                    )
                
                # Import code snippets
                for snippet_data in code_snippets_data:
                    new_doc_id = document_id_mapping.get(snippet_data["document_id"])
                    if not new_doc_id:
                        continue
                    
                    # Convert embedding back to PostgreSQL format
                    embedding_str = None
                    if snippet_data.get("embedding"):
                        embedding_str = convert_embedding_to_postgres(snippet_data["embedding"])
                    
                    await conn.execute(
                        """
                        INSERT INTO code_snippets (document_id, context_id, content, embedding,
                                                 metadata, start_line, end_line, char_start, char_end,
                                                 snippet_type, preview)
                        VALUES ($1, $2, $3, $4::halfvec, $5, $6, $7, $8, $9, $10, $11)
                        """,
                        uuid.UUID(new_doc_id),
                        context_id,
                        snippet_data["content"],
                        embedding_str,
                        json.dumps(snippet_data["metadata"]),
                        snippet_data.get("start_line"),
                        snippet_data.get("end_line"),
                        snippet_data.get("char_start"),
                        snippet_data.get("char_end"),
                        snippet_data.get("snippet_type", "code_block"),
                        snippet_data.get("preview", snippet_data.get("summary", ""))  # Support both old and new format
                    )
                
                # Update context document count
                await conn.execute(
                    """
                    UPDATE contexts
                    SET document_count = (SELECT COUNT(*) FROM documents WHERE context_id = $1),
                        updated_at = NOW()
                    WHERE id = $1
                    """,
                    context_id
                )
                
                return {
                    "success": True,
                    "context_id": str(context_id),
                    "context_name": context_data["name"],
                    "imported_documents": len(documents_data),
                    "imported_chunks": len(chunks_data),
                    "imported_code_snippets": len(code_snippets_data),
                    "message": f"Successfully imported context '{context_data['name']}'"
                }

    async def merge_contexts(self, source_context_ids: list[str], target_context_id: str, mode: str) -> dict:
        """Merge multiple contexts into a target context."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Get target context info
                target_context = await conn.fetchrow(
                    "SELECT name FROM contexts WHERE id = $1",
                    uuid.UUID(target_context_id)
                )
                
                if not target_context:
                    raise ValueError(f"Target context with ID {target_context_id} not found")
                
                # Collect all documents to merge based on mode
                documents_to_merge = []
                
                if mode == "union":
                    # Get all documents from all source contexts
                    for source_id in source_context_ids:
                        source_docs = await conn.fetch(
                            """
                            SELECT id, url, title, content, metadata, source_type, chunk_count
                            FROM documents WHERE context_id = $1
                            """,
                            uuid.UUID(source_id)
                        )
                        documents_to_merge.extend(source_docs)
                
                elif mode == "intersection":
                    # Only keep documents that exist in ALL source contexts (by URL)
                    if not source_context_ids:
                        documents_to_merge = []
                    else:
                        # Get documents from first context
                        first_context_docs = await conn.fetch(
                            """
                            SELECT id, url, title, content, metadata, source_type, chunk_count
                            FROM documents WHERE context_id = $1
                            """,
                            uuid.UUID(source_context_ids[0])
                        )
                        
                        # Filter to only URLs that exist in ALL contexts
                        for doc in first_context_docs:
                            url_exists_in_all = True
                            for source_id in source_context_ids[1:]:
                                url_check = await conn.fetchval(
                                    "SELECT EXISTS(SELECT 1 FROM documents WHERE context_id = $1 AND url = $2)",
                                    uuid.UUID(source_id), doc["url"]
                                )
                                if not url_check:
                                    url_exists_in_all = False
                                    break
                            
                            if url_exists_in_all:
                                documents_to_merge.append(doc)
                
                # Remove duplicates by URL (keep latest based on indexed_at or first encountered)
                unique_documents = {}
                for doc in documents_to_merge:
                    url = doc["url"]
                    if url not in unique_documents:
                        unique_documents[url] = doc
                
                documents_to_merge = list(unique_documents.values())
                
                # Copy documents to target context
                document_id_mapping = {}  # Old ID -> New ID
                for doc in documents_to_merge:
                    # Check if document already exists in target context
                    existing_doc = await conn.fetchval(
                        "SELECT id FROM documents WHERE context_id = $1 AND url = $2",
                        uuid.UUID(target_context_id), doc["url"]
                    )
                    
                    if existing_doc:
                        # Document already exists, use existing ID
                        document_id_mapping[str(doc["id"])] = str(existing_doc)
                        continue
                    
                    # Insert new document
                    new_doc_id = await conn.fetchval(
                        """
                        INSERT INTO documents (context_id, url, title, content, metadata, source_type, chunk_count)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        RETURNING id
                        """,
                        uuid.UUID(target_context_id),
                        doc["url"],
                        doc["title"],
                        doc["content"],
                        doc["metadata"],
                        doc["source_type"],
                        doc.get("chunk_count", 0)
                    )
                    document_id_mapping[str(doc["id"])] = str(new_doc_id)
                
                # Copy chunks for merged documents
                total_chunks = 0
                for old_doc_id, new_doc_id in document_id_mapping.items():
                    chunks = await conn.fetch(
                        """
                        SELECT content, summary, summary_model, text_embedding, code_embedding,
                               chunk_index, metadata, tokens, start_line, end_line, char_start, char_end
                        FROM chunks WHERE document_id = $1
                        """,
                        uuid.UUID(old_doc_id)
                    )
                    
                    for chunk in chunks:
                        # Check if chunk already exists (by document and chunk_index)
                        existing_chunk = await conn.fetchval(
                            "SELECT id FROM chunks WHERE document_id = $1 AND chunk_index = $2",
                            uuid.UUID(new_doc_id), chunk["chunk_index"]
                        )
                        
                        if existing_chunk:
                            continue  # Skip existing chunks
                        
                        await conn.execute(
                            """
                            INSERT INTO chunks (document_id, context_id, content, summary, summary_model,
                                              text_embedding, code_embedding, chunk_index, metadata, tokens,
                                              start_line, end_line, char_start, char_end)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                            """,
                            uuid.UUID(new_doc_id),
                            uuid.UUID(target_context_id),
                            chunk["content"],
                            chunk["summary"],
                            chunk["summary_model"],
                            chunk["text_embedding"],
                            chunk["code_embedding"],
                            chunk["chunk_index"],
                            chunk["metadata"],
                            chunk["tokens"],
                            chunk["start_line"],
                            chunk["end_line"],
                            chunk["char_start"],
                            chunk["char_end"]
                        )
                        total_chunks += 1
                
                # Copy code snippets for merged documents
                total_code_snippets = 0
                for old_doc_id, new_doc_id in document_id_mapping.items():
                    snippets = await conn.fetch(
                        """
                        SELECT content, language, embedding, metadata, start_line, end_line,
                               char_start, char_end, snippet_type, preview
                        FROM code_snippets WHERE document_id = $1
                        """,
                        uuid.UUID(old_doc_id)
                    )
                    
                    for snippet in snippets:
                        await conn.execute(
                            """
                            INSERT INTO code_snippets (document_id, context_id, content, language, embedding,
                                                     metadata, start_line, end_line, char_start, char_end,
                                                     snippet_type, preview)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                            """,
                            uuid.UUID(new_doc_id),
                            uuid.UUID(target_context_id),
                            snippet["content"],
                            snippet["language"],
                            snippet["embedding"],
                            snippet["metadata"],
                            snippet["start_line"],
                            snippet["end_line"],
                            snippet["char_start"],
                            snippet["char_end"],
                            snippet["snippet_type"],
                            snippet["preview"]
                        )
                        total_code_snippets += 1
                
                # Update target context document count
                await conn.execute(
                    """
                    UPDATE contexts
                    SET document_count = (SELECT COUNT(*) FROM documents WHERE context_id = $1),
                        updated_at = NOW()
                    WHERE id = $1
                    """,
                    uuid.UUID(target_context_id)
                )
                
                return {
                    "success": True,
                    "target_context_id": target_context_id,
                    "target_context_name": target_context["name"],
                    "merged_documents": len(documents_to_merge),
                    "merged_chunks": total_chunks,
                    "merged_code_snippets": total_code_snippets,
                    "source_contexts_processed": len(source_context_ids),
                    "message": f"Successfully merged {len(source_context_ids)} contexts into '{target_context['name']}' using {mode} mode"
                }


__all__ = ["ContextManager"]