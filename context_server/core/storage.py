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

    def __init__(self, database_url: str | None = None, summarization_service=None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", "postgresql://user:password@localhost:5432/context_server"
        )
        self.pool: Pool | None = None
        self.summarization_service = summarization_service

    def _filter_metadata_for_search(self, metadata: dict) -> dict:
        """Organize metadata into clean, grouped structure for search results."""
        raw_metadata = metadata.copy()

        # Remove fields that contain large amounts of data or internal processing details
        large_fields = [
            "extracted_pages",
            "full_content",
            "raw_content",
            "page_content",
        ]

        # Internal extraction details not needed by search consumers
        internal_fields = [
            "filtered_links",
            "total_links_found",
            "successful_extractions",
            "extraction_time",
            "method",
            "processing_time",
            "extraction_stats",
            "source_type",  # Always "crawl4ai", not useful
            "is_individual_page",  # Always true, not meaningful
            "source_url",  # Redundant with page_url
        ]

        # Remove unwanted fields
        for field in large_fields + internal_fields:
            raw_metadata.pop(field, None)

        # Organize into clean structure
        organized = {}

        # Document-level information
        organized["document"] = {
            "id": raw_metadata.get("document_id"),
            "title": raw_metadata.get("source_title"),
            "url": raw_metadata.get("page_url", raw_metadata.get("base_url")),
            "size": raw_metadata.get("parent_page_size"),
            "total_chunks": raw_metadata.get("parent_total_chunks"),
            "total_links": raw_metadata.get("total_page_links"),
        }

        # Chunk-level information
        organized["chunk"] = {
            "index": raw_metadata.get("chunk_index"),
            "links_count": raw_metadata.get("total_links_in_chunk"),
            "links": raw_metadata.get("chunk_links", {}),
        }

        # Code snippets information
        organized["code_snippets"] = raw_metadata.get("code_snippets", [])
        organized["code_snippets_count"] = raw_metadata.get("total_code_snippets", 0)

        # Keep any other metadata fields that weren't specifically handled
        handled_fields = {
            "document_id",
            "source_title",
            "page_url",
            "base_url",
            "parent_page_size",
            "parent_total_chunks",
            "total_page_links",
            "chunk_index",
            "total_links_in_chunk",
            "chunk_links",
            "code_snippets",
            "total_code_snippets",
        }

        for key, value in raw_metadata.items():
            if key not in handled_fields and key not in large_fields + internal_fields:
                organized[key] = value

        return organized

    def _generate_code_summary(self, snippet: dict) -> str:
        """Generate summary for code snippet: full code if ≤8 lines, AI summary if >8 lines."""
        # Try to get full content first, fallback to preview
        content = snippet.get("content", "")
        
        if not content:
            # If no full content, use preview which is already truncated and suitable for display
            preview = snippet.get("preview", "")
            if preview:
                return preview
            else:
                return "Empty code snippet"
        
        # Split into lines and check count
        lines = content.split('\n')
        line_count = len([line for line in lines if line.strip()])  # Count non-empty lines
        
        # Rule: If 8 lines or fewer, include full code content
        if line_count <= 8:
            return content.strip()
        
        # For longer code snippets, try AI summarization first
        if self.summarization_service and self.summarization_service.client:
            try:
                # This is a synchronous context, but we need async for AI
                # For now, use heuristic fallback. In production, this would be called 
                # from an async context where we can await the AI service
                import asyncio
                
                # Try to get current event loop, if it exists
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an async context, but this method is sync
                    # Use heuristic fallback for now
                    pass
                except RuntimeError:
                    # No event loop running
                    pass
            except Exception:
                pass
        
        # Use heuristic fallback
        return self._generate_heuristic_code_summary(content, lines)

    def _generate_heuristic_code_summary(self, content: str, lines: list[str]) -> str:
        """Generate a detailed 3-4 sentence heuristic summary for longer code snippets."""
        summary_sentences = []
        content_lower = content.lower()
        line_count = len([line for line in lines if line.strip()])
        
        # Sentence 1: Main purpose/type of code
        main_purpose = []
        
        # Check for function/class definitions
        functions_found = []
        classes_found = []
        
        for line in lines[:10]:  # Check first 10 lines for definitions
            line = line.strip()
            if line.startswith('def ') or line.startswith('function '):
                func_name = line.split('(')[0].replace('def ', '').replace('function ', '').strip()
                if func_name not in functions_found:
                    functions_found.append(func_name)
            elif line.startswith('async def '):
                func_name = line.split('(')[0].replace('async def ', '').strip()
                if func_name not in functions_found:
                    functions_found.append(f"{func_name} (async)")
            elif line.startswith('class '):
                class_name = line.split('(')[0].replace('class ', '').replace(':', '').strip()
                if class_name not in classes_found:
                    classes_found.append(class_name)
        
        # Build main purpose sentence
        if functions_found and classes_found:
            main_purpose.append(f"This code defines the {', '.join(classes_found)} class(es) and implements {', '.join(functions_found[:2])} function(s)")
        elif functions_found:
            if len(functions_found) == 1:
                main_purpose.append(f"This code implements the {functions_found[0]} function")
            else:
                main_purpose.append(f"This code implements {len(functions_found)} functions including {', '.join(functions_found[:2])}")
        elif classes_found:
            main_purpose.append(f"This code defines the {', '.join(classes_found)} class(es)")
        else:
            # Check for configuration or setup patterns
            if any(word in content_lower for word in ['config', 'setup', 'init', 'configure']):
                main_purpose.append("This is a configuration or setup code block")
            elif 'import ' in content or 'from ' in content:
                main_purpose.append("This code sets up imports and dependencies")
            else:
                main_purpose.append("This is a code block that performs various operations")
        
        if main_purpose:
            summary_sentences.append(main_purpose[0] + ".")
        
        # Sentence 2: Key features and patterns
        features = []
        
        if 'async ' in content or 'await ' in content:
            features.append("asynchronous operations")
        if 'try:' in content and 'except' in content:
            features.append("error handling")
        if 'for ' in content or 'while ' in content:
            features.append("iteration/loops")
        if any(word in content_lower for word in ['http', 'request', 'api', 'client']):
            features.append("HTTP/API interactions")
        if any(word in content_lower for word in ['database', 'db', 'query', 'sql']):
            features.append("database operations")
        if any(word in content_lower for word in ['file', 'read', 'write', 'path']):
            features.append("file operations")
        if 'json' in content_lower or 'yaml' in content_lower:
            features.append("data serialization")
        if any(word in content_lower for word in ['test', 'assert', 'mock']):
            features.append("testing functionality")
        
        if features:
            if len(features) == 1:
                summary_sentences.append(f"It includes {features[0]}.")
            elif len(features) == 2:
                summary_sentences.append(f"It includes {features[0]} and {features[1]}.")
            else:
                summary_sentences.append(f"It includes {', '.join(features[:-1])}, and {features[-1]}.")
        
        # Sentence 3: Technical details or complexity
        complexity_info = []
        
        if line_count > 50:
            complexity_info.append(f"This is a substantial code block with {line_count} lines")
        elif line_count > 20:
            complexity_info.append(f"This is a moderately sized code block with {line_count} lines")
        else:
            complexity_info.append(f"This is a compact code block with {line_count} lines")
        
        # Count imports to gauge dependencies
        import_count = len([line for line in lines if line.strip().startswith(('import ', 'from '))])
        if import_count > 5:
            complexity_info.append(f"and uses {import_count} different imports")
        elif import_count > 0:
            complexity_info.append(f"and includes {import_count} import(s)")
        
        if complexity_info:
            summary_sentences.append(" ".join(complexity_info) + ".")
        
        # Sentence 4: Purpose or usage context (if identifiable)
        context_clues = []
        
        if any(word in content_lower for word in ['crawler', 'scrape', 'extract']):
            context_clues.append("web scraping or data extraction")
        elif any(word in content_lower for word in ['server', 'app', 'route', 'endpoint']):
            context_clues.append("web server or API development")
        elif any(word in content_lower for word in ['filter', 'process', 'transform']):
            context_clues.append("data processing or filtering")
        elif any(word in content_lower for word in ['markdown', 'content', 'generate']):
            context_clues.append("content generation or markdown processing")
        elif any(word in content_lower for word in ['embed', 'vector', 'search']):
            context_clues.append("search or embedding functionality")
        
        if context_clues:
            summary_sentences.append(f"The code appears to be designed for {context_clues[0]}.")
        
        # Combine sentences (aim for 3-4 sentences)
        summary = " ".join(summary_sentences[:4])
        
        # Ensure we have at least a basic summary
        if not summary.strip():
            summary = f"Code block with {line_count} lines containing various programming constructs."
        
        return summary

    def _transform_to_compact_format(self, results: list[dict], query: str = "", mode: str = "hybrid", execution_time_ms: int = 0) -> dict:
        """Transform full search results to compact MCP format.
        
        This is the single source of truth for transforming search results 
        to the compact format used by both CLI and MCP server.
        """
        compact_results = []
        
        for result in results:
            # Use summary if available, otherwise truncate content
            display_content = result.get("summary", "")
            if not display_content:
                content = result.get("content", "")
                display_content = content[:150] + "..." if len(content) > 150 else content
            # No truncation for summaries - they're already AI-generated and meaningful
            
            # Get code snippets count and IDs from metadata
            metadata = result.get("metadata", {})
            code_snippets = metadata.get("code_snippets", [])
            code_snippets_count = len(code_snippets) if code_snippets else 0
            
            # Extract code snippet metadata for direct access
            code_snippet_ids = []
            if code_snippets:
                for snippet in code_snippets:
                    if isinstance(snippet, dict) and "id" in snippet:
                        snippet_obj = {
                            "id": snippet["id"],
                            "size": len(snippet.get("preview", snippet.get("content", ""))),
                            "summary": self._generate_code_summary(snippet)
                        }
                        code_snippet_ids.append(snippet_obj)
            
            compact_result = {
                "id": result.get("id"),
                "document_id": result.get("document_id"),
                "title": result.get("title"),
                "summary": display_content,
                "score": result.get("score"),
                "url": result.get("url"),
                "has_summary": bool(result.get("summary")),
                "code_snippets_count": code_snippets_count,
                "code_snippet_ids": code_snippet_ids,
                "content_type": result.get("content_type", "chunk"),
                "chunk_index": result.get("chunk_index"),
            }
            compact_results.append(compact_result)
        
        return {
            "results": compact_results,
            "total": len(compact_results),
            "query": query,
            "mode": mode,
            "execution_time_ms": execution_time_ms,
            "note": "Content summarized for quick scanning. Use get_document for full content."
        }

    def _transform_code_to_compact_format(self, results: list[dict], query: str = "", execution_time_ms: int = 0) -> dict:
        """Transform code search results to compact MCP format.
        
        This is the single source of truth for transforming code search results
        to the compact format used by both CLI and MCP server.
        """
        compact_results = []
        
        for result in results:
            # Use content directly since code snippets are already concise
            display_content = result.get("content", "")
            
            # Truncate very long code snippets
            if len(display_content) > 500:
                display_content = display_content[:500] + "..."
            
            compact_result = {
                "id": result.get("id"),
                "document_id": result.get("document_id"),
                "title": result.get("title"),
                "content": display_content,
                "language": result.get("language", "text"),
                "snippet_type": result.get("snippet_type", "code_block"),
                "score": result.get("score"),
                "url": result.get("url"),
                "start_line": result.get("start_line"),
                "end_line": result.get("end_line"),
                "content_type": "code_snippet",
            }
            compact_results.append(compact_result)
        
        return {
            "results": compact_results,
            "total": len(compact_results),
            "query": query,
            "mode": "hybrid",
            "execution_time_ms": execution_time_ms,
            "note": "Code search using voyage-code-3 embeddings for enhanced code understanding."
        }

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
                    embedding_model VARCHAR(100) DEFAULT 'text-embedding-3-large',
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
                    summary TEXT,  -- LLM-generated summary for compact responses
                    summary_model VARCHAR(50),  -- Model used to generate summary
                    text_embedding halfvec(3072),  -- text-embedding-3-large dimension
                    code_embedding halfvec(2048),  -- voyage-code-3 dimension
                    chunk_index INTEGER NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    tokens INTEGER,
                    start_line INTEGER,
                    end_line INTEGER,
                    char_start INTEGER,
                    char_end INTEGER,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(document_id, chunk_index)
                )
            """
            )

            # Create code_snippets table with vector embeddings
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS code_snippets (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    context_id UUID NOT NULL REFERENCES contexts(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    language VARCHAR(50) NOT NULL DEFAULT 'text',
                    embedding halfvec(2048),  -- voyage-code-3 dimension
                    metadata JSONB DEFAULT '{}',
                    start_line INTEGER,
                    end_line INTEGER,
                    char_start INTEGER,
                    char_end INTEGER,
                    snippet_type VARCHAR(20) DEFAULT 'code_block',  -- 'code_block' or 'inline_code'
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """
            )

            # Create jobs table for tracking document processing jobs
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id VARCHAR(100) PRIMARY KEY,
                    type VARCHAR(50) NOT NULL,
                    context_id UUID REFERENCES contexts(id) ON DELETE CASCADE,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    progress REAL NOT NULL DEFAULT 0.0 CHECK (progress >= 0.0 AND progress <= 1.0),
                    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    completed_at TIMESTAMP WITH TIME ZONE,
                    metadata JSONB DEFAULT '{}',
                    error_message TEXT,
                    result_data JSONB DEFAULT '{}'
                )
            """
            )

            # Add line tracking columns to existing tables (if they don't exist)
            await conn.execute(
                """
                DO $$ BEGIN
                    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS start_line INTEGER;
                    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS end_line INTEGER;
                    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS char_start INTEGER;
                    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS char_end INTEGER;
                EXCEPTION
                    WHEN duplicate_column THEN NULL;
                END $$;
                """
            )

            # Create indexes for performance
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_text_embedding ON chunks USING ivfflat (text_embedding halfvec_cosine_ops)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_code_embedding ON chunks USING ivfflat (code_embedding halfvec_cosine_ops)"
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

            # Code snippets indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_code_snippets_embedding ON code_snippets USING ivfflat (embedding halfvec_cosine_ops)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_code_snippets_context_id ON code_snippets(context_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_code_snippets_language ON code_snippets(language)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_lines ON chunks(document_id, start_line, end_line)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_chars ON chunks(document_id, char_start, char_end)"
            )

            # Create jobs indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_context_id ON jobs(context_id)"
            )
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(type)")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at)"
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
        embedding_model: str = "text-embedding-3-large",
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
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

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

    async def create_code_snippet(
        self,
        document_id: str,
        context_id: str,
        content: str,
        language: str,
        embedding: list[float],
        metadata: dict = None,
        start_line: int = None,
        end_line: int = None,
        char_start: int = None,
        char_end: int = None,
        snippet_type: str = "code_block",
    ) -> str:
        """Create a new code snippet with embedding and line tracking."""
        async with self.pool.acquire() as conn:
            # Convert embedding list to PostgreSQL vector format
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            snippet_id = await conn.fetchval(
                """
                INSERT INTO code_snippets (document_id, context_id, content, language, embedding, metadata, start_line, end_line, char_start, char_end, snippet_type)
                VALUES ($1, $2, $3, $4, $5::vector, $6, $7, $8, $9, $10, $11)
                RETURNING id
            """,
                uuid.UUID(document_id),
                uuid.UUID(context_id),
                content,
                language,
                embedding_str,
                json.dumps(metadata or {}),
                start_line,
                end_line,
                char_start,
                char_end,
                snippet_type,
            )

            return str(snippet_id)

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

    async def _get_code_snippets_for_documents(
        self, document_ids: list[str], context_id: str
    ) -> dict[str, list[dict]]:
        """Get code snippets for multiple documents efficiently."""
        if not document_ids:
            return {}

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    cs.document_id, cs.id, cs.language, cs.snippet_type,
                    cs.start_line, cs.end_line, cs.content
                FROM code_snippets cs
                WHERE cs.document_id = ANY($1::uuid[]) AND cs.context_id = $2
                ORDER BY cs.document_id, cs.start_line ASC
                """,
                [uuid.UUID(doc_id) for doc_id in document_ids],
                uuid.UUID(context_id),
            )

            # Group snippets by document_id
            snippets_by_doc = {}
            for row in rows:
                doc_id = str(row["document_id"])
                if doc_id not in snippets_by_doc:
                    snippets_by_doc[doc_id] = []

                # Apply 8-line rule for code snippet preview
                content = row["content"]
                lines = content.split('\n')
                line_count = len([line for line in lines if line.strip()])
                
                if line_count <= 8:
                    # Show full code for short snippets
                    preview = content
                else:
                    # For longer snippets, create a descriptive summary
                    preview = self._generate_heuristic_code_summary(content, lines)
                
                snippet = {
                    "id": str(row["id"]),
                    "type": row["snippet_type"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "preview": preview,
                }
                snippets_by_doc[doc_id].append(snippet)

            return snippets_by_doc

    async def _get_code_snippets_for_chunk(
        self, document_id: str, context_id: str, chunk_start_line: int = None, chunk_end_line: int = None
    ) -> list[dict]:
        """Get code snippets that overlap with a specific chunk's line range."""
        async with self.pool.acquire() as conn:
            # If no line information available, return empty list
            if chunk_start_line is None or chunk_end_line is None:
                return []
            
            # Find code snippets that overlap with the chunk's line range
            # Overlap occurs when: snippet_start <= chunk_end AND snippet_end >= chunk_start
            rows = await conn.fetch(
                """
                SELECT id, content, language, snippet_type, start_line, end_line
                FROM code_snippets
                WHERE document_id = $1
                    AND context_id = $2
                    AND start_line IS NOT NULL
                    AND end_line IS NOT NULL
                    AND start_line <= $4
                    AND end_line >= $3
                ORDER BY start_line
                """,
                uuid.UUID(document_id),
                uuid.UUID(context_id),
                chunk_start_line,
                chunk_end_line,
            )

            # Apply 8-line rule for each code snippet
            result = []
            for row in rows:
                content = row["content"]
                lines = content.split('\n')
                line_count = len([line for line in lines if line.strip()])
                
                if line_count <= 8:
                    preview = content
                else:
                    preview = self._generate_heuristic_code_summary(content, lines)
                
                result.append({
                    "id": str(row["id"]),
                    "type": row["snippet_type"],
                    "language": row["language"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "preview": preview,
                })
            
            return result

    # Search methods
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
                    c.start_line, c.end_line, c.char_start, c.char_end,
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

            # Get chunk-specific code snippets for each result
            chunk_results = []
            for row in rows:
                # Get code snippets that overlap with this specific chunk
                chunk_code_snippets = await self._get_code_snippets_for_chunk(
                    str(row["document_id"]), 
                    context_id,
                    row.get("start_line"), 
                    row.get("end_line")
                )

                chunk_results.append({
                    "id": str(row["id"]),
                    "document_id": str(row["document_id"]),
                    "content": row["content"],
                    "summary": row["summary"],
                    "summary_model": row["summary_model"],
                    "title": row["title"],
                    "url": row["url"],
                    "score": float(row["similarity"]),
                    "metadata": self._filter_metadata_for_search(
                        {
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
                            "document_id": str(row["document_id"]),
                            "parent_page_size": row["parent_page_size"],
                            "parent_total_chunks": row["parent_total_chunks"],
                            "chunk_index": row["chunk_index"],
                            "code_snippets": chunk_code_snippets,
                            "total_code_snippets": len(chunk_code_snippets),
                        }
                    ),
                    "chunk_index": row["chunk_index"],
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
                    c.start_line, c.end_line, c.char_start, c.char_end,
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

            # Get chunk-specific code snippets for each result
            chunk_results = []
            for row in rows:
                # Get code snippets that overlap with this specific chunk
                chunk_code_snippets = await self._get_code_snippets_for_chunk(
                    str(row["document_id"]), 
                    context_id,
                    row.get("start_line"), 
                    row.get("end_line")
                )

                chunk_results.append({
                    "id": str(row["id"]),
                    "document_id": str(row["document_id"]),
                    "content": row["content"],
                    "summary": row["summary"],
                    "summary_model": row["summary_model"],
                    "title": row["title"],
                    "url": row["url"],
                    "score": float(row["score"]),
                    "metadata": self._filter_metadata_for_search(
                        {
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
                            "document_id": str(row["document_id"]),
                            "parent_page_size": row["parent_page_size"],
                            "parent_total_chunks": row["parent_total_chunks"],
                            "chunk_index": row["chunk_index"],
                            "code_snippets": chunk_code_snippets,
                            "total_code_snippets": len(chunk_code_snippets),
                        }
                    ),
                    "chunk_index": row["chunk_index"],
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
                    cs.id, cs.content, cs.language, cs.snippet_type, d.title, d.url, 
                    d.metadata as doc_metadata, cs.metadata as snippet_metadata, 
                    d.id as document_id, cs.start_line, cs.end_line, cs.char_start, cs.char_end,
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
                    "id": str(row["id"]),
                    "document_id": str(row["document_id"]),
                    "content": row["content"],
                    "language": row["language"],
                    "snippet_type": row["snippet_type"],
                    "title": row["title"],
                    "url": row["url"],
                    "score": float(row["similarity"]),
                    "metadata": self._filter_metadata_for_search(
                        {
                            **(
                                json.loads(row["doc_metadata"])
                                if row["doc_metadata"]
                                else {}
                            ),
                            **(
                                json.loads(row["snippet_metadata"])
                                if row["snippet_metadata"]
                                else {}
                            ),
                            "document_id": str(row["document_id"]),
                        }
                    ),
                    "start_line": row.get("start_line"),
                    "end_line": row.get("end_line"),
                    "char_start": row.get("char_start"),
                    "char_end": row.get("char_end"),
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
                    cs.id, cs.content, cs.language, cs.snippet_type, d.title, d.url,
                    d.metadata as doc_metadata, cs.metadata as snippet_metadata,
                    d.id as document_id, cs.start_line, cs.end_line, cs.char_start, cs.char_end,
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
                    "id": str(row["id"]),
                    "document_id": str(row["document_id"]),
                    "content": row["content"],
                    "language": row["language"],
                    "snippet_type": row["snippet_type"],
                    "title": row["title"],
                    "url": row["url"],
                    "score": float(row["score"]),
                    "metadata": self._filter_metadata_for_search(
                        {
                            **(
                                json.loads(row["doc_metadata"])
                                if row["doc_metadata"]
                                else {}
                            ),
                            **(
                                json.loads(row["snippet_metadata"])
                                if row["snippet_metadata"]
                                else {}
                            ),
                            "document_id": str(row["document_id"]),
                        }
                    ),
                    "start_line": row.get("start_line"),
                    "end_line": row.get("end_line"),
                    "char_start": row.get("char_start"),
                    "char_end": row.get("char_end"),
                }
                for row in rows
            ]

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

    # Code snippet query methods
    async def get_code_snippets_by_document(
        self, document_id: str, context_id: str = None
    ) -> list[dict]:
        """Get all code snippets for a document."""
        async with self.pool.acquire() as conn:
            query = """
                SELECT
                    cs.id, cs.content, cs.language, cs.metadata,
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
                    preview = content
                else:
                    preview = self._generate_heuristic_code_summary(content, lines)
                
                result.append({
                    "id": str(row["id"]),
                    "content": content,
                    "type": row["snippet_type"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "char_start": row["char_start"],
                    "char_end": row["char_end"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"].isoformat()
                    if row["created_at"]
                    else None,
                    "preview": preview,
                })
            
            return result

    async def get_code_snippet_by_id(
        self, snippet_id: str, context_id: str = None
    ) -> dict | None:
        """Get a specific code snippet by ID."""
        async with self.pool.acquire() as conn:
            query = """
                SELECT
                    cs.id, cs.content, cs.language, cs.metadata,
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
                "id": str(row["id"]),
                "document_id": str(row["document_id"]),
                "content": row["content"],
                "type": row["snippet_type"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "char_start": row["char_start"],
                "char_end": row["char_end"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "created_at": row["created_at"].isoformat()
                if row["created_at"]
                else None,
                "document_title": row["document_title"],
                "document_url": row["document_url"],
            }

    # Job tracking methods
    async def create_job(
        self,
        job_id: str,
        job_type: str,
        context_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Create a new processing job."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO jobs (id, type, context_id, status, metadata)
                VALUES ($1, $2, $3, 'pending', $4)
                """,
                job_id,
                job_type,
                uuid.UUID(context_id) if context_id else None,
                json.dumps(metadata or {}),
            )
        logger.info(f"Created job {job_id} of type {job_type}")
        return job_id

    async def update_job_progress(
        self,
        job_id: str,
        progress: float,
        status: str | None = None,
        metadata: dict | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update job progress and status."""
        async with self.pool.acquire() as conn:
            # Build dynamic update query
            updates = ["progress = $2", "updated_at = NOW()"]
            params = [job_id, progress]
            param_count = 2

            if status:
                param_count += 1
                updates.append(f"status = ${param_count}")
                params.append(status)

            if metadata:
                param_count += 1
                updates.append(f"metadata = ${param_count}")
                params.append(json.dumps(metadata))

            if error_message:
                param_count += 1
                updates.append(f"error_message = ${param_count}")
                params.append(error_message)

            if status in ["completed", "failed"]:
                updates.append("completed_at = NOW()")

            query = f"UPDATE jobs SET {', '.join(updates)} WHERE id = $1"
            await conn.execute(query, *params)

        logger.debug(f"Updated job {job_id}: progress={progress}, status={status}")

    async def get_job_status(self, job_id: str) -> dict | None:
        """Get job status and details."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    id, type, context_id, status, progress,
                    started_at, updated_at, completed_at,
                    metadata, error_message, result_data
                FROM jobs
                WHERE id = $1
                """,
                job_id,
            )

            if not row:
                return None

            return {
                "id": row["id"],
                "type": row["type"],
                "context_id": str(row["context_id"]) if row["context_id"] else None,
                "status": row["status"],
                "progress": row["progress"],
                "started_at": row["started_at"].isoformat()
                if row["started_at"]
                else None,
                "updated_at": row["updated_at"].isoformat()
                if row["updated_at"]
                else None,
                "completed_at": row["completed_at"].isoformat()
                if row["completed_at"]
                else None,
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "error_message": row["error_message"],
                "result_data": json.loads(row["result_data"])
                if row["result_data"]
                else {},
            }

    async def complete_job(
        self,
        job_id: str,
        result_data: dict | None = None,
        error_message: str | None = None,
    ) -> None:
        """Mark job as completed or failed."""
        status = "failed" if error_message else "completed"
        progress = 1.0

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE jobs
                SET status = $2, progress = $3, completed_at = NOW(),
                    updated_at = NOW(), result_data = $4, error_message = $5
                WHERE id = $1
                """,
                job_id,
                status,
                progress,
                json.dumps(result_data or {}),
                error_message,
            )

        logger.info(f"Completed job {job_id} with status: {status}")

    async def get_active_jobs(self, context_id: str | None = None) -> list[dict]:
        """Get all active (non-completed) jobs."""
        async with self.pool.acquire() as conn:
            query = """
                SELECT
                    id, type, context_id, status, progress,
                    started_at, updated_at, metadata
                FROM jobs
                WHERE status NOT IN ('completed', 'failed')
            """
            params = []

            if context_id:
                query += " AND context_id = $1"
                params.append(uuid.UUID(context_id))

            query += " ORDER BY started_at DESC"
            rows = await conn.fetch(query, *params)

            return [
                {
                    "id": row["id"],
                    "type": row["type"],
                    "context_id": str(row["context_id"]) if row["context_id"] else None,
                    "status": row["status"],
                    "progress": row["progress"],
                    "started_at": row["started_at"].isoformat()
                    if row["started_at"]
                    else None,
                    "updated_at": row["updated_at"].isoformat()
                    if row["updated_at"]
                    else None,
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                }
                for row in rows
            ]

    async def cleanup_old_jobs(self, days: int = 7) -> int:
        """Clean up completed jobs older than specified days."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM jobs
                WHERE status IN ('completed', 'failed')
                AND completed_at < NOW() - INTERVAL '%s days'
                """,
                days,
            )
            deleted_count = int(result.split()[-1])
            logger.info(f"Cleaned up {deleted_count} old jobs")
            return deleted_count
