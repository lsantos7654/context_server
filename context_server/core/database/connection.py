"""Database connection management and health checks."""

import asyncio
import logging
import os

import asyncpg
from asyncpg import Pool

from context_server.core.database.schema import SchemaManager
from context_server.core.database.models import ContextManager, DocumentManager, ChunkManager, CodeSnippetManager, JobManager
from context_server.core.database.search import SearchManager
from context_server.core.database.operations import OperationsManager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL connection with pgvector for vector storage.
    
    This is the main interface that composes all database operations while
    maintaining backward compatibility with the original storage.py API.
    """

    def __init__(self, database_url: str | None = None, summarization_service=None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", "postgresql://user:password@localhost:5432/context_server"
        )
        self.pool: Pool | None = None
        self.summarization_service = summarization_service

        # Initialize component managers
        self.schema = SchemaManager()
        self.contexts = ContextManager()
        self.documents = DocumentManager() 
        self.chunks = ChunkManager()
        self.code_snippets = CodeSnippetManager()
        self.jobs = JobManager()
        self.search = SearchManager(summarization_service)
        self.operations = OperationsManager(summarization_service)

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
                await self.schema.create_base_schema(self.pool)
                
                # Inject pool into all managers
                self._inject_pool()
                
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

    def _inject_pool(self):
        """Inject database pool into all component managers."""
        managers = [
            self.contexts, self.documents, self.chunks, 
            self.code_snippets, self.jobs, self.search, self.operations
        ]
        for manager in managers:
            manager.pool = self.pool

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

    # ===================
    # Context Operations - Delegate to ContextManager
    # ===================
    
    async def create_context(self, name: str, description: str = "", embedding_model: str = "text-embedding-3-large") -> dict:
        """Create a new context."""
        return await self.contexts.create_context(name, description, embedding_model)
    
    async def get_contexts(self) -> list[dict]:
        """Get all contexts."""
        return await self.contexts.get_contexts()
    
    async def get_context_by_name(self, name: str) -> dict | None:
        """Get context by name."""
        return await self.contexts.get_context_by_name(name)
    
    async def delete_context(self, context_id: str) -> bool:
        """Delete a context and all its data."""
        return await self.contexts.delete_context(context_id)

    # ===================
    # Document Operations - Delegate to DocumentManager
    # ===================
    
    async def create_document(self, context_id: str, url: str, title: str, content: str, metadata: dict, source_type: str, document_type: str = "original") -> str:
        """Create a new document."""
        return await self.documents.create_document(context_id, url, title, content, metadata, source_type, document_type)
    
    async def get_documents(self, context_id: str, offset: int = 0, limit: int = 50) -> dict:
        """Get documents in a context."""
        return await self.documents.get_documents(context_id, offset, limit)
    
    async def delete_documents(self, context_id: str, document_ids: list[str]) -> int:
        """Delete documents from a context."""
        return await self.documents.delete_documents(context_id, document_ids)
    
    async def get_document_by_id(self, context_id: str, document_id: str, document_type: str = "original") -> dict | None:
        """Get document content by ID."""
        return await self.documents.get_document_by_id(context_id, document_id, document_type)
    
    async def get_document_content_by_id(self, document_id: str) -> dict | None:
        """Get document content by ID only (for expansion service)."""
        return await self.documents.get_document_content_by_id(document_id)

    # ===================
    # Chunk Operations - Delegate to ChunkManager
    # ===================
    
    async def create_chunk(self, document_id: str, context_id: str, content: str, embedding: list[float], 
                          chunk_index: int, metadata: dict = None, tokens: int = None, summary: str = None,
                          summary_model: str = None, start_line: int = None, end_line: int = None,
                          char_start: int = None, char_end: int = None, is_code: bool = False) -> str:
        """Create a new chunk with embedding and line tracking."""
        return await self.chunks.create_chunk(
            document_id, context_id, content, embedding, chunk_index, metadata,
            tokens, summary, summary_model, start_line, end_line, char_start, char_end, is_code
        )
    
    async def update_document_chunk_count(self, document_id: str):
        """Update document chunk count."""
        return await self.chunks.update_document_chunk_count(document_id)
    
    async def get_chunk_by_id(self, chunk_id: str, context_id: str = None) -> dict | None:
        """Get a specific chunk by ID with full content and metadata."""
        return await self.chunks.get_chunk_by_id(chunk_id, context_id)

    # ===================
    # Code Snippet Operations - Delegate to CodeSnippetManager
    # ===================
    
    async def create_code_snippet(self, document_id: str, context_id: str, content: str, embedding: list[float],
                                 metadata: dict = None, start_line: int = None, end_line: int = None,
                                 char_start: int = None, char_end: int = None, snippet_type: str = "code_block",
                                 summary: str = None, summary_model: str = None) -> str:
        """Create a new code snippet with embedding and line tracking."""
        return await self.code_snippets.create_code_snippet(
            document_id, context_id, content, embedding, metadata, start_line, end_line,
            char_start, char_end, snippet_type, summary, summary_model
        )
    
    async def get_code_snippets_by_document(self, document_id: str, context_id: str = None) -> list[dict]:
        """Get all code snippets for a document."""
        return await self.code_snippets.get_code_snippets_by_document(document_id, context_id)
    
    async def get_code_snippet_by_id(self, snippet_id: str, context_id: str = None) -> dict | None:
        """Get a specific code snippet by ID."""
        return await self.code_snippets.get_code_snippet_by_id(snippet_id, context_id)

    # ===================
    # Job Operations - Delegate to JobManager
    # ===================
    
    async def create_job(self, job_id: str, job_type: str, context_id: str | None = None, metadata: dict | None = None) -> str:
        """Create a new processing job."""
        return await self.jobs.create_job(job_id, job_type, context_id, metadata)
    
    async def update_job_progress(self, job_id: str, progress: float, status: str | None = None,
                                 metadata: dict | None = None, error_message: str | None = None) -> None:
        """Update job progress and status."""
        return await self.jobs.update_job_progress(job_id, progress, status, metadata, error_message)
    
    async def get_job_status(self, job_id: str) -> dict | None:
        """Get job status and details."""
        return await self.jobs.get_job_status(job_id)
    
    async def complete_job(self, job_id: str, result_data: dict | None = None, error_message: str | None = None) -> None:
        """Mark job as completed or failed."""
        return await self.jobs.complete_job(job_id, result_data, error_message)
    
    async def get_active_jobs(self, context_id: str | None = None) -> list[dict]:
        """Get all active (non-completed) jobs."""
        return await self.jobs.get_active_jobs(context_id)
    
    async def cleanup_old_jobs(self, days: int = 7) -> int:
        """Clean up completed jobs older than specified days."""
        return await self.jobs.cleanup_old_jobs(days)

    # ===================
    # Search Operations - Delegate to SearchManager 
    # ===================
    
    async def vector_search(self, context_id: str, query_embedding: list[float], limit: int = 10,
                           min_similarity: float = 0.7, embedding_type: str = "text") -> list[dict]:
        """Perform vector similarity search on text or code embeddings."""
        return await self.search.vector_search(context_id, query_embedding, limit, min_similarity, embedding_type)
    
    async def fulltext_search(self, context_id: str, query: str, limit: int = 10) -> list[dict]:
        """Perform full-text search."""
        return await self.search.fulltext_search(context_id, query, limit)
    
    async def vector_search_code_snippets(self, context_id: str, query_embedding: list[float],
                                         limit: int = 10, min_similarity: float = 0.7) -> list[dict]:
        """Perform vector similarity search on code snippets."""
        return await self.search.vector_search_code_snippets(context_id, query_embedding, limit, min_similarity)
    
    async def fulltext_search_code_snippets(self, context_id: str, query: str, limit: int = 10) -> list[dict]:
        """Perform full-text search on code snippets."""
        return await self.search.fulltext_search_code_snippets(context_id, query, limit)

    # ===================
    # Operations - Delegate to OperationsManager
    # ===================
    
    def _filter_metadata_for_search(self, metadata: dict) -> dict:
        """Organize metadata into clean, grouped structure for search results."""
        return self.operations.filter_metadata_for_search(metadata)
    
    async def _transform_to_compact_format(self, results: list[dict], query: str = "", mode: str = "hybrid", execution_time_ms: int = 0) -> dict:
        """Transform full search results to compact MCP format."""
        return await self.operations.transform_to_compact_format(results, query, mode, execution_time_ms)
    
    def _transform_code_to_compact_format(self, results: list[dict], query: str = "", execution_time_ms: int = 0) -> dict:
        """Transform code search results to compact MCP format."""
        return self.operations.transform_code_to_compact_format(results, query, execution_time_ms)

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
        """Generate a concise summary for code snippets."""
        content_lower = content.lower()
        line_count = len([line for line in lines if line.strip()])
        
        # Simple language/type detection
        if 'def ' in content or 'function ' in content:
            code_type = "function"
        elif 'class ' in content:
            code_type = "class"
        elif any(keyword in content_lower for keyword in ['config', 'settings', 'options']):
            code_type = "configuration"
        elif any(keyword in content_lower for keyword in ['import', 'from ', 'require']):
            code_type = "imports"
        else:
            code_type = "code"
            
        return f"{line_count}-line {code_type} snippet"

    # Context Export/Import/Merge Operations
    async def export_context(self, context_name: str) -> dict:
        """Export complete context data for backup/migration."""
        context = await self.get_context_by_name(context_name)
        if not context:
            raise ValueError(f"Context '{context_name}' not found")
        
        return await self.contexts.export_context_data(context["id"])

    async def import_context(self, import_request: dict) -> dict:
        """Import context data from export."""
        context_data = import_request["context_data"]
        overwrite_existing = import_request.get("overwrite_existing", False)
        
        return await self.contexts.import_context_data(context_data, overwrite_existing)

    async def merge_contexts(self, source_contexts: list[str], target_context: str, mode: str) -> dict:
        """Merge multiple contexts into a target context."""
        # Get source context IDs
        source_context_ids = []
        for source_name in source_contexts:
            context = await self.get_context_by_name(source_name)
            if not context:
                raise ValueError(f"Source context '{source_name}' not found")
            source_context_ids.append(context["id"])
        
        # Get or create target context
        target_context_data = await self.get_context_by_name(target_context)
        if not target_context_data:
            # Create new target context
            target_context_data = await self.create_context(target_context, f"Merged from {', '.join(source_contexts)}")
        
        return await self.contexts.merge_contexts(source_context_ids, target_context_data["id"], mode)


__all__ = ["DatabaseManager"]