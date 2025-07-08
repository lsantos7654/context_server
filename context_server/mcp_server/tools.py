"""MCP tools for Context Server integration."""

import logging
from typing import Any, Optional

from .client import ContextServerClient, ContextServerError

logger = logging.getLogger(__name__)


class ContextServerTools:
    """Collection of MCP tools for Context Server integration."""

    def __init__(self, client: ContextServerClient):
        """Initialize tools with Context Server client."""
        self.client = client

    # Context Management Tools

    async def create_context(
        self,
        name: str,
        description: str = "",
        embedding_model: str = "text-embedding-3-small",
    ) -> dict[str, Any]:
        """Create a new context for storing documentation.

        Args:
            name: Name of the context (must be unique)
            description: Optional description of the context
            embedding_model: Embedding model to use for vector search

        Returns:
            Dictionary with context information including ID, creation time, etc.

        Raises:
            ContextServerError: If context creation fails or name already exists
        """
        try:
            data = {
                "name": name,
                "description": description,
                "embedding_model": embedding_model,
            }

            result = await self.client.post("/api/contexts", data)
            logger.info(f"Created context: {name}")
            return result

        except ContextServerError as e:
            if e.status_code == 409:
                raise ContextServerError(f"Context '{name}' already exists")
            raise

    async def list_contexts(self) -> list[dict[str, Any]]:
        """List all available contexts.

        Returns:
            List of context dictionaries with metadata
        """
        try:
            result = await self.client.get("/api/contexts")
            logger.info(f"Listed {len(result)} contexts")
            return result

        except ContextServerError:
            raise

    async def get_context(self, context_name: str) -> dict[str, Any]:
        """Get detailed information about a specific context.

        Args:
            context_name: Name of the context to retrieve

        Returns:
            Dictionary with context details including document count, size, etc.

        Raises:
            ContextServerError: If context not found
        """
        try:
            result = await self.client.get(f"/api/contexts/{context_name}")
            logger.info(f"Retrieved context: {context_name}")
            return result

        except ContextServerError as e:
            if e.status_code == 404:
                raise ContextServerError(f"Context '{context_name}' not found")
            raise

    async def delete_context(self, context_name: str) -> dict[str, Any]:
        """Delete a context and all its data.

        Args:
            context_name: Name of the context to delete

        Returns:
            Dictionary indicating success

        Raises:
            ContextServerError: If context not found or deletion fails
        """
        try:
            result = await self.client.delete(f"/api/contexts/{context_name}")
            logger.info(f"Deleted context: {context_name}")
            return result or {
                "success": True,
                "message": f"Context '{context_name}' deleted",
            }

        except ContextServerError as e:
            if e.status_code == 404:
                raise ContextServerError(f"Context '{context_name}' not found")
            raise

    # Document Ingestion Tools

    async def extract_url(
        self, context_name: str, url: str, max_pages: int = 50
    ) -> dict[str, Any]:
        """Extract and index documentation from a website URL.

        Args:
            context_name: Name of context to store the documentation
            url: Website URL to scrape (e.g., https://docs.python.org)
            max_pages: Maximum number of pages to crawl

        Returns:
            Dictionary with job information including job_id and status

        Raises:
            ContextServerError: If context not found or extraction fails
        """
        try:
            data = {
                "source_type": "url",
                "source": url,
                "options": {"max_pages": max_pages},
            }

            result = await self.client.post(
                f"/api/contexts/{context_name}/documents", data
            )
            logger.info(f"Started URL extraction: {url} -> {context_name}")
            return result

        except ContextServerError as e:
            if e.status_code == 404:
                raise ContextServerError(f"Context '{context_name}' not found")
            raise

    async def extract_file(self, context_name: str, file_path: str) -> dict[str, Any]:
        """Extract and index content from a local file.

        Args:
            context_name: Name of context to store the content
            file_path: Path to local file (txt, md, rst supported)

        Returns:
            Dictionary with job information including job_id and status

        Raises:
            ContextServerError: If context not found or file processing fails
        """
        try:
            data = {"source_type": "file", "source": file_path, "options": {}}

            result = await self.client.post(
                f"/api/contexts/{context_name}/documents", data
            )
            logger.info(f"Started file extraction: {file_path} -> {context_name}")
            return result

        except ContextServerError as e:
            if e.status_code == 404:
                raise ContextServerError(f"Context '{context_name}' not found")
            raise

    # Search and Retrieval Tools

    async def search_context(
        self, context_name: str, query: str, mode: str = "hybrid", limit: int = 10
    ) -> dict[str, Any]:
        """Search for content within a context with compact summaries.

        Args:
            context_name: Name of context to search
            query: Search query text
            mode: Search mode - "hybrid", "vector", or "fulltext"
            limit: Maximum number of results to return

        Returns:
            Dictionary with compact search results optimized for MCP responses

        Raises:
            ContextServerError: If context not found or search fails
        """
        try:
            data = {"query": query, "mode": mode, "limit": limit}

            result = await self.client.post(
                f"/api/contexts/{context_name}/search", data
            )
            
            # Transform results to compact format for MCP
            compact_results = []
            for item in result.get("results", []):
                # Use summary if available, otherwise truncate content
                display_content = item.get("summary")
                if not display_content:
                    # Fallback to truncated content
                    content = item.get("content", "")
                    display_content = content[:150] + "..." if len(content) > 150 else content
                
                # Count code snippets and extract IDs for direct access
                code_snippets = item.get("metadata", {}).get("code_snippets", [])
                code_snippets_count = len(code_snippets) if code_snippets else 0
                
                # Extract code snippet IDs for direct access
                code_snippet_ids = []
                if code_snippets:
                    for snippet in code_snippets:
                        if isinstance(snippet, dict) and "id" in snippet:
                            code_snippet_ids.append(snippet["id"])
                
                compact_result = {
                    "id": item.get("id"),
                    "document_id": item.get("document_id"),
                    "title": item.get("title"),
                    "summary": display_content,
                    "score": item.get("score"),
                    "url": item.get("url"),
                    "has_summary": bool(item.get("summary")),
                    "code_snippets_count": code_snippets_count,
                    "code_snippet_ids": code_snippet_ids,
                    "content_type": item.get("content_type", "chunk"),
                    "chunk_index": item.get("chunk_index"),
                }
                compact_results.append(compact_result)
            
            # Return compact response
            compact_response = {
                "results": compact_results,
                "total": result.get("total", len(compact_results)),
                "query": result.get("query", query),
                "mode": result.get("mode", mode),
                "execution_time_ms": result.get("execution_time_ms", 0),
                "note": "Content summarized for MCP. Use get_document for full content."
            }
            
            logger.info(
                f"MCP search completed: {len(compact_results)} compact results for '{query}' in {context_name}"
            )
            return compact_response

        except ContextServerError as e:
            if e.status_code == 404:
                raise ContextServerError(f"Context '{context_name}' not found")
            raise

    async def get_document(self, context_name: str, doc_id: str) -> dict[str, Any]:
        """Get raw content of a specific document.

        Args:
            context_name: Name of context containing the document
            doc_id: ID of the document to retrieve

        Returns:
            Dictionary with document content and metadata

        Raises:
            ContextServerError: If context or document not found
        """
        try:
            result = await self.client.get(
                f"/api/contexts/{context_name}/documents/{doc_id}/raw"
            )
            logger.info(f"Retrieved document: {doc_id} from {context_name}")
            return result

        except ContextServerError as e:
            if e.status_code == 404:
                if "Context" in str(e):
                    raise ContextServerError(f"Context '{context_name}' not found")
                else:
                    raise ContextServerError(
                        f"Document '{doc_id}' not found in context '{context_name}'"
                    )
            raise

    async def get_code_snippets(self, context_name: str, doc_id: str) -> dict[str, Any]:
        """Get all code snippets from a specific document.

        Args:
            context_name: Name of context containing the document
            doc_id: ID of the document

        Returns:
            Dictionary with list of code snippets and metadata

        Raises:
            ContextServerError: If context or document not found
        """
        try:
            result = await self.client.get(
                f"/api/contexts/{context_name}/documents/{doc_id}/code-snippets"
            )
            snippet_count = len(result.get("snippets", []))
            logger.info(
                f"Retrieved {snippet_count} code snippets from document {doc_id} in {context_name}"
            )
            return result

        except ContextServerError as e:
            if e.status_code == 404:
                if "Context" in str(e):
                    raise ContextServerError(f"Context '{context_name}' not found")
                else:
                    raise ContextServerError(
                        f"Document '{doc_id}' not found in context '{context_name}'"
                    )
            raise

    async def get_code_snippet(
        self, context_name: str, snippet_id: str
    ) -> dict[str, Any]:
        """Get a specific code snippet by ID.

        Args:
            context_name: Name of context containing the snippet
            snippet_id: ID of the code snippet to retrieve

        Returns:
            Dictionary with code snippet content and metadata

        Raises:
            ContextServerError: If context or snippet not found
        """
        try:
            result = await self.client.get(
                f"/api/contexts/{context_name}/code-snippets/{snippet_id}"
            )
            logger.info(f"Retrieved code snippet: {snippet_id} from {context_name}")
            return result

        except ContextServerError as e:
            if e.status_code == 404:
                if "Context" in str(e):
                    raise ContextServerError(f"Context '{context_name}' not found")
                else:
                    raise ContextServerError(
                        f"Code snippet '{snippet_id}' not found in context '{context_name}'"
                    )
            raise

    # Job Management Tools

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get the status of a document extraction job.

        Args:
            job_id: ID of the job to check (returned from extract_url or extract_file)

        Returns:
            Dictionary with job status, progress, and metadata including current phase

        Raises:
            ContextServerError: If job not found
        """
        try:
            result = await self.client.get(f"/api/jobs/{job_id}/status")
            logger.info(f"Retrieved job status: {job_id} - {result.get('status', 'unknown')}")
            return result

        except ContextServerError as e:
            if e.status_code == 404:
                raise ContextServerError(f"Job '{job_id}' not found")
            raise

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a running document extraction job.

        Args:
            job_id: ID of the job to cancel

        Returns:
            Dictionary indicating cancellation success

        Raises:
            ContextServerError: If job not found or cannot be cancelled
        """
        try:
            result = await self.client.delete(f"/api/jobs/{job_id}")
            logger.info(f"Cancelled job: {job_id}")
            return result

        except ContextServerError as e:
            if e.status_code == 404:
                raise ContextServerError(f"Job '{job_id}' not found")
            elif e.status_code == 400:
                raise ContextServerError(f"Cannot cancel job: {e.message}")
            raise

    async def cleanup_completed_jobs(self, days: int = 7) -> dict[str, Any]:
        """Clean up completed/failed jobs older than specified days.

        Args:
            days: Remove jobs completed/failed more than this many days ago

        Returns:
            Dictionary with cleanup statistics

        Raises:
            ContextServerError: If cleanup fails
        """
        try:
            data = {"days": days}
            result = await self.client.post("/api/admin/jobs/cleanup", data)
            cleaned_count = result.get("deleted_count", 0)
            logger.info(f"Cleaned up {cleaned_count} old jobs")
            return result

        except ContextServerError:
            raise

    async def get_active_jobs(self, context_id: str = None) -> dict[str, Any]:
        """Get all active jobs, optionally filtered by context.

        Args:
            context_id: Optional context ID to filter jobs

        Returns:
            Dictionary with list of active jobs and total count

        Raises:
            ContextServerError: If request fails
        """
        try:
            params = {}
            if context_id:
                params["context_id"] = context_id
            
            result = await self.client.get("/api/jobs/active", params)
            active_count = len(result.get("active_jobs", []))
            logger.info(f"Retrieved {active_count} active jobs")
            return result

        except ContextServerError:
            raise

    # Utility Tools

    async def list_documents(
        self, context_name: str, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        """List documents in a context.

        Args:
            context_name: Name of context to list documents from
            limit: Maximum number of documents to return
            offset: Number of documents to skip (for pagination)

        Returns:
            Dictionary with list of documents and pagination info

        Raises:
            ContextServerError: If context not found
        """
        try:
            params = {"limit": limit, "offset": offset}
            result = await self.client.get(
                f"/api/contexts/{context_name}/documents", params
            )
            doc_count = len(result.get("documents", []))
            logger.info(f"Listed {doc_count} documents from {context_name}")
            return result

        except ContextServerError as e:
            if e.status_code == 404:
                raise ContextServerError(f"Context '{context_name}' not found")
            raise

    async def delete_documents(
        self, context_name: str, document_ids: list[str]
    ) -> dict[str, Any]:
        """Delete specific documents from a context.

        Args:
            context_name: Name of context to delete documents from
            document_ids: List of document IDs to delete

        Returns:
            Dictionary with deletion result and count

        Raises:
            ContextServerError: If context not found or deletion fails
        """
        try:
            # Use a custom implementation for DELETE with body
            import httpx

            data = {"document_ids": document_ids}
            url = f"{self.client.base_url}/api/contexts/{context_name}/documents"

            async with httpx.AsyncClient(
                timeout=self.client.config.request_timeout
            ) as http_client:
                response = await http_client.request("DELETE", url, json=data)
                result = self.client._handle_response(response)

            deleted_count = result.get("deleted_count", len(document_ids))
            logger.info(f"Deleted {deleted_count} documents from {context_name}")
            return result

        except ContextServerError as e:
            if e.status_code == 404:
                raise ContextServerError(f"Context '{context_name}' not found")
            raise
