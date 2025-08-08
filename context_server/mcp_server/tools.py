"""MCP tools for Context Server integration."""

import logging

from context_server.mcp_server.client import ContextServerClient, ContextServerError
from context_server.models.api.contexts import (
    ContextDeleteResponse,
    ContextListResponse,
    ContextResponse,
)
from context_server.models.api.documents import (
    ChunkResponse,
    CodeSnippetResponse,
    CodeSnippetsResponse,
    DirectoryExtractionResponse,
    DocumentContentResponse,
    DocumentDeleteResponse,
    DocumentsResponse,
    FileProcessingResult,
)
from context_server.models.api.search import (
    CompactCodeSearchResponse,
    CompactSearchResponse,
)
from context_server.models.api.system import (
    ActiveJobsResponse,
    JobCancelResponse,
    JobCleanupResponse,
    JobCreateResponse,
    JobStatusResponse,
)

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
        embedding_model: str = "text-embedding-3-large",
    ) -> ContextResponse:
        """Create a new context for storing documentation.

        Args:
            name: Name of the context (must be unique)
            description: Optional description of the context
            embedding_model: Embedding model to use for vector search

        Returns:
            ContextResponse with context information including ID, creation time, etc.

        Raises:
            ContextServerError: If context creation fails or name already exists
        """
        try:
            # Ensure we use the correct default if embedding_model is not provided
            if not embedding_model:
                embedding_model = "text-embedding-3-large"

            data = {
                "name": name,
                "description": description,
                "embedding_model": embedding_model,
            }

            result = await self.client.post("/api/contexts", data)
            logger.info(f"Created context: {name}")
            # Client now returns typed response, so we can use it directly
            if isinstance(result, ContextResponse):
                return result
            else:
                # Fallback for unexpected response format
                return ContextResponse(**result)

        except ContextServerError as e:
            if e.status_code == 409:
                raise ContextServerError(f"Context '{name}' already exists")
            raise

    async def list_contexts(self) -> ContextListResponse:
        """List all available contexts.

        Returns:
            ContextListResponse with list of contexts and metadata
        """
        try:
            result = await self.client.get("/api/contexts")
            # Client now returns typed response, so we can use it directly
            if isinstance(result, ContextListResponse):
                logger.info(f"Listed {len(result.contexts)} contexts")
                return result
            else:
                # Fallback for unexpected response format
                contexts = (
                    [ContextResponse(**ctx) for ctx in result]
                    if isinstance(result, list)
                    else []
                )
                logger.info(f"Listed {len(contexts)} contexts")
                return ContextListResponse(contexts=contexts, total=len(contexts))

        except ContextServerError:
            raise

    async def get_context(self, context_name: str) -> ContextResponse:
        """Get detailed information about a specific context.

        Args:
            context_name: Name of the context to retrieve

        Returns:
            ContextResponse with context details including document count, size, etc.

        Raises:
            ContextServerError: If context not found
        """
        try:
            result = await self.client.get(f"/api/contexts/{context_name}")
            logger.info(f"Retrieved context: {context_name}")
            # Client now returns typed response, so we can use it directly
            if isinstance(result, ContextResponse):
                return result
            else:
                # Fallback for unexpected response format
                return ContextResponse(**result)

        except ContextServerError as e:
            if e.status_code == 404:
                raise ContextServerError(f"Context '{context_name}' not found")
            raise

    async def delete_context(self, context_name: str) -> ContextDeleteResponse:
        """Delete a context and all its data.

        Args:
            context_name: Name of the context to delete

        Returns:
            ContextDeleteResponse indicating success

        Raises:
            ContextServerError: If context not found or deletion fails
        """
        try:
            result = await self.client.delete(f"/api/contexts/{context_name}")
            logger.info(f"Deleted context: {context_name}")

            # Client now returns typed response, so we can use it directly
            if isinstance(result, ContextDeleteResponse):
                return result
            else:
                # Fallback for unexpected response format
                return ContextDeleteResponse(
                    success=True, message=f"Context '{context_name}' deleted"
                )

        except ContextServerError as e:
            if e.status_code == 404:
                raise ContextServerError(f"Context '{context_name}' not found")
            raise

    # Document Ingestion Tools

    async def extract_url(
        self, context_name: str, url: str, max_pages: int = 50
    ) -> JobCreateResponse:
        """Extract and index documentation from a website URL.

        Args:
            context_name: Name of context to store the documentation
            url: Website URL to scrape (e.g., https://docs.python.org)
            max_pages: Maximum number of pages to crawl

        Returns:
            JobCreateResponse with job information including job_id and status

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
            return JobCreateResponse(**result)

        except ContextServerError as e:
            if e.status_code == 404:
                raise ContextServerError(f"Context '{context_name}' not found")
            raise

    async def extract_file(
        self, context_name: str, file_path: str
    ) -> JobCreateResponse:
        """Extract and index content from a local file.

        Args:
            context_name: Name of context to store the content
            file_path: Path to local file (txt, md, rst supported)

        Returns:
            JobCreateResponse with job information including job_id and status

        Raises:
            ContextServerError: If context not found or file processing fails
        """
        try:
            data = {"source_type": "file", "source": file_path, "options": {}}

            result = await self.client.post(
                f"/api/contexts/{context_name}/documents", data
            )
            logger.info(f"Started file extraction: {file_path} -> {context_name}")
            return JobCreateResponse(**result)

        except ContextServerError as e:
            if e.status_code == 404:
                raise ContextServerError(f"Context '{context_name}' not found")
            raise

    async def extract_local_directory(
        self,
        context_name: str,
        directory_path: str,
        include_patterns: list[str] = None,
        exclude_patterns: list[str] = None,
        max_files: int = 100,
    ) -> DirectoryExtractionResponse:
        """Extract and index content from a local directory.

        Args:
            context_name: Name of context to store the content
            directory_path: Path to local directory to scan
            include_patterns: File patterns to include (e.g., ['*.md', '*.py'])
            exclude_patterns: File patterns to exclude (e.g., ['*.pyc', '__pycache__'])
            max_files: Maximum number of files to process (safety limit)

        Returns:
            Dictionary with extraction summary including processed files count

        Raises:
            ContextServerError: If context not found or directory processing fails
        """
        try:
            import fnmatch
            from pathlib import Path

            source_path = Path(directory_path)
            if not source_path.exists():
                raise ContextServerError(f"Directory does not exist: {directory_path}")

            if not source_path.is_dir():
                raise ContextServerError(f"Path is not a directory: {directory_path}")

            # Set default patterns if none provided
            if not include_patterns:
                include_patterns = [
                    "*.md",
                    "*.txt",
                    "*.rst",
                    "*.py",
                    "*.js",
                    "*.ts",
                    "*.html",
                    "*.json",
                    "*.yaml",
                    "*.yml",
                ]

            if not exclude_patterns:
                exclude_patterns = [
                    "*.pyc",
                    "__pycache__",
                    ".git",
                    ".venv",
                    "node_modules",
                    "*.log",
                    ".DS_Store",
                    "*.tmp",
                    "dist",
                    "build",
                    ".env",
                ]

            # Collect files to process
            files_to_process = []

            for file_path in source_path.rglob("*"):
                if file_path.is_file():
                    # Check exclude patterns first
                    excluded = False
                    for pattern in exclude_patterns:
                        if fnmatch.fnmatch(file_path.name, pattern) or fnmatch.fnmatch(
                            str(file_path.relative_to(source_path)), pattern
                        ):
                            excluded = True
                            break

                    if excluded:
                        continue

                    # Check include patterns
                    included = False
                    for pattern in include_patterns:
                        if fnmatch.fnmatch(file_path.name, pattern):
                            included = True
                            break

                    if included:
                        files_to_process.append(file_path)

                        # Safety limit to prevent processing too many files
                        if len(files_to_process) >= max_files:
                            logger.warning(
                                f"Reached max files limit ({max_files}), stopping scan"
                            )
                            break

            if not files_to_process:
                return DirectoryExtractionResponse(
                    success=True,
                    message="No files found matching the criteria",
                    processed_files=0,
                    failed_files=0,
                    total_files=0,
                    directory_path=directory_path,
                    files=[],
                )

            logger.info(
                f"Found {len(files_to_process)} files to process in {directory_path}"
            )

            # Process files one by one through the API
            processed_files = 0
            failed_files = 0
            file_results = []

            for file_path in files_to_process:
                try:
                    # Read file content
                    try:
                        content = file_path.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        try:
                            content = file_path.read_text(encoding="latin-1")
                        except:
                            logger.warning(f"Skipping binary file: {file_path}")
                            failed_files += 1
                            file_results.append(
                                FileProcessingResult(
                                    file=str(file_path),
                                    status="skipped",
                                    error="Binary file",
                                )
                            )
                            continue

                    # Send to API for processing
                    data = {
                        "source_type": "file",
                        "source": str(file_path),
                        "options": {
                            "filename": file_path.name,
                            "relative_path": str(file_path.relative_to(source_path)),
                            "directory_extraction": True,
                        },
                    }

                    result = await self.client.post(
                        f"/api/contexts/{context_name}/documents", data
                    )

                    processed_files += 1
                    file_results.append(
                        FileProcessingResult(
                            file=str(file_path.relative_to(source_path)),
                            status="processed",
                            job_id=result.get("job_id"),
                        )
                    )

                    logger.debug(
                        f"Processed file {processed_files}/{len(files_to_process)}: {file_path.name}"
                    )

                except Exception as e:
                    failed_files += 1
                    file_results.append(
                        FileProcessingResult(
                            file=str(file_path.relative_to(source_path)),
                            status="failed",
                            error=str(e),
                        )
                    )
                    logger.error(f"Failed to process {file_path}: {e}")

            logger.info(
                f"Directory extraction completed: {processed_files} processed, {failed_files} failed"
            )

            return DirectoryExtractionResponse(
                success=True,
                message=f"Processed {processed_files} files from directory",
                processed_files=processed_files,
                failed_files=failed_files,
                total_files=len(files_to_process),
                directory_path=directory_path,
                files=file_results,
            )

        except ContextServerError:
            raise
        except Exception as e:
            logger.error(f"Directory extraction failed: {e}")
            raise ContextServerError(f"Directory extraction failed: {str(e)}")

    # Search and Retrieval Tools

    async def search_context(
        self, context_name: str, query: str, mode: str = "hybrid", limit: int = 10
    ) -> CompactSearchResponse:
        """Search for content within a context with compact summaries.

        Args:
            context_name: Name of context to search
            query: Search query text
            mode: Search mode - "hybrid", "vector", or "fulltext"
            limit: Maximum number of results to return

        Returns:
            CompactSearchResponse with compact search results optimized for MCP responses

        Raises:
            ContextServerError: If context not found or search fails
        """
        try:
            data = {"query": query, "mode": mode, "limit": limit}
            params = {"format": "compact"}

            result = await self.client.post(
                f"/api/contexts/{context_name}/search", data, params=params
            )

            logger.info(
                f"MCP search completed: {len(result.get('results', []))} compact results for '{query}' in {context_name}"
            )
            return CompactSearchResponse(**result)

        except ContextServerError as e:
            if e.status_code == 404:
                raise ContextServerError(f"Context '{context_name}' not found")
            raise

    async def get_document(
        self,
        context_name: str,
        doc_id: str,
        page_number: int = 1,
        page_size: int = 20000,
    ) -> DocumentContentResponse:
        """Get raw content of a specific document with pagination support.

        Args:
            context_name: Name of context containing the document
            doc_id: ID of the document to retrieve
            page_number: Page number to retrieve (1-based)
            page_size: Number of characters per page (default: 20000 for Claude's limit)

        Returns:
            Dictionary with document content and pagination metadata

        Raises:
            ContextServerError: If context or document not found
        """
        try:
            params = {"page_number": page_number, "page_size": page_size}

            result = await self.client.get(
                f"/api/contexts/{context_name}/documents/{doc_id}/raw", params
            )

            # Add pagination metadata to the response
            content_length = len(result.get("content", ""))
            total_pages = max(
                1,
                (result.get("full_content_length", content_length) + page_size - 1)
                // page_size,
            )

            # Only include essential fields to reduce response size
            # Exclude metadata which can contain large extracted page content
            paginated_result = {
                "id": result.get("id"),
                "title": result.get("title"),
                "url": result.get("url"),
                "content": result.get("content", ""),
                "document_type": result.get("document_type"),
                "created_at": result.get("created_at"),
                "full_content_length": result.get(
                    "full_content_length", content_length
                ),
                "pagination": {
                    "page_number": page_number,
                    "page_size": page_size,
                    "total_pages": total_pages,
                    "current_page_length": content_length,
                    "has_next_page": page_number < total_pages,
                    "has_previous_page": page_number > 1,
                },
                "note": "Metadata excluded to reduce response size. Full metadata available via direct API call.",
            }

            logger.info(
                f"Retrieved document page {page_number}/{total_pages}: {doc_id} from {context_name}"
            )
            return DocumentContentResponse(**paginated_result)

        except ContextServerError as e:
            if e.status_code == 404:
                if "Context" in str(e):
                    raise ContextServerError(f"Context '{context_name}' not found")
                else:
                    raise ContextServerError(
                        f"Document '{doc_id}' not found in context '{context_name}'"
                    )
            raise

    async def get_code_snippets(
        self, context_name: str, doc_id: str
    ) -> CodeSnippetsResponse:
        """Get all code snippets from a specific document.

        Args:
            context_name: Name of context containing the document
            doc_id: ID of the document

        Returns:
            CodeSnippetsResponse with list of code snippets and metadata

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
            # Ensure result has required fields
            if "total" not in result:
                result["total"] = snippet_count
            if "document_id" not in result:
                result["document_id"] = doc_id
            return CodeSnippetsResponse(**result)

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
    ) -> CodeSnippetResponse:
        """Get a specific code snippet by ID.

        Args:
            context_name: Name of context containing the snippet
            snippet_id: ID of the code snippet to retrieve

        Returns:
            CodeSnippetResponse with code snippet content and metadata

        Raises:
            ContextServerError: If context or snippet not found
        """
        try:
            result = await self.client.get(
                f"/api/contexts/{context_name}/code-snippets/{snippet_id}"
            )
            logger.info(f"Retrieved code snippet: {snippet_id} from {context_name}")
            return CodeSnippetResponse(**result)

        except ContextServerError as e:
            if e.status_code == 404:
                if "Context" in str(e):
                    raise ContextServerError(f"Context '{context_name}' not found")
                else:
                    raise ContextServerError(
                        f"Code snippet '{snippet_id}' not found in context '{context_name}'"
                    )
            raise

    async def get_chunk(self, context_name: str, chunk_id: str) -> ChunkResponse:
        """Get a specific chunk by ID with full content and metadata.

        Args:
            context_name: Name of context containing the chunk
            chunk_id: ID of the chunk to retrieve

        Returns:
            ChunkResponse with chunk content and metadata

        Raises:
            ContextServerError: If context or chunk not found
        """
        try:
            result = await self.client.get(
                f"/api/contexts/{context_name}/chunks/{chunk_id}"
            )
            logger.info(f"Retrieved chunk: {chunk_id} from {context_name}")
            return ChunkResponse(**result)

        except ContextServerError as e:
            if e.status_code == 404:
                if "Context" in str(e):
                    raise ContextServerError(f"Context '{context_name}' not found")
                else:
                    raise ContextServerError(
                        f"Chunk '{chunk_id}' not found in context '{context_name}'"
                    )
            raise

    async def search_code(
        self, context_name: str, query: str, language: str = None, limit: int = 10
    ) -> CompactCodeSearchResponse:
        """Search for code snippets within a context using code-optimized embeddings.

        Args:
            context_name: Name of context to search
            query: Search query text (e.g., 'function definition', 'error handling')
            language: Optional language filter (e.g., 'python', 'javascript')
            limit: Maximum number of results to return

        Returns:
            CompactCodeSearchResponse with code search results optimized for development

        Raises:
            ContextServerError: If context not found or search fails
        """
        try:
            data = {
                "query": query,
                "mode": "hybrid",  # Use hybrid search for best results
                "limit": limit,
            }
            params = {"format": "compact"}

            result = await self.client.post(
                f"/api/contexts/{context_name}/search/code", data, params=params
            )

            # Filter by language if specified (note: language field was removed from results)
            # This filtering is now mostly obsolete since language was removed from the data
            if language:
                # Add language filter info to response for user information
                result["language_filter"] = language
                result[
                    "note"
                ] = f"Language filter '{language}' applied (results may not be language-specific)"

            logger.info(
                f"Code search completed: {len(result.get('results', []))} results for '{query}' in {context_name}"
            )
            return CompactCodeSearchResponse(**result)

        except ContextServerError as e:
            if e.status_code == 404:
                raise ContextServerError(f"Context '{context_name}' not found")
            raise

    # Job Management Tools

    async def get_job_status(self, job_id: str) -> JobStatusResponse:
        """Get the status of a document extraction job.

        Args:
            job_id: ID of the job to check (returned from extract_url or extract_file)

        Returns:
            JobStatusResponse with job status, progress, and metadata including current phase

        Raises:
            ContextServerError: If job not found
        """
        try:
            result = await self.client.get(f"/api/jobs/{job_id}/status")
            logger.info(
                f"Retrieved job status: {job_id} - {result.get('status', 'unknown')}"
            )
            return JobStatusResponse(**result)

        except ContextServerError as e:
            if e.status_code == 404:
                raise ContextServerError(f"Job '{job_id}' not found")
            raise

    async def cancel_job(self, job_id: str) -> JobCancelResponse:
        """Cancel a running document extraction job.

        Args:
            job_id: ID of the job to cancel

        Returns:
            JobCancelResponse indicating cancellation success

        Raises:
            ContextServerError: If job not found or cannot be cancelled
        """
        try:
            result = await self.client.delete(f"/api/jobs/{job_id}")
            logger.info(f"Cancelled job: {job_id}")
            return JobCancelResponse(**result)

        except ContextServerError as e:
            if e.status_code == 404:
                raise ContextServerError(f"Job '{job_id}' not found")
            elif e.status_code == 400:
                raise ContextServerError(f"Cannot cancel job: {e.message}")
            raise

    async def cleanup_completed_jobs(self, days: int = 7) -> JobCleanupResponse:
        """Clean up completed/failed jobs older than specified days.

        Args:
            days: Remove jobs completed/failed more than this many days ago

        Returns:
            JobCleanupResponse with cleanup statistics

        Raises:
            ContextServerError: If cleanup fails
        """
        try:
            data = {"days": days}
            result = await self.client.post("/api/admin/jobs/cleanup", data)
            cleaned_count = result.get("deleted_count", 0)
            logger.info(f"Cleaned up {cleaned_count} old jobs")
            return JobCleanupResponse(**result)

        except ContextServerError:
            raise

    async def get_active_jobs(self, context_id: str = None) -> ActiveJobsResponse:
        """Get all active jobs, optionally filtered by context.

        Args:
            context_id: Optional context ID to filter jobs

        Returns:
            ActiveJobsResponse with list of active jobs and total count

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
            return ActiveJobsResponse(**result)

        except ContextServerError:
            raise

    # Utility Tools

    async def list_documents(
        self, context_name: str, limit: int = 50, offset: int = 0
    ) -> DocumentsResponse:
        """List documents in a context.

        Args:
            context_name: Name of context to list documents from
            limit: Maximum number of documents to return
            offset: Number of documents to skip (for pagination)

        Returns:
            DocumentsResponse with list of documents and pagination info

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
            return DocumentsResponse(**result)

        except ContextServerError as e:
            if e.status_code == 404:
                raise ContextServerError(f"Context '{context_name}' not found")
            raise

    async def delete_documents(
        self, context_name: str, document_ids: list[str]
    ) -> DocumentDeleteResponse:
        """Delete specific documents from a context.

        Args:
            context_name: Name of context to delete documents from
            document_ids: List of document IDs to delete

        Returns:
            DocumentDeleteResponse with deletion result and count

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

            # Ensure all required fields are present
            if "message" not in result:
                result["message"] = f"Deleted {deleted_count} documents"
            if "context_name" not in result:
                result["context_name"] = context_name
            if "deleted_count" not in result:
                result["deleted_count"] = deleted_count

            return DocumentDeleteResponse(**result)

        except ContextServerError as e:
            if e.status_code == 404:
                raise ContextServerError(f"Context '{context_name}' not found")
            raise
