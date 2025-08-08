"""Document processing pipeline - the main coordinator of the Context Server system.

This module orchestrates the three-document processing pipeline:
1. Original: Raw parsed markdown content
2. Code Snippets: Extracted code blocks with voyage-code-3 embeddings
3. Cleaned Markdown: Text with code snippet placeholders for improved search
"""

import asyncio
import logging
import re
from pathlib import Path

from context_server.core.services.embeddings import (
    EmbeddingService,
    VoyageEmbeddingService,
)
from context_server.core.services.extraction import Crawl4aiExtractor
from context_server.core.services.llm import SummarizationService
from context_server.core.text import TextChunker
from context_server.models.domain.documents import (
    ProcessedChunk,
    ProcessedDocument,
    ProcessingResult,
)
from context_server.models.domain.snippets import CodeSnippet, ExtractedCodeSnippet

logger = logging.getLogger(__name__)


class CodeSnippetExtractor:
    """Extracts code snippets from text content."""

    def __init__(self):
        # Enhanced regex patterns for different code block formats
        self.markdown_code_pattern = re.compile(
            r"```(\w*)\n(.*?)\n```", re.DOTALL | re.MULTILINE
        )
        self.inline_code_pattern = re.compile(r"`([^`\n]+)`")

        # Additional patterns for various code block formats
        self.html_code_pattern = re.compile(
            r"<pre[^>]*><code[^>]*>(.*?)</code></pre>", re.DOTALL | re.IGNORECASE
        )
        self.indented_code_pattern = re.compile(
            r"(?:^|\n)((?:    .*(?:\n|$))+)", re.MULTILINE
        )

    def extract_code_snippets(self, text: str) -> list[ExtractedCodeSnippet]:
        """Extract code snippets from text content.

        Returns:
            list[ExtractedCodeSnippet]: Typed code snippet models without embeddings
        """
        code_snippets = []

        # Extract markdown code blocks
        for match in self.markdown_code_pattern.finditer(text):
            code_content = match.group(2).strip()

            if self._is_valid_code_snippet(code_content):
                snippet = ExtractedCodeSnippet(
                    content=code_content,
                    type="code_block",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    char_count=len(code_content),
                    line_count=len(code_content.split("\n")),
                    original_match=match.group(
                        0
                    ),  # Store original text for replacement
                )

                code_snippets.append(snippet)

        # Extract inline code (only if significant)
        for match in self.inline_code_pattern.finditer(text):
            code_content = match.group(1).strip()

            # Only extract inline code that looks substantial
            if len(code_content) > 20 and self._is_valid_code_snippet(code_content):
                snippet = ExtractedCodeSnippet(
                    content=code_content,
                    type="inline_code",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    char_count=len(code_content),
                    line_count=1,
                    original_match=match.group(
                        0
                    ),  # Store original text for replacement
                )

                code_snippets.append(snippet)

        logger.debug(f"Extracted {len(code_snippets)} code snippets")
        return code_snippets

    def _is_valid_code_snippet(self, content: str) -> bool:
        """Check if content looks like a valid code snippet."""
        if not content or len(content.strip()) < 10:
            return False

        # Skip if it's just plain text
        code_indicators = [
            "{",
            "}",
            "(",
            ")",
            "[",
            "]",
            ";",
            "=",
            "->",
            "def ",
            "function",
            "class ",
            "import ",
            "from ",
            "const ",
            "let ",
            "var ",
            "if ",
            "for ",
            "while ",
            "return ",
            "print(",
            "console.log",
            "System.out",
        ]

        return any(indicator in content for indicator in code_indicators)

    def create_cleaned_content_with_real_uuids(
        self, original_text: str, code_snippets_with_uuids: list[dict]
    ) -> str:
        """Create cleaned content by replacing code snippets with placeholders using real UUIDs.

        Args:
            original_text: The original text content
            code_snippets_with_uuids: List of code snippet dicts with real 'uuid' field

        Returns:
            str: Cleaned text with CODE_SNIPPET placeholders containing real UUIDs
        """
        processed_text = original_text

        # Sort code snippets by start position in reverse order to avoid position shifts
        sorted_snippets = sorted(
            code_snippets_with_uuids, key=lambda x: x.get("start_pos", 0), reverse=True
        )

        # Replace each code snippet with a placeholder
        for snippet_info in sorted_snippets:
            original_match = snippet_info.get("original_match", "")
            if original_match and original_match in processed_text:
                # Create placeholder with real UUID
                placeholder = self._create_code_placeholder_with_uuid(snippet_info)
                processed_text = processed_text.replace(original_match, placeholder, 1)

        return processed_text

    def _create_code_placeholder_with_uuid(self, snippet_info: dict) -> str:
        """Create a structured placeholder for a code snippet using real UUID."""
        # Generate a preview of the first 4-5 lines of code
        content = snippet_info["content"]
        preview = self._generate_code_preview(content)

        placeholder = (
            f"[CODE_SNIPPET: "
            f"size={snippet_info['char_count']}_chars, "
            f'preview="{preview}", '
            f"snippet_id={snippet_info['uuid']}]"
        )

        return placeholder

    def _generate_code_preview(self, code: str) -> str:
        """Generate a preview of the first 4-5 lines of code."""
        lines = code.split("\n")

        # Get first 4-5 meaningful lines (skip empty lines and comments)
        meaningful_lines = []
        for line in lines:
            stripped = line.strip()
            if (
                stripped
                and not stripped.startswith("#")
                and not stripped.startswith("//")
            ):
                meaningful_lines.append(line.rstrip())
                if len(meaningful_lines) >= 4:
                    break

        # If we don't have enough meaningful lines, take first few lines regardless
        if len(meaningful_lines) < 2:
            meaningful_lines = [line.rstrip() for line in lines[:4] if line.strip()]

        # Join with space and truncate to reasonable length for placeholder
        preview = " ".join(meaningful_lines)
        if len(preview) > 200:
            preview = preview[:197] + "..."

        # Escape quotes in preview to avoid breaking the placeholder format
        preview = preview.replace('"', '\\"')

        return preview


class DocumentProcessor:
    """Main document processor that orchestrates the three-document pipeline."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        code_embedding_service: VoyageEmbeddingService | None = None,
        summarization_service: SummarizationService | None = None,
    ):
        """Initialize the document processor with services."""
        self.embedding_service = embedding_service or EmbeddingService()
        self.code_embedding_service = code_embedding_service or VoyageEmbeddingService()
        self.summarization_service = summarization_service or SummarizationService()

        self.extractor = Crawl4aiExtractor()
        self.text_chunker = TextChunker(
            chunk_size=4000,  # Even larger chunks for better context (was 2500)
            chunk_overlap=800,  # 20% overlap for context continuity (was 500)
            chunk_type="text",
        )
        self.code_extractor = CodeSnippetExtractor()

    async def process_url(
        self, url: str, options: dict | None = None, job_id: str | None = None, db=None
    ) -> ProcessingResult:
        """Process a URL through the complete pipeline."""
        try:
            logger.info(f"Starting URL processing: {url}")

            # Extract max_pages from options if provided, otherwise use default
            max_pages = options.get("max_pages", 50) if options else 50

            # Report progress if job tracking is enabled
            if job_id and db:
                await db.update_job_progress(
                    job_id, 0.1, metadata={"phase": "extracting", "url": url}
                )

            # Step 1: Extract content from URL
            extraction_result = await self.extractor.extract_url(url, max_pages)

            if not extraction_result.success:
                return ProcessingResult(
                    documents=[],
                    success=False,
                    error=f"Extraction failed: {extraction_result.error}",
                )

            # Step 2: Always process as an array of pages (unified processing)
            processed_documents = []

            # Debug logging to understand what we got from extraction
            logger.info(
                f"Extraction result: is_multi_page={extraction_result.is_multi_page}, "
                f"individual_pages={len(extraction_result.individual_pages) if extraction_result.individual_pages else 0}"
            )

            # Always process as an array of pages
            pages_to_process = []

            if extraction_result.is_multi_page and extraction_result.individual_pages:
                # Use the individual pages from multi-page extraction
                pages_to_process = extraction_result.individual_pages
            else:
                # Create a single PageResult from the main extraction
                from context_server.core.services.extraction import PageResult

                single_page = PageResult(
                    url=url,
                    title=extraction_result.metadata.get("title", "Untitled"),
                    content=extraction_result.content,
                    metadata=extraction_result.metadata,
                )
                pages_to_process = [single_page]

            # Now process ALL cases with parallel logic
            logger.info(f"Processing {len(pages_to_process)} page(s) in parallel")

            # Create a semaphore to limit concurrent document processing
            max_concurrent_docs = min(
                3, len(pages_to_process)
            )  # Max 3 concurrent documents
            semaphore = asyncio.Semaphore(max_concurrent_docs)
            completed_count = 0

            async def process_single_document(i: int, page) -> "ProcessedDocument":
                """Process a single document with concurrency control."""
                async with semaphore:
                    # Report starting to process this document
                    if job_id and db:
                        await db.update_job_progress(
                            job_id,
                            0.1,  # Don't calculate progress linearly for parallel processing
                            metadata={
                                "phase": "processing_document_start",
                                "current_page": i + 1,
                                "total_pages": len(pages_to_process),
                                "current_document_url": page.url,
                                "current_document_title": page.title,
                                "document_status": "starting",
                            },
                        )

                    processed_doc = await self._process_content(
                        url=page.url,
                        title=page.title,
                        content=page.content,
                        metadata=page.metadata,
                        job_id=job_id,
                        db=db,
                    )

                    # Report completion of this document
                    nonlocal completed_count
                    completed_count += 1
                    if job_id and db:
                        progress = 0.1 + (
                            0.7 * (completed_count / len(pages_to_process))
                        )
                        await db.update_job_progress(
                            job_id,
                            progress,
                            metadata={
                                "phase": "processing_document_complete",
                                "current_page": i + 1,
                                "total_pages": len(pages_to_process),
                                "current_document_url": page.url,
                                "current_document_title": page.title,
                                "document_status": "completed",
                                "completed_count": completed_count,
                                "processing_mode": "parallel",
                            },
                        )

                    logger.info(
                        f"✓ Processed document {i+1}/{len(pages_to_process)} ({completed_count}/{len(pages_to_process)} total): {page.url}"
                    )
                    return processed_doc

            # Process all documents concurrently
            logger.info(
                f"Starting parallel processing with max {max_concurrent_docs} concurrent documents"
            )
            document_tasks = [
                process_single_document(i, page)
                for i, page in enumerate(pages_to_process)
            ]

            # Wait for all documents to complete
            processed_documents = await asyncio.gather(
                *document_tasks, return_exceptions=True
            )

            # Filter out any exceptions and log errors
            successful_documents = []
            for i, result in enumerate(processed_documents):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process document {i+1}: {result}")
                else:
                    successful_documents.append(result)

            processed_documents = successful_documents

            logger.info(
                f"Successfully processed {len(processed_documents)} documents from URL: {url}"
            )
            return ProcessingResult(documents=processed_documents, success=True)

        except Exception as e:
            error_msg = f"Failed to process URL {url}: {str(e)}"
            logger.error(error_msg)
            return ProcessingResult(documents=[], success=False, error=error_msg)

    async def process_file(
        self,
        file_path: str | Path,
        options: dict | None = None,
        job_id: str | None = None,
        db=None,
    ) -> ProcessingResult:
        """Process a local file through the complete pipeline."""
        try:
            logger.info(f"Starting file processing: {file_path}")

            # Report progress if job tracking is enabled
            if job_id and db:
                await db.update_job_progress(
                    job_id,
                    0.1,
                    metadata={"phase": "reading_file", "file_path": str(file_path)},
                )

            # Step 1: Extract content from file
            extraction_result = await self.extractor.extract_file(file_path)

            if not extraction_result.success:
                return ProcessingResult(
                    documents=[],
                    success=False,
                    error=f"File extraction failed: {extraction_result.error}",
                )

            # Step 2: Process the extracted content
            processed_doc = await self._process_content(
                url=f"file://{file_path}",
                title=extraction_result.metadata.get("title", "Untitled"),
                content=extraction_result.content,
                metadata=extraction_result.metadata,
                job_id=job_id,
                db=db,
            )

            logger.info(f"Successfully processed file: {file_path}")
            return ProcessingResult(documents=[processed_doc], success=True)

        except Exception as e:
            error_msg = f"Failed to process file {file_path}: {str(e)}"
            logger.error(error_msg)
            return ProcessingResult(documents=[], success=False, error=error_msg)

    async def _process_content(
        self,
        url: str,
        title: str,
        content: str,
        metadata: dict,
        job_id: str | None = None,
        db=None,
    ) -> ProcessedDocument:
        """Process content through the three-document pipeline.

        NEW SIMPLIFIED FLOW:
        1. Extract code snippet content (no IDs yet)
        2. Generate embeddings for code snippets
        3. Store code snippets to get real UUIDs (this happens in API layer)
        4. Create cleaned content with real UUIDs
        5. Process text chunks with corrected content
        """

        # Step 1: Extract code snippets (no generic IDs generated)
        if job_id and db:
            await db.update_job_progress(
                job_id,
                0.25,
                metadata={
                    "phase": "code_extraction",
                    "status": "analyzing code blocks",
                    "content_size": len(content),
                    "current_document_url": url,
                    "current_document_title": title,
                },
            )

        code_snippets_info = self.code_extractor.extract_code_snippets(content)

        # Step 2: Generate embeddings for code snippets
        code_snippets = []
        if code_snippets_info:
            logger.info(f"Processing {len(code_snippets_info)} code snippets")

            if job_id and db:
                await db.update_job_progress(
                    job_id,
                    0.4,
                    metadata={
                        "phase": "code_embedding",
                        "snippets_found": len(code_snippets_info),
                        "status": "generating code embeddings",
                        "model": "voyage-code-3",
                        "current_document_url": url,
                        "current_document_title": title,
                    },
                )

            code_contents = [snippet.content for snippet in code_snippets_info]

            try:
                code_embeddings = await self.code_embedding_service.embed_batch(
                    code_contents
                )

                for snippet_info, embedding in zip(code_snippets_info, code_embeddings):
                    if embedding:  # Only include if embedding was successful
                        code_snippet = CodeSnippet(
                            content=snippet_info.content,
                            embedding=embedding,
                            metadata={
                                "snippet_type": snippet_info.type,
                                "char_count": snippet_info.char_count,
                                "line_count": snippet_info.line_count,
                                # Store original match info for placeholder creation
                                "start_pos": snippet_info.start_pos,
                                "end_pos": snippet_info.end_pos,
                                "original_match": snippet_info.original_match,
                            },
                        )
                        code_snippets.append(code_snippet)

            except Exception as e:
                logger.warning(f"Code embedding failed: {e}")
                # Continue without code embeddings

        # Step 3: Create cleaned content with temporary placeholders for chunking
        # We'll create placeholders with temporary IDs, then replace with real UUIDs in API layer
        temp_cleaned_content = self._create_temp_cleaned_content(
            content, code_snippets_info
        )

        # Step 4: Chunk the content
        if job_id and db:
            await db.update_job_progress(
                job_id,
                0.55,
                metadata={
                    "phase": "text_chunking",
                    "status": "creating text chunks",
                    "content_size": len(temp_cleaned_content),
                    "code_snippets_processed": len(code_snippets),
                    "current_document_url": url,
                    "current_document_title": title,
                },
            )

        text_chunks = self.text_chunker.chunk_text(temp_cleaned_content)

        # Step 5: Generate embeddings and summaries for text chunks
        processed_chunks = []
        if text_chunks:
            logger.info(f"Processing {len(text_chunks)} text chunks")

            if job_id and db:
                await db.update_job_progress(
                    job_id,
                    0.7,
                    metadata={
                        "phase": "text_embedding",
                        "chunks_created": len(text_chunks),
                        "status": "generating text embeddings and summaries",
                        "embedding_model": "text-embedding-3-large",
                        "summary_model": getattr(
                            self.summarization_service, "model", "gpt-4o-mini"
                        ),
                        "current_document_url": url,
                        "current_document_title": title,
                    },
                )

            # Generate embeddings
            chunk_contents = [chunk.content for chunk in text_chunks]

            try:
                # Generate embeddings, titles, and summaries concurrently
                embeddings_task = self.embedding_service.embed_batch(chunk_contents)
                titles_summaries_task = (
                    self.summarization_service.generate_titles_and_summaries_batch(
                        chunk_contents
                    )
                )

                embeddings, titles_summaries = await asyncio.gather(
                    embeddings_task, titles_summaries_task, return_exceptions=True
                )

                # Handle exceptions
                if isinstance(embeddings, Exception):
                    logger.warning(f"Text embedding failed: {embeddings}")
                    embeddings = [None] * len(chunk_contents)

                if isinstance(titles_summaries, Exception):
                    logger.warning(
                        f"Title and summary generation failed: {titles_summaries}"
                    )
                    titles_summaries = [(None, None)] * len(chunk_contents)

                # Extract titles and summaries from tuple results
                titles = [ts[0] if ts else None for ts in titles_summaries]
                summaries = [ts[1] if ts else None for ts in titles_summaries]

                # Create processed chunks (code snippet linking will be done in API layer)
                for chunk, embedding, title, summary in zip(
                    text_chunks, embeddings, titles, summaries
                ):
                    if embedding:  # Only include if embedding was successful
                        processed_chunk = ProcessedChunk(
                            content=chunk.content,
                            embedding=embedding,
                            metadata=chunk.metadata,
                            tokens=chunk.tokens,
                            title=title,
                            summary=summary,
                            summary_model=(
                                self.summarization_service.model if summary else None
                            ),
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                            char_start=chunk.char_start,
                            char_end=chunk.char_end,
                        )
                        processed_chunks.append(processed_chunk)

            except Exception as e:
                logger.error(f"Text processing failed: {e}")
                # Continue with reduced functionality

        # Step 6: Create final processed document
        processed_document = ProcessedDocument(
            url=url,
            title=title,
            content=content,  # Keep original content
            cleaned_content=None,  # Will be created in API layer with real UUIDs
            chunks=processed_chunks,
            code_snippets=code_snippets,
            metadata={
                **metadata,
                "processing_stats": {
                    "original_length": len(content),
                    "total_chunks": len(processed_chunks),
                    "total_code_snippets": len(code_snippets),
                    "chunks_with_embeddings": len(processed_chunks),
                    "code_snippets_with_embeddings": len(code_snippets),
                },
            },
        )

        logger.info(
            f"Document processing complete: {len(processed_chunks)} chunks, {len(code_snippets)} code snippets"
        )
        return processed_document

    def _create_temp_cleaned_content(
        self, original_content: str, code_snippets_info: list[ExtractedCodeSnippet]
    ) -> str:
        """Create cleaned content with temporary placeholders for chunking purposes.

        This creates placeholders using temporary IDs that will be replaced with real UUIDs
        in the API layer after code snippets are stored in the database.
        """
        if not code_snippets_info:
            return original_content

        processed_text = original_content

        # Sort code snippets by start position in reverse order to avoid position shifts
        sorted_snippets = sorted(
            code_snippets_info, key=lambda x: x.start_pos, reverse=True
        )

        # Replace each code snippet with a temporary placeholder
        for i, snippet_info in enumerate(sorted_snippets):
            original_match = snippet_info.original_match
            if original_match and original_match in processed_text:
                # Create temporary placeholder (will be replaced with real UUID later)
                temp_id = f"temp_snippet_{i}"
                preview = self.code_extractor._generate_code_preview(
                    snippet_info.content
                )

                placeholder = (
                    f"[CODE_SNIPPET: "
                    f"size={snippet_info.char_count}_chars, "
                    f'preview="{preview}", '
                    f"snippet_id={temp_id}]"
                )
                processed_text = processed_text.replace(original_match, placeholder, 1)

        return processed_text

    def _find_code_snippets_in_chunk(
        self, chunk_content: str, code_snippets_info: list[ExtractedCodeSnippet]
    ) -> list[str]:
        """Find which code snippets are referenced in a chunk."""
        referenced_snippet_ids = []

        # Look for CODE_SNIPPET placeholders in the chunk content
        placeholder_pattern = r"\[CODE_SNIPPET:.*?snippet_id=([^]]+)\]"
        matches = re.findall(placeholder_pattern, chunk_content)

        for match in matches:
            # Extract the snippet ID (removing any quotes)
            snippet_id = match.strip("\"'")
            if snippet_id not in referenced_snippet_ids:
                referenced_snippet_ids.append(snippet_id)

        return referenced_snippet_ids


__all__ = ["DocumentProcessor", "CodeSnippetExtractor", "ProcessingResult"]
