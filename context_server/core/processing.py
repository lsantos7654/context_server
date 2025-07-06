"""Document processing service that integrates with existing extraction pipeline."""

import asyncio
import logging
import re

# Import existing extraction functionality
from dataclasses import dataclass
from pathlib import Path

from .chunking import TextChunker
from .crawl4ai_extraction import Crawl4aiExtractor
from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class ProcessedChunk:
    """A processed text chunk with embedding and line tracking."""

    content: str
    embedding: list[float]
    metadata: dict
    tokens: int
    start_line: int = None
    end_line: int = None
    char_start: int = None
    char_end: int = None


@dataclass
class CodeSnippet:
    """A code snippet extracted from content."""

    content: str
    language: str
    embedding: list[float]
    metadata: dict
    start_line: int = None
    end_line: int = None
    char_start: int = None
    char_end: int = None


@dataclass
class ProcessedDocument:
    """A processed document with chunks and code snippets."""

    url: str
    title: str
    content: str
    chunks: list[ProcessedChunk]
    code_snippets: list[CodeSnippet]
    metadata: dict


@dataclass
class ProcessingResult:
    """Result of document processing operation."""

    documents: list[ProcessedDocument]
    success: bool
    error: str | None = None


class CodeSnippetExtractor:
    """Extracts code snippets from text content."""

    def __init__(self):
        # Regex patterns for different code block formats
        self.markdown_code_pattern = re.compile(
            r"```(\w*)\n(.*?)\n```", re.DOTALL | re.MULTILINE
        )
        self.inline_code_pattern = re.compile(r"`([^`\n]+)`")

    def extract_code_snippets(
        self, content: str, url: str = "", title: str = ""
    ) -> tuple[list[dict], str]:
        """
        Extract code snippets from content and return cleaned content.

        Returns:
            Tuple of (code_snippets_list, content_without_code_blocks)
        """
        snippets = []
        lines = content.splitlines()

        # Track character positions
        char_pos = 0
        line_char_map = []
        for line in lines:
            line_char_map.append(char_pos)
            char_pos += len(line) + 1  # +1 for newline

        # Extract markdown code blocks
        for match in self.markdown_code_pattern.finditer(content):
            language = match.group(1) or "text"
            code_content = match.group(2).strip()
            start_char = match.start()
            end_char = match.end()

            # Find line numbers
            start_line = self._char_to_line(start_char, line_char_map)
            end_line = self._char_to_line(end_char, line_char_map)

            snippet = {
                "content": code_content,
                "language": language,
                "start_line": start_line,
                "end_line": end_line,
                "char_start": start_char,
                "char_end": end_char,
                "type": "code_block",
                "source_url": url,
                "source_title": title,
            }
            snippets.append(snippet)

        # Remove code blocks from content for regular chunking
        cleaned_content = self.markdown_code_pattern.sub("", content)

        # Also extract significant inline code (longer than 10 chars)
        for match in self.inline_code_pattern.finditer(content):
            code_content = match.group(1)
            if len(code_content) > 10:  # Only significant inline code
                start_char = match.start()
                end_char = match.end()
                start_line = self._char_to_line(start_char, line_char_map)
                end_line = start_line

                snippet = {
                    "content": code_content,
                    "language": "text",  # Unknown language for inline code
                    "start_line": start_line,
                    "end_line": end_line,
                    "char_start": start_char,
                    "char_end": end_char,
                    "type": "inline_code",
                    "source_url": url,
                    "source_title": title,
                }
                snippets.append(snippet)

        return snippets, cleaned_content

    def _char_to_line(self, char_pos: int, line_char_map: list[int]) -> int:
        """Convert character position to line number."""
        for i, line_start in enumerate(line_char_map):
            if char_pos < line_start + (
                len(line_char_map) > i + 1 and line_char_map[i + 1] - line_start or 0
            ):
                return i
        return len(line_char_map) - 1


class DocumentProcessor:
    """Processes documents using existing extraction pipeline and creates embeddings."""

    def __init__(self, embedding_service: EmbeddingService | None = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.chunker = TextChunker()
        self.extractor = Crawl4aiExtractor()
        self.code_extractor = CodeSnippetExtractor()

    def _extract_links_from_chunk(self, chunk_content: str) -> dict:
        """Extract links from chunk content and return link metadata."""
        import re
        from urllib.parse import urljoin, urlparse

        links_in_chunk = {}

        # Pattern for markdown links: [text](url)
        markdown_pattern = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

        # Find markdown links only (more reliable than raw URL extraction)
        for match in markdown_pattern.finditer(chunk_content):
            text = match.group(1)
            href = match.group(2).strip()

            # Clean up href - remove quotes and extra characters
            href = href.split(" ")[0].strip("\"'")

            # Skip empty, anchor-only, or invalid URLs
            if href and not href.startswith("#") and href.startswith("http"):
                # Validate URL structure
                try:
                    parsed = urlparse(href)
                    if parsed.scheme and parsed.netloc:
                        links_in_chunk[href] = {"text": text, "href": href}
                except:
                    # Skip malformed URLs
                    continue

        return {
            "total_links_in_chunk": len(links_in_chunk),
            "chunk_links": links_in_chunk,
        }

    async def process_url(
        self, url: str, options: dict | None = None, job_id: str | None = None, db=None
    ) -> ProcessingResult:
        """Process a URL with per-document processing and storage."""
        try:
            logger.info(f"Processing URL: {url}")

            # Report crawling phase start
            if job_id and db:
                await db.update_job_progress(
                    job_id, 0.1, metadata={"phase": "crawling", "url": url}
                )

            # Use existing extraction pipeline with timeout
            max_pages = options.get("max_pages", 50) if options else 50

            # Add timeout to extraction
            extraction_timeout = min(
                600, max_pages * 30
            )  # 30s per page, max 10 minutes
            try:
                result = await asyncio.wait_for(
                    self.extractor.extract_from_url(url, max_pages=max_pages),
                    timeout=extraction_timeout,
                )
            except asyncio.TimeoutError:
                logger.error(f"URL extraction timed out after {extraction_timeout}s")
                return ProcessingResult(
                    documents=[],
                    success=False,
                    error=f"Extraction timed out after {extraction_timeout}s",
                )

            if not result.success:
                return ProcessingResult(documents=[], success=False, error=result.error)

            # Report content extraction completed
            if job_id and db:
                await db.update_job_progress(
                    job_id,
                    0.2,
                    metadata={
                        "phase": "content_extracted",
                        "pages_found": len(result.metadata.get("extracted_pages", [])),
                    },
                )

            # Process and store documents individually for fault tolerance
            documents = []
            failed_pages = []
            extracted_pages = result.metadata.get("extracted_pages", [])
            total_pages = len(extracted_pages) if extracted_pages else 1

            if extracted_pages:
                # Process each page individually with immediate storage if db available
                for idx, page_info in enumerate(extracted_pages):
                    page_url = page_info["url"]
                    page_content = page_info["content"]

                    try:
                        # Create page-specific metadata
                        page_metadata = result.metadata.copy()
                        page_metadata["page_url"] = page_url
                        page_metadata["is_individual_page"] = True

                        # Add link counts if available
                        if "link_counts" in page_info:
                            page_metadata.update(page_info["link_counts"])

                        # Create document title from page URL
                        page_title = self._create_title_from_url(page_url, url)

                        # Process content with timeout protection
                        processing_timeout = 300  # 5 minutes per document
                        document = await asyncio.wait_for(
                            self._process_content(
                                content=page_content,
                                url=page_url,
                                title=page_title,
                                metadata=page_metadata,
                            ),
                            timeout=processing_timeout,
                        )
                        documents.append(document)

                        logger.info(
                            f"Successfully processed page {idx + 1}/{total_pages}: {page_title}"
                        )

                    except asyncio.TimeoutError:
                        logger.error(
                            f"Processing timeout for page {idx + 1}/{total_pages}: {page_url}"
                        )
                        failed_pages.append(
                            {"url": page_url, "error": "Processing timeout"}
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to process page {idx + 1}/{total_pages}: {page_url} - {e}"
                        )
                        failed_pages.append({"url": page_url, "error": str(e)})

                    # Report progress after each page
                    if job_id and db:
                        progress = 0.2 + (0.6 * (idx + 1) / total_pages)  # 20% to 80%
                        await db.update_job_progress(
                            job_id,
                            progress,
                            metadata={
                                "phase": "chunking_and_embedding",
                                "processed_pages": idx + 1,
                                "total_pages": total_pages,
                                "successful_pages": len(documents),
                                "failed_pages": len(failed_pages),
                            },
                        )
            else:
                # Fallback to single document if no individual pages
                try:
                    document = await asyncio.wait_for(
                        self._process_content(
                            content=result.content,
                            url=url,
                            title=f"Documentation from {url}",
                            metadata=result.metadata,
                        ),
                        timeout=300,  # 5 minute timeout
                    )
                    documents = [document]

                    # Report progress for single document
                    if job_id and db:
                        await db.update_job_progress(
                            job_id,
                            0.8,
                            metadata={
                                "phase": "chunking_and_embedding",
                                "processed_pages": 1,
                                "total_pages": 1,
                            },
                        )

                except asyncio.TimeoutError:
                    logger.error(f"Processing timeout for main document: {url}")
                    return ProcessingResult(
                        documents=[], success=False, error="Document processing timeout"
                    )

            # Create result with success/failure information
            processing_result = ProcessingResult(documents=documents, success=True)

            # Add failure information to metadata if any pages failed
            if failed_pages:
                if not processing_result.documents:
                    # All pages failed
                    return ProcessingResult(
                        documents=[],
                        success=False,
                        error=f"All {len(failed_pages)} pages failed to process",
                    )
                else:
                    # Partial success - add failed pages to first document's metadata
                    processing_result.documents[0].metadata[
                        "failed_pages"
                    ] = failed_pages
                    logger.warning(
                        f"URL processing completed with {len(failed_pages)} failed pages out of {total_pages}"
                    )

            return processing_result

        except Exception as e:
            logger.error(f"Failed to process URL {url}: {e}")
            return ProcessingResult(documents=[], success=False, error=str(e))

    async def process_file(
        self,
        file_path: str,
        options: dict | None = None,
        job_id: str | None = None,
        db=None,
    ) -> ProcessingResult:
        """Process a file using the existing extraction pipeline."""
        try:
            logger.info(f"Processing file: {file_path}")

            path = Path(file_path)
            if not path.exists():
                return ProcessingResult(
                    documents=[], success=False, error=f"File not found: {file_path}"
                )

            # Use existing PDF extraction for PDFs
            if path.suffix.lower() == ".pdf":
                result = self.extractor.extract_from_pdf(file_path)

                if not result.success:
                    return ProcessingResult(
                        documents=[], success=False, error=result.error
                    )

                document = await self._process_content(
                    content=result.content,
                    url=f"file://{file_path}",
                    title=path.stem,
                    metadata=result.metadata,
                )

                return ProcessingResult(documents=[document], success=True)

            # For other file types, read as text
            elif path.suffix.lower() in [".txt", ".md", ".rst"]:
                content = path.read_text(encoding="utf-8")

                document = await self._process_content(
                    content=content,
                    url=f"file://{file_path}",
                    title=path.stem,
                    metadata={
                        "file_type": path.suffix,
                        "file_size": path.stat().st_size,
                    },
                )

                return ProcessingResult(documents=[document], success=True)

            else:
                return ProcessingResult(
                    documents=[],
                    success=False,
                    error=f"Unsupported file type: {path.suffix}",
                )

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return ProcessingResult(documents=[], success=False, error=str(e))

    async def _process_content(
        self, content: str, url: str, title: str, metadata: dict
    ) -> ProcessedDocument:
        """Process content into chunks with embeddings and extract code snippets."""
        try:
            # Extract code snippets before chunking
            (
                code_snippet_data,
                cleaned_content,
            ) = self.code_extractor.extract_code_snippets(content, url, title)

            # Split cleaned content into chunks (without code blocks)
            chunks = self.chunker.chunk_text(cleaned_content)

            # Create embeddings for each chunk
            processed_chunks = []

            # Process chunks in batches to avoid rate limits
            batch_size = 10
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]

                # Get embeddings for the batch (handle missing API key)
                try:
                    embeddings = await self.embedding_service.embed_batch(
                        [chunk.content for chunk in batch]
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to generate embeddings (missing API key?): {e}"
                    )
                    # Create dummy embeddings for testing
                    embeddings = [[0.0] * 1536 for _ in batch]

                # Create processed chunks
                for chunk, embedding in zip(batch, embeddings):
                    # Extract links from this chunk
                    chunk_link_data = self._extract_links_from_chunk(chunk.content)

                    processed_chunk = ProcessedChunk(
                        content=chunk.content,
                        embedding=embedding,
                        metadata={
                            **chunk.metadata,
                            **chunk_link_data,  # Add chunk-level link data
                            "source_url": url,
                            "source_title": title,
                        },
                        tokens=chunk.tokens,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                    )
                    processed_chunks.append(processed_chunk)

                # Small delay to be respectful to embedding API
                if i + batch_size < len(chunks):
                    await asyncio.sleep(0.1)

            logger.info(f"Processed {len(processed_chunks)} chunks for {title}")

            # Process code snippets with embeddings
            processed_code_snippets = []
            if code_snippet_data:
                # Get embeddings for code snippets in batches
                for i in range(0, len(code_snippet_data), batch_size):
                    batch = code_snippet_data[i : i + batch_size]

                    # Get embeddings for the batch
                    try:
                        embeddings = await self.embedding_service.embed_batch(
                            [snippet["content"] for snippet in batch]
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate embeddings for code snippets: {e}"
                        )
                        # Create dummy embeddings for testing
                        embeddings = [[0.0] * 1536 for _ in batch]

                    # Create processed code snippets
                    for snippet_data, embedding in zip(batch, embeddings):
                        processed_snippet = CodeSnippet(
                            content=snippet_data["content"],
                            language=snippet_data["language"],
                            embedding=embedding,
                            metadata={
                                **snippet_data,
                                "source_url": url,
                                "source_title": title,
                            },
                            start_line=snippet_data.get("start_line"),
                            end_line=snippet_data.get("end_line"),
                            char_start=snippet_data.get("char_start"),
                            char_end=snippet_data.get("char_end"),
                        )
                        processed_code_snippets.append(processed_snippet)

                    # Small delay to be respectful to embedding API
                    if i + batch_size < len(code_snippet_data):
                        await asyncio.sleep(0.1)

                logger.info(
                    f"Processed {len(processed_code_snippets)} code snippets for {title}"
                )

            return ProcessedDocument(
                url=url,
                title=title,
                content=content,
                chunks=processed_chunks,
                code_snippets=processed_code_snippets,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to process content for {title}: {e}")
            raise

    def _create_title_from_url(self, page_url: str, base_url: str) -> str:
        """Create a meaningful title from the page URL."""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(page_url)

            # Extract path parts and create a title
            path_parts = [part for part in parsed.path.strip("/").split("/") if part]

            if path_parts:
                # Use the last meaningful part of the path
                title_part = path_parts[-1].replace("-", " ").replace("_", " ").title()
                if title_part and title_part.lower() not in ["index", "home", "main"]:
                    return f"{title_part} - {parsed.netloc}"

            # Fallback to just the domain and path
            if parsed.path and parsed.path != "/":
                return f"{parsed.netloc}{parsed.path}"

            return f"Documentation from {page_url}"

        except Exception:
            return f"Documentation from {page_url}"
