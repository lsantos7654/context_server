"""Document processing service that integrates with existing extraction pipeline."""

import asyncio
import logging
import re

# Import existing extraction functionality
from dataclasses import dataclass
from pathlib import Path

from .chunking import TextChunker
from .crawl4ai_extraction import Crawl4aiExtractor
from .embeddings import EmbeddingService, VoyageEmbeddingService
from .summarization import SummarizationService

logger = logging.getLogger(__name__)


@dataclass
class ProcessedChunk:
    """A processed text chunk with embedding and line tracking."""

    content: str
    embedding: list[float]
    metadata: dict
    tokens: int
    summary: str = None
    summary_model: str = None
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
        # Enhanced regex patterns for different code block formats
        self.markdown_code_pattern = re.compile(
            r"```(\w*)\n(.*?)\n```", re.DOTALL | re.MULTILINE
        )
        self.inline_code_pattern = re.compile(r"`([^`\n]+)`")
        
        # Additional patterns for various code block formats
        self.html_code_pattern = re.compile(
            r"<(?:pre|code)[^>]*>(.*?)</(?:pre|code)>", re.DOTALL | re.IGNORECASE
        )
        self.indented_code_pattern = re.compile(
            r"^(?: {4}|\t)(.+)$", re.MULTILINE
        )
        # Pattern for common code indicators followed by blocks
        self.code_indicator_pattern = re.compile(
            r"(?:Example|Code|Sample|Usage):\s*\n\n((?:(?:    |\t).*\n?)+)", re.MULTILINE
        )

    def extract_code_snippets(
        self, content: str, url: str = "", title: str = ""
    ) -> tuple[list[dict], str]:
        """
        Extract code snippets from content and return cleaned content with inline placeholders.

        Returns:
            Tuple of (code_snippets_list, content_with_inline_placeholders)
        """
        import uuid
        snippets = []
        lines = content.splitlines()

        # Track character positions
        char_pos = 0
        line_char_map = []
        for line in lines:
            line_char_map.append(char_pos)
            char_pos += len(line) + 1  # +1 for newline

        # Track replacements to apply them in reverse order (to maintain positions)
        replacements = []

        # Extract markdown code blocks (filter out very short ones)
        for match in self.markdown_code_pattern.finditer(content):
            language = match.group(1) or "text"
            code_content = match.group(2).strip()
            
            # Skip very short code blocks (likely just noise)
            if len(code_content) < 20:
                continue
                
            start_char = match.start()
            end_char = match.end()

            # Find line numbers
            start_line = self._char_to_line(start_char, line_char_map)
            end_line = self._char_to_line(end_char, line_char_map)

            # Generate unique snippet ID for this instance
            snippet_id = str(uuid.uuid4())

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
                "snippet_id": snippet_id,  # Add the unique ID
            }
            snippets.append(snippet)

            # Create placeholder for inline replacement
            placeholder = f"[CODE_SNIPPET: language={language}, size={len(code_content)}chars, snippet_id={snippet_id}]"
            replacements.append((start_char, end_char, placeholder))

        # Extract and handle HTML code blocks
        for match in self.html_code_pattern.finditer(content):
            code_content = match.group(1).strip()
            if len(code_content) > 20:  # Only substantial code blocks
                start_char = match.start()
                end_char = match.end()
                start_line = self._char_to_line(start_char, line_char_map)
                end_line = self._char_to_line(end_char, line_char_map)

                # Generate unique snippet ID for this instance
                snippet_id = str(uuid.uuid4())

                snippet = {
                    "content": code_content,
                    "language": "html",  # HTML-embedded code
                    "start_line": start_line,
                    "end_line": end_line,
                    "char_start": start_char,
                    "char_end": end_char,
                    "type": "html_code_block",
                    "source_url": url,
                    "source_title": title,
                    "snippet_id": snippet_id,
                }
                snippets.append(snippet)

                # Create placeholder for inline replacement
                placeholder = f"[CODE_SNIPPET: language=html, size={len(code_content)}chars, snippet_id={snippet_id}]"
                replacements.append((start_char, end_char, placeholder))

        # Extract significant inline code (longer than 100 chars to reduce noise)
        for match in self.inline_code_pattern.finditer(content):
            code_content = match.group(1)
            # Filter out short inline code snippets that are just noise
            if len(code_content) > 100 and '\n' not in code_content:  # Only substantial single-line inline code
                start_char = match.start()
                end_char = match.end()
                start_line = self._char_to_line(start_char, line_char_map)
                end_line = start_line

                # Generate unique snippet ID for this instance
                snippet_id = str(uuid.uuid4())

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
                    "snippet_id": snippet_id,
                }
                snippets.append(snippet)

                # Create placeholder for inline replacement
                placeholder = f"[CODE_SNIPPET: language=text, size={len(code_content)}chars, snippet_id={snippet_id}]"
                replacements.append((start_char, end_char, placeholder))

        # Apply all replacements in reverse order to maintain character positions
        cleaned_content = content
        for start_char, end_char, placeholder in reversed(sorted(replacements)):
            cleaned_content = cleaned_content[:start_char] + placeholder + cleaned_content[end_char:]

        # Note: Skipping indented code blocks for now as they're harder to position accurately
        # They can be handled in a future enhancement if needed

        # Additional cleaning: remove extra whitespace and empty lines left by code removal
        cleaned_content = self._clean_whitespace_artifacts(cleaned_content)

        return snippets, cleaned_content

    def _char_to_line(self, char_pos: int, line_char_map: list[int]) -> int:
        """Convert character position to line number."""
        for i, line_start in enumerate(line_char_map):
            if char_pos < line_start + (
                len(line_char_map) > i + 1 and line_char_map[i + 1] - line_start or 0
            ):
                return i
        return len(line_char_map) - 1

    def _clean_whitespace_artifacts(self, content: str) -> str:
        """Clean up whitespace artifacts left by code block removal."""
        # Remove multiple consecutive empty lines
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Remove trailing whitespace from lines
        content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
        
        # Remove lines that are just whitespace
        content = re.sub(r'^\s*$\n', '', content, flags=re.MULTILINE)
        
        # Clean up markdown artifacts left by code removal
        content = re.sub(r'^\s*```\s*$', '', content, flags=re.MULTILINE)  # Orphaned code fence
        content = re.sub(r'^\s*\*\s*$', '', content, flags=re.MULTILINE)   # Orphaned bullet points
        
        return content.strip()


class DocumentProcessor:
    """Processes documents using existing extraction pipeline and creates embeddings."""

    def __init__(self, 
                 embedding_service: EmbeddingService | None = None, 
                 code_embedding_service: VoyageEmbeddingService | None = None,
                 summarization_service: SummarizationService | None = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.code_embedding_service = code_embedding_service or VoyageEmbeddingService()
        self.summarization_service = summarization_service or SummarizationService()
        self.text_chunker = TextChunker(chunk_size=2500, chunk_overlap=500, chunk_type="text")
        self.code_chunker = TextChunker(chunk_size=1800, chunk_overlap=360, chunk_type="code")
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
                        processed_documents = await asyncio.wait_for(
                            self._process_content(
                                content=page_content,
                                url=page_url,
                                title=page_title,
                                metadata=page_metadata,
                            ),
                            timeout=processing_timeout,
                        )
                        documents.extend(processed_documents)  # Now extends with list of 3 documents

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
                    processed_documents = await asyncio.wait_for(
                        self._process_content(
                            content=result.content,
                            url=url,
                            title=f"Documentation from {url}",
                            metadata=result.metadata,
                        ),
                        timeout=300,  # 5 minute timeout
                    )
                    documents = processed_documents

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

            # For supported file types, read as text
            if path.suffix.lower() in [".txt", ".md", ".rst"]:
                content = path.read_text(encoding="utf-8")

                processed_documents = await self._process_content(
                    content=content,
                    url=f"file://{file_path}",
                    title=path.stem,
                    metadata={
                        "file_type": path.suffix,
                        "file_size": path.stat().st_size,
                    },
                )

                return ProcessingResult(documents=processed_documents, success=True)

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
    ) -> list[ProcessedDocument]:
        """Process content into three documents: original, code snippets, and cleaned markdown."""
        try:
            # Extract code snippets before chunking
            (
                code_snippet_data,
                cleaned_content,
            ) = self.code_extractor.extract_code_snippets(content, url, title)

            # Document 1: Original markdown (raw content)
            original_chunks = self.text_chunker.chunk_text(content)
            original_processed_chunks = await self._process_chunks_with_embeddings(
                original_chunks, url, title, self.embedding_service, is_code=False
            )

            # Document 2: Code snippets only
            code_snippets_content = ""
            if code_snippet_data:
                # Combine all code snippets into a single document
                code_snippets_content = "\n\n".join([
                    f"```{snippet['language']}\n{snippet['content']}\n```"
                    for snippet in code_snippet_data
                ])

            code_processed_chunks = []
            code_processed_snippets = []
            
            if code_snippets_content:
                # Chunk the code snippets document with code embeddings for chunks table
                code_chunks = self.code_chunker.chunk_text(code_snippets_content)
                code_processed_chunks = await self._process_chunks_with_embeddings(
                    code_chunks, url, title, self.code_embedding_service, is_code=True
                )

                # Process individual code snippets with code embeddings for code_snippets table
                code_processed_snippets = await self._process_code_snippets_with_embeddings(
                    code_snippet_data, url, title
                )

            # Document 3: Cleaned markdown with code snippet placeholders
            cleaned_with_placeholders = self._create_cleaned_markdown_with_placeholders(
                cleaned_content, code_snippet_data
            )
            cleaned_chunks = self.text_chunker.chunk_text(cleaned_with_placeholders)
            cleaned_processed_chunks = await self._process_chunks_with_embeddings(
                cleaned_chunks, url, title, self.embedding_service, is_code=False
            )

            # Create three ProcessedDocument objects
            documents = []

            # 1. Original document
            documents.append(ProcessedDocument(
                url=url,
                title=f"{title} (Original)",
                content=content,
                chunks=original_processed_chunks,
                code_snippets=[],
                metadata={**metadata, "document_type": "original"},
            ))

            # 2. Code snippets document
            documents.append(ProcessedDocument(
                url=url,
                title=f"{title} (Code Snippets)",
                content=code_snippets_content,
                chunks=code_processed_chunks,  # Chunked code content with text embeddings
                code_snippets=code_processed_snippets,  # Individual code snippets with code embeddings
                metadata={**metadata, "document_type": "code_snippets"},
            ))

            # 3. Cleaned markdown document
            documents.append(ProcessedDocument(
                url=url,
                title=f"{title} (Cleaned)",
                content=cleaned_with_placeholders,
                chunks=cleaned_processed_chunks,
                code_snippets=[],
                metadata={**metadata, "document_type": "cleaned_markdown"},
            ))

            logger.info(f"Processed content into 3 documents for {title}")
            return documents

        except Exception as e:
            logger.error(f"Failed to process content for {title}: {e}")
            raise

    async def _process_chunks_with_embeddings(
        self, chunks: list, url: str, title: str, embedding_service, is_code: bool = False
    ) -> list[ProcessedChunk]:
        """Process chunks with embeddings and summaries."""
        processed_chunks = []
        batch_size = 10

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Get embeddings for the batch
            try:
                embeddings = await embedding_service.embed_batch(
                    [chunk.content for chunk in batch]
                )
            except Exception as e:
                logger.warning(
                    f"Failed to generate embeddings (missing API key?): {e}"
                )
                # Create dummy embeddings for testing
                embedding_dim = embedding_service.get_dimension()
                embeddings = [[0.0] * embedding_dim for _ in batch]

            # Generate summaries for the batch
            summaries = []
            summary_model = None
            if self.summarization_service.client:
                try:
                    chunk_data = [(f"chunk_{i+j}", chunk.content) for j, chunk in enumerate(batch)]
                    summary_results = await self.summarization_service.summarize_chunks_batch(
                        chunk_data, batch_size=5
                    )
                    summaries = [result[1] for result in summary_results]
                    summary_model = self.summarization_service.model
                    logger.debug(f"Generated {len([s for s in summaries if s])} summaries for batch")
                except Exception as e:
                    logger.warning(f"Failed to generate summaries: {e}")
                    summaries = [None] * len(batch)
            else:
                logger.debug("Summarization service not available, using fallback")
                summaries = [
                    self.summarization_service.get_fallback_summary(chunk.content)
                    for chunk in batch
                ]

            # Create processed chunks
            for chunk, embedding, summary in zip(batch, embeddings, summaries):
                # Extract links from this chunk
                chunk_link_data = self._extract_links_from_chunk(chunk.content)

                processed_chunk = ProcessedChunk(
                    content=chunk.content,
                    embedding=embedding,
                    summary=summary,
                    summary_model=summary_model,
                    metadata={
                        **chunk.metadata,
                        **chunk_link_data,
                        "source_url": url,
                        "source_title": title,
                        "is_code": is_code,
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

        return processed_chunks

    async def _process_code_snippets_with_embeddings(
        self, code_snippet_data: list, url: str, title: str
    ) -> list[CodeSnippet]:
        """Process code snippets with embeddings."""
        processed_code_snippets = []
        batch_size = 10

        for i in range(0, len(code_snippet_data), batch_size):
            batch = code_snippet_data[i : i + batch_size]

            # Get embeddings for the batch using code embedding service
            try:
                embeddings = await self.code_embedding_service.embed_batch(
                    [snippet["content"] for snippet in batch]
                )
            except Exception as e:
                logger.warning(
                    f"Failed to generate code embeddings: {e}"
                )
                # Create dummy embeddings for testing
                embedding_dim = self.code_embedding_service.get_dimension()
                embeddings = [[0.0] * embedding_dim for _ in batch]

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

        return processed_code_snippets

    def _create_cleaned_markdown_with_placeholders(
        self, cleaned_content: str, code_snippet_data: list
    ) -> str:
        """Create cleaned markdown with enhanced code snippet placeholders using storage layer summaries."""
        # If no code snippets, return cleaned content as-is
        if not code_snippet_data:
            return cleaned_content
        
        # Import storage manager to use consistent summary generation
        from ..core.storage import DatabaseManager
        
        # Create a temporary database manager instance for summary generation
        # This ensures we use the same logic as search results
        db_manager = DatabaseManager()
        
        # Process each snippet to add enhanced summaries to placeholders
        # The cleaned_content already has inline placeholders from extract_code_snippets()
        # We need to update those placeholders with proper summaries
        
        for snippet in code_snippet_data:
            snippet_id = snippet.get("snippet_id")
            if not snippet_id:
                continue  # Skip if no snippet_id (shouldn't happen with new logic)
            
            # Generate summary using the same logic as the storage layer
            summary = db_manager._generate_code_summary(snippet)
            
            # Find and replace the placeholder in cleaned_content
            old_placeholder = f"[CODE_SNIPPET: language={snippet.get('language', 'text')}, size={len(snippet.get('content', ''))}chars, snippet_id={snippet_id}]"
            new_placeholder = f"[CODE_SNIPPET: language={snippet.get('language', 'text')}, size={len(snippet.get('content', ''))}chars, summary=\"{summary}\", snippet_id={snippet_id}]"
            
            cleaned_content = cleaned_content.replace(old_placeholder, new_placeholder)
        
        return cleaned_content

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
