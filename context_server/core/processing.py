"""Document processing service that integrates with existing extraction pipeline."""

import asyncio
import logging

# Import existing extraction functionality
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append("/app/src")  # Add src path for imports

from src.core.crawl4ai_extraction import Crawl4aiExtractor

from .chunking import TextChunker
from .content_analysis import ContentAnalyzer
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
class ProcessedDocument:
    """A processed document with chunks."""

    url: str
    title: str
    content: str
    chunks: list[ProcessedChunk]
    metadata: dict


@dataclass
class ProcessingResult:
    """Result of document processing operation."""

    documents: list[ProcessedDocument]
    success: bool
    error: str | None = None


class DocumentProcessor:
    """Processes documents using existing extraction pipeline and creates embeddings."""

    def __init__(self, embedding_service: EmbeddingService | None = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.chunker = TextChunker()
        self.extractor = Crawl4aiExtractor()
        self.content_analyzer = ContentAnalyzer()

    async def process_url(
        self, url: str, options: dict | None = None
    ) -> ProcessingResult:
        """Process a URL using the existing crawl4ai extraction pipeline."""
        try:
            logger.info(f"Processing URL: {url}")

            # Use existing extraction pipeline
            max_pages = options.get("max_pages", 50) if options else 50
            result = await self.extractor.extract_from_url(url, max_pages=max_pages)

            if not result.success:
                return ProcessingResult(documents=[], success=False, error=result.error)

            # Create separate documents for each extracted page if available
            documents = []
            extracted_pages = result.metadata.get("extracted_pages", [])

            if extracted_pages:
                # Process each page as a separate document
                for page_info in extracted_pages:
                    page_url = page_info["url"]
                    page_content = page_info["content"]

                    # Create page-specific metadata (exclude batch-level statistics)
                    page_metadata = self._create_page_metadata(
                        result.metadata, page_info
                    )
                    page_metadata["page_url"] = page_url
                    page_metadata["is_individual_page"] = True
                    page_metadata[
                        "extraction_success"
                    ] = True  # Individual page was successfully extracted

                    # Create document title from page URL
                    page_title = self._create_title_from_url(page_url, url)

                    document = await self._process_content(
                        content=page_content,
                        url=page_url,  # Use individual page URL
                        title=page_title,
                        metadata=page_metadata,
                    )
                    documents.append(document)
            else:
                # Fallback to single document if no individual pages
                document = await self._process_content(
                    content=result.content,
                    url=url,
                    title=f"Documentation from {url}",
                    metadata=result.metadata,
                )
                documents = [document]

            return ProcessingResult(documents=documents, success=True)

        except Exception as e:
            logger.error(f"Failed to process URL {url}: {e}")
            return ProcessingResult(documents=[], success=False, error=str(e))

    async def process_file(
        self, file_path: str, options: dict | None = None
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

    async def process_git_repo(
        self, repo_url: str, options: dict | None = None
    ) -> ProcessingResult:
        """Process a Git repository."""
        # TODO: Implement Git repository processing
        # This would involve cloning the repo and processing relevant files
        return ProcessingResult(
            documents=[],
            success=False,
            error="Git repository processing not yet implemented",
        )

    async def _process_content(
        self, content: str, url: str, title: str, metadata: dict
    ) -> ProcessedDocument:
        """Process content into chunks with embeddings."""
        try:
            # Split content into chunks
            chunks = self.chunker.chunk_text(content)

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
                    processed_chunk = ProcessedChunk(
                        content=chunk.content,
                        embedding=embedding,
                        metadata={
                            **chunk.metadata,
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

            return ProcessedDocument(
                url=url,
                title=title,
                content=content,
                chunks=processed_chunks,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to process content for {title}: {e}")
            raise

    def _create_page_metadata(self, batch_metadata: dict, page_info: dict) -> dict:
        """Create page-specific metadata excluding batch-level statistics."""
        # Define batch-level fields that should NOT be copied to individual pages
        batch_only_fields = {
            "total_links_found",
            "filtered_links",
            "successful_extractions",
            "extracted_pages",  # Don't include the full page list in each page
        }

        # Copy base metadata excluding batch statistics
        page_metadata = {
            key: value
            for key, value in batch_metadata.items()
            if key not in batch_only_fields
        }

        # Add page-specific information
        page_metadata.update(
            {
                "content_length": len(page_info.get("content", "")),
                "filename": page_info.get("filename"),
                "processing_time": batch_metadata.get(
                    "extraction_time"
                ),  # Individual processing time if available
            }
        )

        # Perform content analysis
        try:
            analysis = self.content_analyzer.analyze_content(
                page_info.get("content", "")
            )
            page_metadata.update(
                {
                    "content_type": analysis.content_type,
                    "primary_language": analysis.primary_language,
                    "summary": analysis.summary,
                    "code_percentage": analysis.code_percentage,
                    "detected_patterns": analysis.detected_patterns,
                    "key_concepts": analysis.key_concepts[:5],  # Limit to top 5
                    "api_references": analysis.api_references[:10],  # Limit to top 10
                    "code_blocks_count": len(analysis.code_blocks),
                }
            )

            # Store detailed code analysis if significant code content
            if analysis.code_percentage > 10:
                page_metadata["code_analysis"] = {
                    "functions": [
                        func
                        for block in analysis.code_blocks
                        for func in block.functions
                    ],
                    "classes": [
                        cls for block in analysis.code_blocks for cls in block.classes
                    ],
                    "imports": [
                        imp for block in analysis.code_blocks for imp in block.imports
                    ],
                    "languages": list(
                        set(block.language for block in analysis.code_blocks)
                    ),
                }
        except Exception as e:
            logger.warning(
                f"Content analysis failed for {page_info.get('url', 'unknown')}: {e}"
            )
            # Set basic defaults if analysis fails
            page_metadata.update(
                {
                    "content_type": "general",
                    "primary_language": None,
                    "summary": "Content analysis unavailable",
                    "code_percentage": 0.0,
                    "detected_patterns": {},
                    "key_concepts": [],
                    "api_references": [],
                    "code_blocks_count": 0,
                }
            )

        return page_metadata

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
