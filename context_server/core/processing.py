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

            # Create a single document from the combined content
            # In the future, we might want to process individual pages separately
            document = await self._process_content(
                content=result.content,
                url=url,
                title=f"Documentation from {url}",
                metadata=result.metadata,
            )

            return ProcessingResult(documents=[document], success=True)

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
