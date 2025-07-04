"""
Unified document processor system.

Consolidates and refactors processors from embed.py into a clean plugin architecture
following CLAUDE.md principles.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from git import Repo

from ..utils.pdf_spliter import split_pdf_vertically
from ..utils.segment_tables import extract_table_from_pdf
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessingOptions:
    """Configuration options for document processing."""

    # Processing flags
    extract_tables: bool = False
    split_vertical: bool = False

    # Output configuration
    output_dir: Path = Path("output")

    # Model parameters
    embedding_dim: int = 1536
    max_token_size: int = 8192

    # Chunking options
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class ProcessingResult:
    """Result of document processing operation."""

    success: bool
    content: str = ""
    chunks: List[str] = None
    metadata: Dict[str, Any] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []
        if self.metadata is None:
            self.metadata = {}

    @classmethod
    def error(cls, message: str) -> "ProcessingResult":
        """Create error result."""
        return cls(success=False, error=message)


class DocumentProcessor(ABC):
    """
    Abstract base class for document processors.

    Follows CLAUDE.md principles:
    - Abstract base class for extensibility
    - Clear interface contracts
    - Single responsibility per processor type
    """

    def __init__(self, options: Optional[ProcessingOptions] = None):
        self.options = options or ProcessingOptions()

    @abstractmethod
    def can_handle(self, source: Union[str, Path]) -> bool:
        """Determine if this processor can handle the given source."""
        pass

    @abstractmethod
    def get_source_type(self) -> str:
        """Return the source type identifier for this processor."""
        pass

    @abstractmethod
    def process(self, source: Union[str, Path]) -> ProcessingResult:
        """Process the source and return results."""
        pass


class PDFProcessor(DocumentProcessor):
    """
    PDF document processor with table extraction capabilities.

    Consolidates PDF processing logic from embed.py with enhanced
    configurability and cleaner interface.
    """

    def __init__(self, options: Optional[ProcessingOptions] = None):
        super().__init__(options)
        self.converter = DocumentConverter()

    def can_handle(self, source: Union[str, Path]) -> bool:
        """Check if source is a PDF file."""
        if isinstance(source, str):
            return source.lower().endswith(".pdf")
        return source.suffix.lower() == ".pdf"

    def get_source_type(self) -> str:
        """Return source type identifier."""
        return "pdf"

    def process(self, source: Union[str, Path]) -> ProcessingResult:
        """
        Process PDF file with optional table extraction and vertical splitting.

        Args:
            source: Path to PDF file

        Returns:
            ProcessingResult with content and metadata
        """
        try:
            file_path = Path(source)
            logger.info("Processing PDF", file_path=str(file_path))

            # Handle vertical splitting if requested
            if self.options.split_vertical:
                logger.info("Applying vertical PDF splitting")
                file_path = self._handle_vertical_split(file_path)

            # Extract tables if requested
            table_data = None
            if self.options.extract_tables:
                logger.info("Extracting tables from PDF")
                table_data = self._extract_tables(file_path)

            # Convert PDF to markdown
            result = self.converter.convert(file_path)
            content = result.document.export_to_markdown()

            # Create chunks using hybrid chunker
            chunks = self._create_chunks(content)

            metadata = {
                "source_type": self.get_source_type(),
                "file_path": str(file_path),
                "has_tables": table_data is not None,
                "table_count": len(table_data) if table_data else 0,
                "chunk_count": len(chunks),
                "split_vertical": self.options.split_vertical,
            }

            if table_data:
                metadata["tables"] = table_data

            return ProcessingResult(
                success=True, content=content, chunks=chunks, metadata=metadata
            )

        except Exception as e:
            logger.error("PDF processing failed", file_path=str(source), error=str(e))
            return ProcessingResult.error(f"Failed to process PDF: {e}")

    def _handle_vertical_split(self, file_path: Path) -> Path:
        """Handle vertical PDF splitting."""
        try:
            output_path = self.options.output_dir / f"split_{file_path.name}"
            split_pdf_vertically(str(file_path), str(output_path))
            return output_path
        except Exception as e:
            logger.warning("Vertical split failed, using original", error=str(e))
            return file_path

    def _extract_tables(self, file_path: Path) -> Optional[List[Dict]]:
        """Extract tables from PDF."""
        try:
            return extract_table_from_pdf(str(file_path))
        except Exception as e:
            logger.warning("Table extraction failed", error=str(e))
            return None

    def _create_chunks(self, content: str) -> List[str]:
        """Create text chunks from content."""
        try:
            chunker = HybridChunker(
                chunk_size=self.options.chunk_size,
                chunk_overlap=self.options.chunk_overlap,
            )
            chunks = chunker.chunk(content)
            return [chunk.text for chunk in chunks]
        except Exception as e:
            logger.warning("Chunking failed, using simple splitting", error=str(e))
            return self._simple_chunk(content)

    def _simple_chunk(self, content: str, max_size: int = 1000) -> List[str]:
        """Simple fallback chunking."""
        words = content.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            if current_size + len(word) > max_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1  # +1 for space

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


class URLProcessor(DocumentProcessor):
    """
    URL document processor for web content extraction.

    Simplified from embed.py to focus on core URL processing functionality.
    """

    def __init__(self, options: Optional[ProcessingOptions] = None):
        super().__init__(options)
        self.converter = DocumentConverter()

    def can_handle(self, source: Union[str, Path]) -> bool:
        """Check if source is a URL."""
        if isinstance(source, Path):
            return False
        return str(source).startswith(("http://", "https://"))

    def get_source_type(self) -> str:
        """Return source type identifier."""
        return "url"

    def process(self, source: Union[str, Path]) -> ProcessingResult:
        """
        Process URL content.

        Args:
            source: URL to process

        Returns:
            ProcessingResult with content and metadata
        """
        try:
            url = str(source)
            logger.info("Processing URL", url=url)

            # Convert URL to markdown
            result = self.converter.convert(url)
            content = result.document.export_to_markdown()

            # Create chunks
            chunks = self._create_chunks(content)

            metadata = {
                "source_type": self.get_source_type(),
                "url": url,
                "chunk_count": len(chunks),
            }

            return ProcessingResult(
                success=True, content=content, chunks=chunks, metadata=metadata
            )

        except Exception as e:
            logger.error("URL processing failed", url=str(source), error=str(e))
            return ProcessingResult.error(f"Failed to process URL: {e}")

    def _create_chunks(self, content: str) -> List[str]:
        """Create text chunks from content."""
        try:
            chunker = HybridChunker(
                chunk_size=self.options.chunk_size,
                chunk_overlap=self.options.chunk_overlap,
            )
            chunks = chunker.chunk(content)
            return [chunk.text for chunk in chunks]
        except Exception as e:
            logger.warning("Chunking failed, using simple splitting", error=str(e))
            # Simple fallback chunking
            lines = content.split("\n")
            chunks = []
            current_chunk = []
            current_size = 0

            for line in lines:
                if current_size + len(line) > self.options.chunk_size and current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = [line]
                    current_size = len(line)
                else:
                    current_chunk.append(line)
                    current_size += len(line)

            if current_chunk:
                chunks.append("\n".join(current_chunk))

            return chunks


class GitProcessor(DocumentProcessor):
    """
    Git repository processor for extracting documentation from repositories.

    Simplified and improved from embed.py version.
    """

    def __init__(self, options: Optional[ProcessingOptions] = None):
        super().__init__(options)

    def can_handle(self, source: Union[str, Path]) -> bool:
        """Check if source is a git repository URL or local repo."""
        if isinstance(source, Path):
            return (source / ".git").exists()

        source_str = str(source)
        return source_str.startswith(
            ("git://", "https://github.com", "git@github.com")
        ) or source_str.endswith(".git")

    def get_source_type(self) -> str:
        """Return source type identifier."""
        return "git"

    def process(self, source: Union[str, Path]) -> ProcessingResult:
        """
        Process git repository content.

        Args:
            source: Git repository URL or local path

        Returns:
            ProcessingResult with content and metadata
        """
        try:
            logger.info("Processing git repository", source=str(source))

            # Clone or use existing repository
            repo_path = self._prepare_repository(source)

            # Extract documentation files
            doc_files = self._find_documentation_files(repo_path)

            if not doc_files:
                return ProcessingResult.error(
                    "No documentation files found in repository"
                )

            # Process all documentation files
            combined_content = []
            total_chunks = []

            for doc_file in doc_files:
                try:
                    content = doc_file.read_text(encoding="utf-8")
                    combined_content.append(
                        f"# {doc_file.relative_to(repo_path)}\n\n{content}"
                    )

                    # Create chunks for this file
                    file_chunks = self._create_chunks(content)
                    total_chunks.extend(file_chunks)

                except Exception as e:
                    logger.warning(
                        "Failed to process file", file_path=str(doc_file), error=str(e)
                    )

            content = "\n\n---\n\n".join(combined_content)

            metadata = {
                "source_type": self.get_source_type(),
                "repository": str(source),
                "repo_path": str(repo_path),
                "doc_files_count": len(doc_files),
                "chunk_count": len(total_chunks),
                "doc_files": [str(f.relative_to(repo_path)) for f in doc_files],
            }

            return ProcessingResult(
                success=True, content=content, chunks=total_chunks, metadata=metadata
            )

        except Exception as e:
            logger.error("Git processing failed", source=str(source), error=str(e))
            return ProcessingResult.error(f"Failed to process git repository: {e}")

    def _prepare_repository(self, source: Union[str, Path]) -> Path:
        """Clone repository or return existing path."""
        if isinstance(source, Path) and source.exists():
            return source

        # Clone repository
        repo_name = str(source).split("/")[-1].replace(".git", "")
        clone_path = self.options.output_dir / f"repos/{repo_name}"

        if clone_path.exists():
            logger.info("Using existing repository clone", path=str(clone_path))
            return clone_path

        logger.info("Cloning repository", url=str(source), target=str(clone_path))
        clone_path.parent.mkdir(parents=True, exist_ok=True)
        Repo.clone_from(str(source), clone_path)

        return clone_path

    def _find_documentation_files(self, repo_path: Path) -> List[Path]:
        """Find documentation files in repository."""
        doc_patterns = [
            "*.md",
            "*.rst",
            "*.txt",
            "README*",
            "CHANGELOG*",
            "CONTRIBUTING*",
            "docs/**/*.md",
            "doc/**/*.md",
            "documentation/**/*.md",
        ]

        doc_files = []
        for pattern in doc_patterns:
            doc_files.extend(repo_path.glob(pattern))

        # Remove duplicates and sort
        unique_files = list(set(doc_files))
        unique_files.sort()

        logger.info("Found documentation files", count=len(unique_files))
        return unique_files

    def _create_chunks(self, content: str) -> List[str]:
        """Create text chunks from content."""
        # Simple line-based chunking for text files
        lines = content.split("\n")
        chunks = []
        current_chunk = []
        current_size = 0

        for line in lines:
            if current_size + len(line) > self.options.chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_size = len(line)
            else:
                current_chunk.append(line)
                current_size += len(line)

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks


class ProcessorFactory:
    """
    Factory for creating appropriate document processors.

    Implements factory pattern following CLAUDE.md principles for
    object creation and extensibility.
    """

    def __init__(self, options: Optional[ProcessingOptions] = None):
        self.options = options or ProcessingOptions()
        self.processors = [
            PDFProcessor(self.options),
            URLProcessor(self.options),
            GitProcessor(self.options),
        ]

    def get_processor(self, source: Union[str, Path]) -> Optional[DocumentProcessor]:
        """
        Get appropriate processor for the given source.

        Args:
            source: Source to process (URL, file path, etc.)

        Returns:
            Appropriate DocumentProcessor or None if no processor can handle the source
        """
        for processor in self.processors:
            if processor.can_handle(source):
                logger.debug(
                    "Selected processor",
                    processor_type=processor.get_source_type(),
                    source=str(source),
                )
                return processor

        logger.warning("No processor found for source", source=str(source))
        return None

    def get_supported_types(self) -> List[str]:
        """Get list of supported source types."""
        return [processor.get_source_type() for processor in self.processors]
