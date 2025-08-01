"""Context Server core module - Pipeline-centric architecture.

The core module is organized around the main pipeline coordinator with supporting services:

- pipeline.py: Main document processing coordinator (TOP LEVEL)
- database/: PostgreSQL operations and storage management
- services/: External API integrations (embeddings, extraction, LLM)
- text/: Text processing utilities (chunking, cleaning)
"""

# Database manager - primary storage interface
from context_server.core.database import DatabaseManager

# Main pipeline coordinator - the heart of the system
from context_server.core.pipeline import DocumentProcessor

# Service integrations
from context_server.core.services import *

# Text processing utilities
from context_server.core.text import *

__all__ = [
    # Main coordinator
    "DocumentProcessor",
    # Database operations
    "DatabaseManager",
    # Service integrations
    "EmbeddingService",
    "VoyageEmbeddingService",
    "Crawl4aiExtractor",
    "ExtractionResult",
    "FileUtils",
    "URLUtils",
    "SummarizationService",
    # Text processing
    "TextChunker",
    "TextChunk",
    "MarkdownCleaner",
]
