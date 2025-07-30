"""
Simplified crawl4ai-based document extraction system.

Uses proven optimal configuration for high-quality multi-page extraction
with minimal filtering to preserve content quality.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

# No typing imports needed for Python 3.12+
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

from context_server.core.text.cleaning import MarkdownCleaner
from context_server.core.services.extraction.utils import FileUtils, URLUtils

# Configure structured logging
logger = logging.getLogger(__name__)


class ExtractionResult:
    """Result of document extraction operation."""

    def __init__(
        self,
        success: bool = True,
        content: str = "",
        metadata: dict | None = None,
        error: str | None = None,
        extracted_pages: list | None = None,
    ):
        self.success = success
        self.content = content
        self.metadata = metadata or {}
        self.error = error
        self.extracted_pages = extracted_pages or []

    @classmethod
    def error(cls, message: str) -> "ExtractionResult":
        """Create error result."""
        return cls(success=False, error=message)


class Crawl4aiExtractor:
    """High-performance document extraction using crawl4ai with optimized configuration."""

    def __init__(self):
        """Initialize extractor with optimal configuration."""
        self.cleaner = MarkdownCleaner()

    async def extract_url(self, url: str, max_pages: int = 50) -> ExtractionResult:
        """Extract content from URL with multi-page crawling support."""
        try:
            # Normalize and validate URL
            normalized_url = URLUtils.normalize_url(url)
            domain = URLUtils.extract_domain(normalized_url)
            
            logger.info(f"Starting extraction from {normalized_url} (max_pages: {max_pages})")

            # Configure optimal deep crawling strategy for page discovery
            deep_crawl_strategy = BFSDeepCrawlStrategy(
                max_depth=3,              # Reasonable depth for documentation sites
                include_external=False,   # Stay within the same domain
                max_pages=max_pages      # Respect user's page limit
            )

            # Optimal configuration proven by testing (from old working code)
            crawler_config = CrawlerRunConfig(
                # Multi-page discovery
                deep_crawl_strategy=deep_crawl_strategy,
                # Basic cleanup only - NO aggressive content filtering
                word_count_threshold=10,                              # Remove tiny blocks
                excluded_tags=["nav", "footer", "header", "aside"],   # Remove navigation
                exclude_external_links=True,                         # Clean up links
                # Quality-focused settings
                page_timeout=30000,
                cache_mode="enabled",
                markdown_generator=None   # KEY: No content filtering for quality
            )

            # Execute extraction with crawler config (BFS strategy already included)
            async with AsyncWebCrawler(verbose=False, headless=True) as crawler:
                start_time = datetime.now()
                
                # Run simplified extraction (from old working code)
                results = await crawler.arun(url=normalized_url, config=crawler_config)

                # Handle both single result and list of results
                if not isinstance(results, list):
                    results = [results] if results.success else []
                
                logger.info(f"Crawl4ai found {len(results)} pages to process")
                
                extraction_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Extraction completed in {extraction_time:.2f}s")

                # Validate results
                if not results:
                    error_msg = "No pages extracted from crawl"
                    logger.error(error_msg)
                    return ExtractionResult.error(error_msg)

                # Process pages with simplified high-quality approach (from old working code)
                extracted_contents = []
                total_content_length = 0
                successful_pages = 0

                for i, result in enumerate(results[:max_pages]):
                    if not result.success or not result.markdown:
                        logger.debug(f"Skipping page {i+1}: No content extracted")
                        continue

                    # Always use raw_markdown for optimal quality (proven by testing)
                    content = result.markdown.raw_markdown or result.markdown
                    
                    # Basic content validation
                    if len(content.strip()) < 100:
                        logger.debug(f"Skipping {result.url} - content too short")
                        continue

                    # Optional light cleaning (preserve content quality)
                    cleaned_content = self.cleaner.clean_content(content)

                    extracted_contents.append({
                        "url": result.url,
                        "content": cleaned_content,
                        "length": len(cleaned_content),
                        "title": getattr(result, 'title', '') or f"Page {i+1}"
                    })

                    total_content_length += len(cleaned_content)
                    successful_pages += 1

                    logger.info(f"✓ Page {i+1}: {len(cleaned_content):,} chars from {result.url}")

                if not extracted_contents:
                    return ExtractionResult.error("No valid content extracted from any page")

                # Combine all content with clear separators (from old working code)
                combined_content = "\n\n---\n\n".join(
                    [item["content"] for item in extracted_contents]
                )

                # Create simple, clear metadata without large content arrays
                metadata = {
                    "source_type": "simplified_crawl4ai",  
                    "base_url": normalized_url,
                    "pages_found": len(results),
                    "pages_extracted": successful_pages,
                    "total_content_length": total_content_length,
                    "average_page_size": total_content_length // successful_pages if successful_pages else 0,
                    "extraction_time": datetime.now().isoformat(),
                    "page_urls": [page["url"] for page in extracted_contents],
                }

                logger.info(
                    f"Simplified extraction completed: {successful_pages}/{len(results)} pages, "
                    f"{total_content_length:,} total chars"
                )

                return ExtractionResult(
                    success=True, 
                    content=combined_content, 
                    metadata=metadata,
                    extracted_pages=extracted_contents
                )

        except Exception as e:
            error_msg = f"Extraction failed for {url}: {str(e)}"
            logger.error(error_msg)
            return ExtractionResult.error(error_msg)

    async def extract_file(self, file_path: str | Path) -> ExtractionResult:
        """Extract content from local file."""
        try:
            path = Path(file_path)
            
            if not path.exists():
                return ExtractionResult.error(f"File not found: {file_path}")
            
            if not path.is_file():
                return ExtractionResult.error(f"Path is not a file: {file_path}")
            
            # Read file content
            try:
                content = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                # Try with different encoding
                try:
                    content = path.read_text(encoding='latin-1')
                except Exception as e:
                    return ExtractionResult.error(f"Could not read file {file_path}: {e}")
            
            # Clean content if it's markdown
            if path.suffix.lower() in ['.md', '.markdown']:
                content = self.cleaner.clean_content(content)
            
            # Build metadata
            metadata = {
                'source_file': str(path.absolute()),
                'source_type': 'file',
                'file_size': path.stat().st_size,
                'file_extension': path.suffix,
                'title': FileUtils.create_safe_filename(path.stem),
                'extracted_at': datetime.now().isoformat(),
                'content_length': len(content),
            }
            
            logger.info(f"Successfully extracted {len(content)} characters from file: {path.name}")
            
            return ExtractionResult(
                success=True,
                content=content,
                metadata=metadata
            )
            
        except Exception as e:
            error_msg = f"File extraction failed for {file_path}: {str(e)}"
            logger.error(error_msg)
            return ExtractionResult.error(error_msg)


__all__ = ["Crawl4aiExtractor", "ExtractionResult"]