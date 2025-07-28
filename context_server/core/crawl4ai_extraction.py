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

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

from .cleaning import MarkdownCleaner
from .utils import FileUtils, URLUtils

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
    """
    Simplified extractor using crawl4ai with optimal configuration.

    Uses proven multi-page crawling approach with minimal filtering
    to preserve content quality while discovering related pages.
    """

    def __init__(self, output_dir: Path | None = None, config: dict | None = None):
        """Initialize extractor with optional output directory and configuration."""
        self.output_dir = FileUtils.ensure_directory(output_dir or Path("output"))

        # Initialize markdown cleaner for basic content processing
        self.cleaner = MarkdownCleaner()

        # Simple configuration - focus on quality over aggressive filtering
        self.config = config or {}
        
        # Extraction mode: "quality" (default) or "legacy" for backward compatibility
        self.extraction_mode = self.config.get("extraction_mode", "quality")

    async def extract_from_url(self, url: str, max_pages: int = 50) -> ExtractionResult:
        """Extract content from URL using simplified high-quality approach."""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Starting simplified crawl4ai extraction (mode: {self.extraction_mode})",
                    extra={"url": url, "max_pages": max_pages, "attempt": attempt + 1},
                )

                async with AsyncWebCrawler(verbose=False, headless=True) as crawler:
                    # Configure optimal deep crawling strategy for page discovery
                    deep_crawl_strategy = BFSDeepCrawlStrategy(
                        max_depth=3,              # Reasonable depth for documentation sites
                        include_external=False,   # Stay within the same domain
                        max_pages=max_pages      # Respect user's page limit
                    )

                    # Optimal configuration proven by testing
                    config = CrawlerRunConfig(
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

                    # Run simplified extraction
                    results = await crawler.arun(url=url, config=config)

                    # Handle both single result and list of results
                    if not isinstance(results, list):
                        results = [results] if results.success else []
                    
                    logger.info(f"Crawl4ai found {len(results)} pages to process")

                    # Validate results
                    if not results:
                        error_msg = "No pages extracted from crawl"
                        if attempt < max_retries - 1:
                            logger.warning(f"Attempt {attempt + 1} failed: {error_msg}. Retrying...")
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            return ExtractionResult.error(error_msg)

                    # Process pages with simplified high-quality approach
                    extracted_contents = []
                    total_content_length = 0
                    successful_pages = 0

                    for i, result in enumerate(results[:max_pages]):
                        if not result.success or not result.markdown:
                            logger.debug(f"Skipping page {i+1}: No content extracted")
                            continue

                        # Always use raw_markdown for optimal quality (proven by testing)
                        content = result.markdown.raw_markdown or result.markdown
                        
                        # Check for error content (the root issue from conversation)
                        if self._contains_error_content(content):
                            logger.warning(f"Error content detected in {result.url}, skipping")
                            continue

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

                        # Save individual files for debugging
                        if self.output_dir:
                            filename = self._create_filename_from_url(result.url)
                            file_path = self.output_dir / filename
                            file_path.write_text(cleaned_content, encoding="utf-8")
                            logger.debug(f"Saved {file_path}")

                        logger.info(f"✓ Page {i+1}: {len(cleaned_content):,} chars from {result.url}")

                    if not extracted_contents:
                        error_msg = "No valid content extracted from any page"
                        if attempt < max_retries - 1:
                            logger.warning(f"Attempt {attempt + 1} failed: {error_msg}. Retrying...")
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            return ExtractionResult.error(error_msg)

                    # Combine all content with clear separators
                    combined_content = "\n\n---\n\n".join(
                        [item["content"] for item in extracted_contents]
                    )

                    # Create simple, clear metadata without large content arrays
                    metadata = {
                        "source_type": "simplified_crawl4ai",
                        "extraction_mode": self.extraction_mode,
                        "base_url": url,
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
                error_msg = f"Extraction failed: {e}"
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {error_msg}. Retrying...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Extraction failed after {max_retries} attempts: {e}")
                    return ExtractionResult.error(
                        f"Extraction failed after {max_retries} attempts: {e}"
                    )

    def _contains_error_content(self, content: str) -> bool:
        """
        Check if content contains crawl4ai error messages.
        
        This addresses the root issue identified in the conversation summary.
        """
        error_indicators = [
            "Crawl4AI Error:",
            "Invalid expression", 
            "This page is not fully supported"
        ]
        return any(indicator in content for indicator in error_indicators)

    def _create_filename_from_url(self, url: str) -> str:
        """Create safe filename from URL."""
        parsed = urlparse(url)

        # Use path for filename, fallback to domain
        path_part = parsed.path.strip("/") or parsed.netloc

        # Use FileUtils for safe filename creation
        filename = FileUtils.create_safe_filename(path_part)

        # Ensure it ends with .md
        if not filename.endswith(".md"):
            filename += ".md"

        # Handle edge cases
        if filename == ".md" or filename == "_.md":
            filename = "index.md"

        return filename
