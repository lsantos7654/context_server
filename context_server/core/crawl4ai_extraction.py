"""
Simple crawl4ai-based document extraction system.

Replaces the complex tiered extraction system with a single, unified approach
using crawl4ai for all URL-based extraction.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

# No typing imports needed for Python 3.12+
from urllib.parse import urlparse

from crawl4ai import (
    AsyncWebCrawler,
    BM25ContentFilter,
    CacheMode,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    PruningContentFilter,
)
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
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
    ):
        self.success = success
        self.content = content
        self.metadata = metadata or {}
        self.error = error

    @classmethod
    def error(cls, message: str) -> "ExtractionResult":
        """Create error result."""
        return cls(success=False, error=message)


class Crawl4aiExtractor:
    """
    Simple extractor using crawl4ai for all URL-based extraction.

    Replaces the complex tiered system with a unified approach that handles:
    - Static sites with sitemaps
    - JavaScript-rendered documentation sites
    - Modern SPA frameworks
    - Navigation-based discovery
    """

    def __init__(self, output_dir: Path | None = None, config: dict | None = None):
        """Initialize extractor with optional output directory and configuration."""
        self.output_dir = FileUtils.ensure_directory(output_dir or Path("output"))

        # Initialize markdown cleaner for enhanced content processing
        self.cleaner = MarkdownCleaner()

        # Configuration for crawl4ai built-in filtering
        self.config = config or {}
        self.filtering_config = self.config.get("filtering", {})

        # Content filtering strategy
        self.content_filter_type = self.filtering_config.get("content_filter", "bm25")

    async def extract_from_url(self, url: str, max_pages: int = 50) -> ExtractionResult:
        """Extract content from URL using crawl4ai's built-in deep crawling and filtering."""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                logger.info(
                    "Starting crawl4ai deep extraction with built-in filtering",
                    extra={"url": url, "max_pages": max_pages, "attempt": attempt + 1},
                )

                async with AsyncWebCrawler(verbose=False, headless=True) as crawler:
                    # Create content filter optimized for documentation cleaning
                    content_filter = PruningContentFilter(
                        threshold=0.45,  # Dynamic threshold for adaptive content filtering
                        threshold_type="dynamic",
                        min_word_threshold=5,  # Ignore very short text blocks
                    )

                    # Configure deep crawling strategy
                    deep_crawl_strategy = BFSDeepCrawlStrategy(
                        max_depth=4,  # Increased depth to reach concept pages (/ -> /concepts/ -> /concepts/event-handling/)
                        include_external=False,
                        max_pages=max_pages  # Use the passed max_pages parameter
                        # Note: Removed score_threshold as it was filtering out too many valid links
                    )

                    # Enhanced configuration with documentation-optimized filtering
                    config = CrawlerRunConfig(
                        # Deep crawling configuration
                        deep_crawl_strategy=deep_crawl_strategy,
                        # Content filtering optimized for documentation sites
                        excluded_tags=[
                            "nav", "footer", "header", "sidebar",  # Navigation elements
                            ".breadcrumb", ".pagination", ".toc",  # Navigation helpers
                            ".version-list", ".highlights", ".releases",  # Version/release content
                            ".navigation", ".menu", ".nav-menu",  # More navigation variants
                        ],
                        word_count_threshold=10,  # Filter out very short content blocks
                        exclude_external_links=True,
                        # Enhanced markdown generation with dynamic content filtering
                        markdown_generator=DefaultMarkdownGenerator(
                            content_filter=content_filter
                        ),
                        # Basic settings
                        magic=True,  # Enable magic mode for better JS handling
                        page_timeout=30000,
                        verbose=False,
                        cache_mode=CacheMode.ENABLED,
                    )

                    # Run deep crawl with built-in filtering
                    results = await crawler.arun(url=url, config=config)

                    # Debug: Log what we got back
                    logger.info(f"DEBUG: crawl4ai returned type: {type(results)}")
                    if isinstance(results, list):
                        logger.info(f"DEBUG: List with {len(results)} items")
                    else:
                        logger.info(
                            f"DEBUG: Single result, success: {results.success if hasattr(results, 'success') else 'unknown'}"
                        )
                        if hasattr(results, "links"):
                            logger.info(
                                f"DEBUG: Links found: internal={len(results.links.get('internal', []))} external={len(results.links.get('external', []))}"
                            )

                    # Handle results from deep crawling (can be single result or list)
                    if isinstance(results, list):
                        if not results:
                            error_msg = "Deep crawl returned no results"
                            if attempt < max_retries - 1:
                                logger.warning(
                                    f"Attempt {attempt + 1} failed: {error_msg}. Retrying..."
                                )
                                await asyncio.sleep(retry_delay)
                                continue
                            else:
                                return ExtractionResult.error(error_msg)

                        # Process multiple results from deep crawl
                        extracted_contents = []
                        successful_pages = 0

                        logger.info(f"Deep crawl found {len(results)} pages")

                        for result in results[:max_pages]:  # Respect max_pages limit
                            if result.success and result.markdown:
                                # Prioritize fit_markdown for cleaner content
                                original_content = result.markdown
                                filtered_content = (
                                    result.fit_markdown
                                    if hasattr(result, "fit_markdown")
                                    and result.fit_markdown
                                    else result.markdown
                                )

                                # Log filtering effectiveness
                                if (
                                    hasattr(result, "fit_markdown")
                                    and result.fit_markdown
                                ):
                                    logger.info(f"Using filtered content for {result.url}")
                                    content = result.fit_markdown
                                else:
                                    logger.info(
                                        f"No fit_markdown available for {result.url}, using original"
                                    )
                                    content = result.markdown

                                # Clean content with existing cleaner for additional processing
                                cleaned_content = self.cleaner.clean_content(content)

                                if (
                                    len(cleaned_content.strip()) < 100
                                ):  # Skip tiny pages
                                    logger.debug(
                                        f"Skipping {result.url} - content too short"
                                    )
                                    continue

                                extracted_contents.append(
                                    {
                                        "url": result.url,
                                        "content": cleaned_content,
                                        "depth": result.metadata.get("depth", 0),
                                        "original_length": len(result.markdown)
                                        if result.markdown
                                        else 0,
                                        "filtered_length": len(cleaned_content),
                                    }
                                )

                                successful_pages += 1

                                # Save individual files
                                if self.output_dir:
                                    filename = self._create_filename_from_url(
                                        result.url
                                    )
                                    file_path = self.output_dir / filename
                                    file_path.write_text(
                                        cleaned_content, encoding="utf-8"
                                    )
                                    logger.debug(f"Saved {file_path}")

                        if not extracted_contents:
                            error_msg = "No valid content extracted from deep crawl"
                            if attempt < max_retries - 1:
                                logger.warning(
                                    f"Attempt {attempt + 1} failed: {error_msg}. Retrying..."
                                )
                                await asyncio.sleep(retry_delay)
                                continue
                            else:
                                return ExtractionResult.error(error_msg)

                        # Combine all content
                        combined_content = "\\n\\n---\\n\\n".join(
                            [item["content"] for item in extracted_contents]
                        )

                        # Create minimal metadata
                        metadata = {
                            "source_type": "crawl4ai_deep",
                            "base_url": url,
                            "extracted_pages": extracted_contents,
                        }

                    else:
                        # Handle single result (fallback)
                        result = results

                        if not result.success:
                            error_msg = f"Deep crawl failed: {result.error}"
                            if attempt < max_retries - 1:
                                logger.warning(
                                    f"Attempt {attempt + 1} failed: {error_msg}. Retrying..."
                                )
                                await asyncio.sleep(retry_delay)
                                continue
                            else:
                                return ExtractionResult.error(error_msg)

                        # Prioritize fit_markdown for cleaner content
                        if hasattr(result, "fit_markdown") and result.fit_markdown:
                            logger.info("Using filtered content")
                            content = result.fit_markdown
                        else:
                            logger.info(
                                f"No fit_markdown available, using original content ({len(result.markdown)} chars)"
                            )
                            content = result.markdown

                        # Clean content with existing cleaner for additional processing
                        combined_content = self.cleaner.clean_content(content)

                        if len(combined_content.strip()) < 100:
                            error_msg = "Extracted content too short after filtering"
                            if attempt < max_retries - 1:
                                logger.warning(
                                    f"Attempt {attempt + 1} failed: {error_msg}. Retrying..."
                                )
                                await asyncio.sleep(retry_delay)
                                continue
                            else:
                                return ExtractionResult.error(error_msg)

                        # Save cleaned content
                        if self.output_dir:
                            filename = self._create_filename_from_url(url)
                            file_path = self.output_dir / filename
                            file_path.write_text(combined_content, encoding="utf-8")
                            logger.debug(f"Saved filtered content to {file_path}")

                        # Create minimal metadata
                        metadata = {
                            "source_type": "crawl4ai_deep",
                            "base_url": url,
                        }

                        # Add link information if available
                        if result.links:
                            metadata["links"] = {
                                "internal": len(result.links.get("internal", [])),
                                "external": len(result.links.get("external", [])),
                            }

                    logger.info("Deep crawl extraction completed")

                    return ExtractionResult(
                        success=True, content=combined_content, metadata=metadata
                    )

            except Exception as e:
                error_msg = f"Deep crawl extraction failed: {e}"
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {error_msg}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(
                        "Deep crawl extraction failed after all retries",
                        extra={"url": url, "error": str(e), "attempts": max_retries},
                    )
                    return ExtractionResult.error(
                        f"Deep crawl extraction failed after {max_retries} attempts: {e}"
                    )

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
