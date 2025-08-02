"""
Enhanced crawl4ai-based document extraction system.

Features intelligent deep crawling with BestFirstCrawlingStrategy,
advanced filtering, URL scoring, and real-time processing capabilities.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy, BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, SEOFilter, URLPatternFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

from context_server.core.services.extraction.utils import FileUtils, URLUtils
from context_server.core.text.cleaning import MarkdownCleaner

# Configure structured logging
logger = logging.getLogger(__name__)


@dataclass
class DeepCrawlConfig:
    """Configuration for intelligent deep crawling."""

    # Strategy selection
    strategy_type: str = "best_first"  # "bfs", "dfs", "best_first"
    max_depth: int = 3
    max_pages: int = 50
    include_external: bool = False

    # Intelligent features
    keywords: List[str] = field(default_factory=list)
    url_patterns: List[str] = field(default_factory=list)
    seo_threshold: float = 0.5
    content_threshold: float = 0.7

    # Performance settings
    streaming: bool = True
    enable_scoring: bool = True
    enable_filtering: bool = True

    # Crawler settings
    word_count_threshold: int = 15
    page_timeout: int = 20000
    excluded_tags: List[str] = field(
        default_factory=lambda: ["nav", "footer", "header", "aside", "script", "style"]
    )

    @classmethod
    def for_documentation(
        cls, keywords: Optional[List[str]] = None
    ) -> "DeepCrawlConfig":
        """Optimized configuration for documentation sites."""
        return cls(
            strategy_type="best_first",
            max_depth=4,
            max_pages=100,
            keywords=keywords or ["docs", "guide", "tutorial", "api", "reference"],
            url_patterns=["*docs*", "*guide*", "*tutorial*", "*reference*", "*api*"],
            seo_threshold=0.3,  # Lowered from 0.6 to be less restrictive
            content_threshold=0.8,
            streaming=True,
            enable_scoring=True,
            enable_filtering=False,  # Temporarily disabled to test page discovery
        )

    @classmethod
    def for_comprehensive(cls) -> "DeepCrawlConfig":
        """Configuration for comprehensive site crawling."""
        return cls(
            strategy_type="bfs",
            max_depth=3,
            max_pages=200,
            streaming=True,
            enable_scoring=False,  # No scoring for comprehensive crawl
            enable_filtering=False,
        )

    @classmethod
    def for_focused(cls, keywords: List[str], max_pages: int = 25) -> "DeepCrawlConfig":
        """Configuration for focused, high-relevance crawling."""
        return cls(
            strategy_type="best_first",
            max_depth=2,
            max_pages=max_pages,
            keywords=keywords,
            content_threshold=0.9,  # Very high relevance only
            enable_scoring=True,
            enable_filtering=True,
        )


@dataclass
class PageResult:
    """Individual page result from multi-page extraction."""

    url: str
    title: str
    content: str
    metadata: dict
    relevance_score: float = 1.0


class ExtractionResult:
    """Result of document extraction operation."""

    def __init__(
        self,
        success: bool = True,
        content: str = "",
        metadata: dict | None = None,
        error: str | None = None,
        extracted_pages: list | None = None,
        individual_pages: List[PageResult] | None = None,
    ):
        self.success = success
        self.content = content  # Will be empty for multi-page extractions
        self.metadata = metadata or {}
        self.error = error
        self.extracted_pages = extracted_pages or []  # Legacy format
        self.individual_pages = individual_pages or []  # New individual page format

    @classmethod
    def error(cls, message: str) -> "ExtractionResult":
        """Create error result."""
        return cls(success=False, error=message)

    @property
    def is_multi_page(self) -> bool:
        """Check if this result contains multiple individual pages."""
        return len(self.individual_pages) > 0


class Crawl4aiExtractor:
    """Intelligent document extraction using crawl4ai with advanced deep crawling capabilities."""

    def __init__(self):
        """Initialize extractor with intelligent configuration."""
        self.cleaner = MarkdownCleaner()

    def _create_deep_crawl_strategy(self, config: DeepCrawlConfig):
        """Create the appropriate deep crawling strategy based on configuration."""

        # Create scorer if enabled
        scorer = None
        if config.enable_scoring and config.keywords:
            scorer = KeywordRelevanceScorer(keywords=config.keywords, weight=0.7)
            logger.info(f"Created keyword scorer with keywords: {config.keywords}")

        # Create filter chain if enabled
        filter_chain = None
        if config.enable_filtering:
            filters = []

            # Add URL pattern filter
            if config.url_patterns:
                filters.append(URLPatternFilter(patterns=config.url_patterns))
                logger.info(f"Added URL pattern filter: {config.url_patterns}")

            # Add SEO filter
            if config.seo_threshold > 0 and config.keywords:
                filters.append(
                    SEOFilter(threshold=config.seo_threshold, keywords=config.keywords)
                )
                logger.info(f"Added SEO filter with threshold {config.seo_threshold}")

            if filters:
                filter_chain = FilterChain(filters)

        # Log filtering status
        if filter_chain:
            logger.info("✅ Filter chain created with filters enabled")
        else:
            logger.info("🚫 No filters applied - all discovered URLs will be crawled")

        # Create strategy based on type
        strategy_params = {
            "max_depth": config.max_depth,
            "include_external": config.include_external,
            "max_pages": config.max_pages,
        }

        # Only add filter_chain if it's not None
        if filter_chain is not None:
            strategy_params["filter_chain"] = filter_chain

        if config.strategy_type == "best_first":
            # BestFirstCrawlingStrategy does NOT use score_threshold (processes by highest score first)
            if scorer is not None:
                strategy_params["url_scorer"] = scorer
            return BestFirstCrawlingStrategy(**strategy_params)
        else:  # fallback to BFS
            # BFS/DFS strategies REQUIRE score_threshold parameter
            strategy_params[
                "score_threshold"
            ] = 0.3  # Minimum score for URLs to be crawled
            return BFSDeepCrawlStrategy(**strategy_params)

    async def extract_url(
        self,
        url: str,
        max_pages: int = 50,
        config: Optional[DeepCrawlConfig] = None,
        keywords: Optional[List[str]] = None,
    ) -> ExtractionResult:
        """Extract content from URL with intelligent deep crawling support."""
        try:
            # Normalize and validate URL
            normalized_url = URLUtils.normalize_url(url)
            domain = URLUtils.extract_domain(normalized_url)

            # Create intelligent crawling configuration
            if config is None:
                # Auto-detect best configuration based on URL and keywords
                if keywords or any(
                    pattern in normalized_url.lower()
                    for pattern in ["docs", "guide", "tutorial", "api"]
                ):
                    config = DeepCrawlConfig.for_documentation(keywords)
                    logger.info("Using documentation-optimized crawling configuration")
                else:
                    config = DeepCrawlConfig.for_comprehensive()
                    logger.info("Using comprehensive crawling configuration")

            # Override max_pages if provided
            config.max_pages = max_pages

            logger.info(f"Starting intelligent extraction from {normalized_url}")
            logger.info(
                f"Strategy: {config.strategy_type}, Max depth: {config.max_depth}, "
                f"Max pages: {config.max_pages}, Streaming: {config.streaming}"
            )

            # Create intelligent deep crawling strategy
            deep_crawl_strategy = self._create_deep_crawl_strategy(config)

            # Minimal crawler configuration to debug browser issues
            crawler_config = CrawlerRunConfig(
                # Intelligent multi-page discovery
                deep_crawl_strategy=deep_crawl_strategy,
                # Minimal settings
                word_count_threshold=10,  # Very low threshold
                exclude_external_links=True,
                # Very basic settings
                page_timeout=10000,
                cache_mode=CacheMode.ENABLED,
                stream=False,
                # No wait condition to avoid timeout issues
            )

            # Execute intelligent extraction with enhanced crawler config
            # Try with more robust browser configuration
            async with AsyncWebCrawler(
                verbose=True,  # Enable verbose logging for debugging
                headless=True,
                browser_type="chromium",  # Explicitly specify browser
                sleep_on_close=False,  # Don't sleep when closing
            ) as crawler:
                start_time = datetime.now()

                # Run intelligent extraction with streaming support
                if config.streaming:
                    logger.info("Using streaming mode for real-time processing")

                logger.info(
                    f"Starting crawl with strategy: {config.strategy_type}, max_depth: {config.max_depth}, max_pages: {config.max_pages}"
                )
                results = await crawler.arun(url=normalized_url, config=crawler_config)

                # Handle async generator for streaming mode
                if hasattr(results, "__aiter__"):
                    # It's an async generator, collect all results
                    collected_results = []
                    async for result in results:
                        collected_results.append(result)
                    results = collected_results
                elif not isinstance(results, list):
                    # Handle single result
                    results = (
                        [results]
                        if hasattr(results, "success") and results.success
                        else []
                    )

                extraction_time = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"Intelligent crawling found {len(results)} pages in {extraction_time:.2f}s"
                )

                # Debug: Log details about discovered pages
                if results:
                    logger.info("🔍 Discovered pages:")
                    for i, result in enumerate(results[:10]):  # Show first 10
                        logger.info(
                            f"  {i+1}. {result.url} (success: {result.success})"
                        )
                    if len(results) > 10:
                        logger.info(f"  ... and {len(results) - 10} more pages")
                else:
                    logger.warning(
                        "❌ No pages discovered - this indicates a crawling configuration issue"
                    )

                # Validate results
                if not results:
                    error_msg = "No pages extracted from intelligent crawl"
                    logger.error(error_msg)
                    return ExtractionResult.error(error_msg)

                # Process pages with enhanced quality assessment
                extracted_contents = []
                total_content_length = 0
                successful_pages = 0
                relevance_scores = []

                for i, result in enumerate(results):
                    if not result.success or not result.markdown:
                        logger.debug(f"Skipping page {i+1}: No content extracted")
                        continue

                    # Use raw_markdown for optimal quality
                    content = result.markdown.raw_markdown or result.markdown

                    # Enhanced content validation
                    if (
                        len(content.strip()) < config.word_count_threshold * 5
                    ):  # More sophisticated threshold
                        logger.debug(
                            f"Skipping {result.url} - content below quality threshold"
                        )
                        continue

                    # Enhanced content cleaning
                    cleaned_content = self.cleaner.clean_content(content)

                    # Calculate relevance score if keywords provided
                    relevance_score = 1.0  # Default score
                    if config.keywords:
                        keyword_matches = sum(
                            1
                            for keyword in config.keywords
                            if keyword.lower() in cleaned_content.lower()
                        )
                        relevance_score = min(
                            keyword_matches / len(config.keywords), 1.0
                        )
                        relevance_scores.append(relevance_score)

                    extracted_contents.append(
                        {
                            "url": result.url,
                            "content": cleaned_content,
                            "length": len(cleaned_content),
                            "title": getattr(result, "title", "")
                            or URLUtils.url_to_title(result.url),
                            "relevance_score": relevance_score,
                            "extraction_timestamp": datetime.now().isoformat(),
                        }
                    )

                    total_content_length += len(cleaned_content)
                    successful_pages += 1

                    logger.info(
                        f"✓ Page {i+1}: {len(cleaned_content):,} chars "
                        f"(relevance: {relevance_score:.2f}) from {result.url}"
                    )

                if not extracted_contents:
                    return ExtractionResult.error(
                        "No high-quality content extracted from any page"
                    )

                # Create individual PageResult objects for each extracted page
                individual_pages = []
                for item in extracted_contents:
                    page_result = PageResult(
                        url=item["url"],
                        title=item["title"],
                        content=item["content"],
                        metadata={
                            "length": item["length"],
                            "relevance_score": item["relevance_score"],
                            "extraction_timestamp": item["extraction_timestamp"],
                            "source_type": "intelligent_crawl4ai",
                            "crawl_strategy": config.strategy_type,
                        },
                        relevance_score=item["relevance_score"],
                    )
                    individual_pages.append(page_result)

                # Create overall extraction metadata
                metadata = {
                    "source_type": "intelligent_crawl4ai",
                    "crawl_strategy": config.strategy_type,
                    "base_url": normalized_url,
                    "pages_found": len(results),
                    "pages_extracted": successful_pages,
                    "total_content_length": total_content_length,
                    "average_page_size": (
                        total_content_length // successful_pages
                        if successful_pages
                        else 0
                    ),
                    "average_relevance_score": (
                        sum(relevance_scores) / len(relevance_scores)
                        if relevance_scores
                        else 1.0
                    ),
                    "extraction_time": extraction_time,
                    "config": {
                        "strategy": config.strategy_type,
                        "max_depth": config.max_depth,
                        "keywords_used": config.keywords,
                        "url_patterns": config.url_patterns,
                        "streaming": config.streaming,
                        "filters_enabled": config.enable_filtering,
                        "scoring_enabled": config.enable_scoring,
                    },
                    "page_urls": [page.url for page in individual_pages],
                    "page_titles": [page.title for page in individual_pages],
                    "relevance_scores": relevance_scores,
                }

                logger.info(
                    f"Intelligent extraction completed: {successful_pages} individual pages, "
                    f"{total_content_length:,} total chars, "
                    f"avg relevance: {metadata['average_relevance_score']:.3f}"
                )

                return ExtractionResult(
                    success=True,
                    content="",  # No combined content for multi-page extractions
                    metadata=metadata,
                    individual_pages=individual_pages,
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
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Try with different encoding
                try:
                    content = path.read_text(encoding="latin-1")
                except Exception as e:
                    return ExtractionResult.error(
                        f"Could not read file {file_path}: {e}"
                    )

            # Clean content if it's markdown
            if path.suffix.lower() in [".md", ".markdown"]:
                content = self.cleaner.clean_content(content)

            # Build metadata
            metadata = {
                "source_file": str(path.absolute()),
                "source_type": "file",
                "file_size": path.stat().st_size,
                "file_extension": path.suffix,
                "title": FileUtils.create_safe_filename(path.stem),
                "extracted_at": datetime.now().isoformat(),
                "content_length": len(content),
            }

            logger.info(
                f"Successfully extracted {len(content)} characters from file: {path.name}"
            )

            return ExtractionResult(success=True, content=content, metadata=metadata)

        except Exception as e:
            error_msg = f"File extraction failed for {file_path}: {str(e)}"
            logger.error(error_msg)
            return ExtractionResult.error(error_msg)


__all__ = ["Crawl4aiExtractor", "ExtractionResult", "PageResult", "DeepCrawlConfig"]
