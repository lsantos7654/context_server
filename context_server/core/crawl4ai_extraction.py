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

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from docling.document_converter import DocumentConverter

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

        # Keep Docling for non-URL document processing (PDFs, etc.)
        self.docling_converter = DocumentConverter()

        # Initialize markdown cleaner for enhanced content processing
        self.cleaner = MarkdownCleaner()

        # Configuration for filtering behavior
        self.config = config or {}
        self.filtering_config = self.config.get("filtering", {})

        # Configure custom priority patterns
        self.custom_high_priority = self.filtering_config.get(
            "high_priority_patterns", []
        )
        self.custom_medium_priority = self.filtering_config.get(
            "medium_priority_patterns", []
        )
        self.custom_skip_patterns = self.filtering_config.get(
            "additional_skip_patterns", []
        )

    async def extract_from_url(self, url: str, max_pages: int = 50) -> ExtractionResult:
        """Extract content from URL using crawl4ai with retry logic."""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                logger.info(
                    "Starting crawl4ai extraction",
                    extra={"url": url, "max_pages": max_pages, "attempt": attempt + 1},
                )

                # Initialize crawler with minimal config to avoid BrowserConfig errors
                # Simplified configuration to avoid multiple browser_config arguments
                async with AsyncWebCrawler(verbose=False) as crawler:
                    # First, get the main page and discover all internal links
                    initial_result = await crawler.arun(
                        url=url,
                        config=CrawlerRunConfig(
                            # Wait for dynamic content to load
                            js_code=["window.scrollTo(0, document.body.scrollHeight);"],
                            wait_for="css:body",
                            # Only get internal links for this domain
                            exclude_external_links=True,
                            # Cache for efficiency
                            cache_mode=CacheMode.ENABLED,
                            # Give time for JavaScript to execute but not too long
                            delay_before_return_html=1.5,
                            # Disable verbose to avoid BrowserConfig issues
                            verbose=False,
                            # Add timeout to prevent hanging
                            page_timeout=30000,  # 30 seconds
                        ),
                    )

                    if not initial_result.success:
                        error_msg = (
                            f"Failed to fetch initial page: {initial_result.error}"
                        )
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Attempt {attempt + 1} failed: {error_msg}. Retrying..."
                            )
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            return ExtractionResult.error(error_msg)

                    # Extract internal links
                    internal_links = initial_result.links.get("internal", [])

                    # Filter and limit links
                    doc_links = self._filter_documentation_links(
                        internal_links, url, max_pages
                    )

                    logger.info(
                        f"Found {len(internal_links)} internal links, "
                        f"filtered to {len(doc_links)} documentation links"
                    )

                    # Validate extraction completeness
                    validation_result = self._validate_extraction_completeness(
                        doc_links,
                        [
                            link["href"] if isinstance(link, dict) else link
                            for link in internal_links
                        ],
                        url,
                    )

                    # Log validation warnings
                    if validation_result["warnings"]:
                        for warning in validation_result["warnings"]:
                            logger.warning(f"Extraction validation: {warning}")

                    if validation_result["missing_important_urls"]:
                        logger.warning(
                            f"Missing {len(validation_result['missing_important_urls'])} "
                            f"important URLs (first 3): "
                            f"{validation_result['missing_important_urls'][:3]}"
                        )

                    # If no links found, just extract the single page
                    if not doc_links:
                        doc_links = [url]

                    # Extract content from all documentation links with timeout
                    results = await crawler.arun_many(
                        urls=[
                            link["href"] if isinstance(link, dict) else link
                            for link in doc_links
                        ],
                        config=CrawlerRunConfig(
                            exclude_external_links=True,
                            cache_mode=CacheMode.ENABLED,
                            # Don't need to wait as long for individual pages
                            delay_before_return_html=0.8,
                            # Disable verbose to avoid BrowserConfig issues
                            verbose=False,
                            # Add timeout for individual pages
                            page_timeout=20000,  # 20 seconds per page
                        ),
                    )

                    # Process results and save files
                    extracted_contents = []
                    successful_extractions = 0

                    for i, result in enumerate(results):
                        if result.success and result.markdown:
                            content = result.markdown
                            page_url = (
                                doc_links[i]["href"]
                                if isinstance(doc_links[i], dict)
                                else doc_links[i]
                            )

                            # Clean and validate content using enhanced cleaner
                            cleaned_content = self.cleaner.clean_content(content)

                            if len(cleaned_content.strip()) < 100:  # Skip tiny pages
                                logger.debug(f"Skipping {page_url} - content too short")
                                continue

                            # Save individual file
                            if self.output_dir:
                                filename = self._create_filename_from_url(page_url)
                                file_path = self.output_dir / filename
                                file_path.write_text(cleaned_content, encoding="utf-8")
                                logger.debug(f"Saved {file_path}")

                            extracted_contents.append(
                                {
                                    "url": page_url,
                                    "content": cleaned_content,
                                    "filename": filename if self.output_dir else None,
                                }
                            )
                            successful_extractions += 1
                        else:
                            page_url = (
                                doc_links[i]["href"]
                                if isinstance(doc_links[i], dict)
                                else doc_links[i]
                            )
                            error_msg = (
                                result.error
                                if hasattr(result, "error")
                                else "Unknown error"
                            )
                            logger.debug(f"Failed to extract {page_url}: {error_msg}")

                    if not extracted_contents:
                        error_msg = "No content successfully extracted"
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Attempt {attempt + 1} failed: {error_msg}. Retrying..."
                            )
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            return ExtractionResult.error(error_msg)

                    # Return metadata for all extracted contents instead of combining
                    base_metadata = {
                        "source_type": "crawl4ai",
                        "base_url": url,
                        "total_links_found": len(internal_links),
                        "filtered_links": len(doc_links),
                        "successful_extractions": successful_extractions,
                        "extraction_time": datetime.now().isoformat(),
                    }

                    logger.info(
                        "Crawl4ai extraction completed",
                        extra={
                            "successful_extractions": successful_extractions,
                            "total_links": len(doc_links),
                        },
                    )

                    # For backwards compatibility, combine content but also store individual pages
                    combined_content = "\n\n---\n\n".join(
                        [item["content"] for item in extracted_contents]
                    )

                    # Add individual page info to metadata
                    base_metadata["extracted_pages"] = extracted_contents

                    return ExtractionResult(
                        success=True, content=combined_content, metadata=base_metadata
                    )

            except Exception as e:
                error_msg = f"Crawl4ai extraction failed: {e}"
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {error_msg}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(
                        "Crawl4ai extraction failed after all retries",
                        extra={"url": url, "error": str(e), "attempts": max_retries},
                    )
                    return ExtractionResult.error(
                        f"Crawl4ai extraction failed after {max_retries} attempts: {e}"
                    )

    def extract_from_pdf(self, file_path: str | Path) -> ExtractionResult:
        """Extract content from PDF file using Docling."""
        try:
            path = Path(file_path).absolute()
            logger.info("Extracting PDF content", extra={"file_path": str(path)})

            result = self.docling_converter.convert(path)
            content = result.document.export_to_markdown()

            metadata = {
                "source_type": "pdf",
                "file_path": str(path),
                "extraction_time": datetime.now().isoformat(),
            }

            return ExtractionResult(success=True, content=content, metadata=metadata)

        except Exception as e:
            logger.error(
                "PDF extraction failed",
                extra={"file_path": str(file_path), "error": str(e)},
            )
            return ExtractionResult.error(f"Failed to extract PDF: {e}")

    def _filter_documentation_links(
        self, links: list, base_url: str, max_pages: int
    ) -> list:
        """Filter links to focus on documentation content with smart prioritization."""
        if not links:
            logger.info("No links to filter")
            return []

        # Convert to URLs if they're link objects
        urls = []
        for link in links:
            if isinstance(link, dict):
                urls.append(link.get("href", ""))
            else:
                urls.append(str(link))

        logger.info(f"Starting URL filtering with {len(urls)} discovered URLs")

        # Debug: Log some example URLs
        sample_urls = urls[:10] if len(urls) <= 10 else urls[:5] + ["..."] + urls[-5:]
        logger.info(f"Sample URLs: {sample_urls}")

        # Parse the base URL to understand what section we're targeting
        parsed_base = urlparse(base_url)
        base_path = parsed_base.path.rstrip("/")

        logger.info(f"Base URL: {base_url}, Base path: {base_path}")

        # Filter and prioritize URLs with smart prioritization
        high_priority_urls = []  # Critical content (widgets, core examples)
        medium_priority_urls = []  # Documentation, tutorials
        low_priority_urls = []  # Other content
        skipped_urls = []
        skip_reasons = {}

        for url in urls:
            if not url or not isinstance(url, str):
                skipped_urls.append(url)
                skip_reasons[str(url)] = "empty_or_not_string"
                continue

            # Skip external links (double-check)
            if not URLUtils.is_same_domain(base_url, url):
                skipped_urls.append(url)
                skip_reasons[url] = "external_domain"
                continue

            parsed_url = urlparse(url)
            url_path = parsed_url.path.rstrip("/")

            # Skip common non-documentation patterns (configurable)
            default_skip_patterns = [
                "/login",
                "/signup",
                "/settings",
                "/profile",
                "mailto:",
                "tel:",
                "javascript:",
                ".jpg",
                ".png",
                ".gif",
                ".pdf",
                ".zip",
                "/api/",
                "/admin/",
                "/dashboard/",
                "#",  # Skip anchors
            ]
            skip_patterns = default_skip_patterns + self.custom_skip_patterns

            should_skip = False
            for pattern in skip_patterns:
                if pattern in url.lower():
                    skipped_urls.append(url)
                    skip_reasons[url] = f"matches_skip_pattern_{pattern}"
                    should_skip = True
                    break

            if should_skip:
                continue

            # Skip if it's the same as base URL
            if url.rstrip("/") == base_url.rstrip("/"):
                skipped_urls.append(url)
                skip_reasons[url] = "same_as_base_url"
                continue

            # HIGH PRIORITY: Widgets and core components (configurable)
            priority_assigned = False
            default_high_priority_patterns = [
                "/examples/widgets/",
                "/widgets/",
                "/components/",
            ]
            high_priority_patterns = (
                default_high_priority_patterns + self.custom_high_priority
            )

            for pattern in high_priority_patterns:
                if pattern in url_path:
                    high_priority_urls.append(url)
                    logger.debug(f"HIGH PRIORITY: {url} (matches {pattern})")
                    priority_assigned = True
                    break

            if priority_assigned:
                continue

            # MEDIUM PRIORITY: Documentation, tutorials, examples (configurable)
            default_medium_priority_patterns = [
                "/tutorials/",
                "/examples/",
                "/docs/",
                "/guide/",
                "/reference/",
                "/installation/",
            ]
            medium_priority_patterns = (
                default_medium_priority_patterns + self.custom_medium_priority
            )

            for pattern in medium_priority_patterns:
                if pattern in url_path:
                    medium_priority_urls.append(url)
                    logger.debug(f"MEDIUM PRIORITY: {url} (matches {pattern})")
                    priority_assigned = True
                    break

            if priority_assigned:
                continue

            # Prioritize URLs that are "under" the target path
            if base_path and base_path != "/":
                if url_path.startswith(base_path):
                    # This URL is under our target section - medium priority
                    medium_priority_urls.append(url)
                    logger.debug(f"MEDIUM PRIORITY: {url} (under target path)")
                elif url_path != "/":
                    # This is other documentation - low priority
                    low_priority_urls.append(url)
                    logger.debug(f"LOW PRIORITY: {url} (other documentation)")
                else:
                    skipped_urls.append(url)
                    skip_reasons[url] = "root_path"
            else:
                # If targeting root, this is general content - low priority
                if url_path != "/":
                    low_priority_urls.append(url)
                    logger.debug(f"LOW PRIORITY: {url} (general content)")
                else:
                    # Root path gets medium priority
                    medium_priority_urls.append(url)
                    logger.debug(f"MEDIUM PRIORITY: {url} (root path)")

        # Combine URLs in priority order: high -> medium -> low
        all_urls = high_priority_urls + medium_priority_urls + low_priority_urls

        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(all_urls))

        # Always include the original URL if not already included
        if base_url not in unique_urls:
            unique_urls.insert(0, base_url)

        # Limit to max_pages
        limited_urls = unique_urls[:max_pages]

        # Log filtering statistics
        logger.info(
            f"URL Filtering Results:"
            f"\n  - Discovered: {len(urls)}"
            f"\n  - High Priority: {len(high_priority_urls)}"
            f"\n  - Medium Priority: {len(medium_priority_urls)}"
            f"\n  - Low Priority: {len(low_priority_urls)}"
            f"\n  - Skipped: {len(skipped_urls)}"
            f"\n  - After dedup: {len(unique_urls)}"
            f"\n  - Final (after limit): {len(limited_urls)}"
        )

        # Log specific table widget URL if it's being looked for
        table_urls = [url for url in urls if "table" in url.lower()]
        if table_urls:
            logger.info(f"Found {len(table_urls)} table-related URLs:")
            for table_url in table_urls:
                if table_url in limited_urls:
                    logger.info(f"  ✓ INCLUDED: {table_url}")
                elif table_url in skip_reasons:
                    logger.warning(
                        f"  ✗ SKIPPED: {table_url} (reason: {skip_reasons[table_url]})"
                    )
                else:
                    logger.warning(f"  ? UNKNOWN: {table_url}")

        # Log some examples of skipped URLs for debugging
        if skipped_urls:
            skip_examples = {}
            for url in skipped_urls[:20]:  # Limit examples
                reason = skip_reasons.get(url, "unknown")
                if reason not in skip_examples:
                    skip_examples[reason] = []
                skip_examples[reason].append(url)

            logger.debug("Examples of skipped URLs by reason:")
            for reason, example_urls in skip_examples.items():
                logger.debug(f"  {reason}: {example_urls[:3]}")

        return limited_urls

    def _validate_extraction_completeness(
        self, extracted_urls: list[str], all_discovered_urls: list[str], base_url: str
    ) -> dict:
        """Validate that important URLs were not missed during extraction."""
        validation_result = {
            "missing_important_urls": [],
            "warnings": [],
            "coverage_stats": {},
        }

        # Convert extracted URLs to a set for faster lookup
        extracted_set = set(extracted_urls)

        # Check for missing widget/component URLs if this looks like a documentation site
        if any(
            pattern in base_url.lower()
            for pattern in ["docs", "documentation", "guide", "tutorial", "examples"]
        ):
            # Look for important patterns that should be included
            important_patterns = [
                ("widgets", "Widget documentation"),
                ("components", "Component documentation"),
                ("api", "API documentation"),
                ("examples", "Code examples"),
            ]

            for pattern, description in important_patterns:
                discovered_with_pattern = [
                    url for url in all_discovered_urls if pattern in url.lower()
                ]
                extracted_with_pattern = [
                    url for url in extracted_urls if pattern in url.lower()
                ]

                if discovered_with_pattern and not extracted_with_pattern:
                    validation_result["warnings"].append(
                        f"Found {len(discovered_with_pattern)} {description} URLs but none were extracted"
                    )
                    validation_result["missing_important_urls"].extend(
                        discovered_with_pattern[:5]  # Include first 5 examples
                    )

                validation_result["coverage_stats"][pattern] = {
                    "discovered": len(discovered_with_pattern),
                    "extracted": len(extracted_with_pattern),
                    "coverage_ratio": (
                        len(extracted_with_pattern) / len(discovered_with_pattern)
                        if discovered_with_pattern
                        else 1.0
                    ),
                }

        # Check for specific important patterns for ratatui site
        if "ratatui.rs" in base_url:
            # Check for table widget specifically
            table_urls = [url for url in all_discovered_urls if "table" in url.lower()]
            table_extracted = [url for url in extracted_urls if "table" in url.lower()]

            if table_urls and not table_extracted:
                validation_result["warnings"].append(
                    f"Found {len(table_urls)} table-related URLs but none were extracted"
                )
                validation_result["missing_important_urls"].extend(table_urls)

        return validation_result

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
