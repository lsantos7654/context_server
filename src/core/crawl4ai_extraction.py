"""
Simple crawl4ai-based document extraction system.

Replaces the complex tiered extraction system with a single, unified approach
using crawl4ai for all URL-based extraction.
"""

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

    def __init__(self, output_dir: Path | None = None):
        """Initialize extractor with optional output directory."""
        self.output_dir = FileUtils.ensure_directory(output_dir or Path("output"))

        # Keep Docling for non-URL document processing (PDFs, etc.)
        self.docling_converter = DocumentConverter()

        # Initialize markdown cleaner for enhanced content processing
        self.cleaner = MarkdownCleaner()

    async def extract_from_url(self, url: str, max_pages: int = 50) -> ExtractionResult:
        """Extract content from URL using crawl4ai."""
        try:
            logger.info(
                "Starting crawl4ai extraction",
                extra={"url": url, "max_pages": max_pages},
            )

            async with AsyncWebCrawler() as crawler:
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
                        # Give time for JavaScript to execute
                        delay_before_return_html=2.0,
                    ),
                )

                if not initial_result.success:
                    return ExtractionResult.error(
                        f"Failed to fetch initial page: {initial_result.error}"
                    )

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

                # If no links found, just extract the single page
                if not doc_links:
                    doc_links = [url]

                # Extract content from all documentation links
                results = await crawler.arun_many(
                    urls=[
                        link["href"] if isinstance(link, dict) else link
                        for link in doc_links
                    ],
                    config=CrawlerRunConfig(
                        exclude_external_links=True,
                        cache_mode=CacheMode.ENABLED,
                        # Don't need to wait as long for individual pages
                        delay_before_return_html=1.0,
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
                    return ExtractionResult.error("No content successfully extracted")

                # Combine all content
                combined_content = "\n\n---\n\n".join(
                    [item["content"] for item in extracted_contents]
                )

                metadata = {
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

                return ExtractionResult(
                    success=True, content=combined_content, metadata=metadata
                )

        except Exception as e:
            logger.error(
                "Crawl4ai extraction failed", extra={"url": url, "error": str(e)}
            )
            return ExtractionResult.error(f"Crawl4ai extraction failed: {e}")

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
        """Filter links to focus on documentation content."""
        if not links:
            return []

        # Convert to URLs if they're link objects
        urls = []
        for link in links:
            if isinstance(link, dict):
                urls.append(link.get("href", ""))
            else:
                urls.append(str(link))

        # Parse the base URL to understand what section we're targeting
        parsed_base = urlparse(base_url)
        base_path = parsed_base.path.rstrip("/")

        # Filter and prioritize URLs
        prioritized_urls = []
        related_urls = []

        for url in urls:
            if not url or not isinstance(url, str):
                continue

            # Skip external links (double-check)
            if not URLUtils.is_same_domain(base_url, url):
                continue

            parsed_url = urlparse(url)
            url_path = parsed_url.path.rstrip("/")

            # Skip common non-documentation patterns
            skip_patterns = [
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

            if any(pattern in url.lower() for pattern in skip_patterns):
                continue

            # Prioritize URLs that are "under" the target path
            if base_path and base_path != "/":
                if url_path.startswith(base_path):
                    # This URL is under our target section - high priority
                    prioritized_urls.append(url)
                elif (
                    url_path != "/"
                    and not url_path.startswith("/showcase")
                    and not url_path.startswith("/highlights")
                ):
                    # This is a related documentation URL - lower priority
                    related_urls.append(url)
            else:
                # If targeting root, include all documentation
                if not url_path.startswith("/showcase") and not url_path.startswith(
                    "/highlights"
                ):
                    prioritized_urls.append(url)

        # Combine prioritized and related URLs, with prioritized first
        all_urls = prioritized_urls + related_urls

        # Remove duplicates while preserving order
        unique_urls = list(dict.fromkeys(all_urls))

        # Always include the original URL if not already included
        if base_url not in unique_urls:
            unique_urls.insert(0, base_url)

        # Limit to max_pages
        limited_urls = unique_urls[:max_pages]

        logger.info(
            f"Filtered {len(urls)} links to {len(limited_urls)} documentation URLs "
            f"(prioritized: {len(prioritized_urls)}, related: {len(related_urls)})"
        )

        return limited_urls

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
