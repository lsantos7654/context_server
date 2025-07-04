"""
Unified document extraction system.

Consolidates document processing functionality from extract.py, smart_extract.py,
and parts of embed.py into a clean, reusable interface following CLAUDE.md principles.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union
from urllib.parse import urlparse

import requests
from docling.document_converter import DocumentConverter

from ..utils.sitemap import get_sitemap_urls

# Configure structured logging
logger = logging.getLogger(__name__)


class ExtractionResult:
    """Result of document extraction operation."""

    def __init__(
        self,
        success: bool = True,
        content: str = "",
        metadata: Optional[Dict] = None,
        error: Optional[str] = None,
    ):
        self.success = success
        self.content = content
        self.metadata = metadata or {}
        self.error = error

    @classmethod
    def error(cls, message: str) -> "ExtractionResult":
        """Create error result."""
        return cls(success=False, error=message)


class UnifiedDocumentExtractor:
    """
    Unified document extractor that replaces DocumentConverterCLI and
    consolidates extraction logic from multiple modules.

    Follows CLAUDE.md principles:
    - Single responsibility: document extraction
    - Dependency injection: converter is injected
    - Clean abstractions: simple interface
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize extractor with optional output directory."""
        self.converter = DocumentConverter()
        self.output_dir = output_dir or Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_url(self, url: str) -> ExtractionResult:
        """Extract content from URL with automatic method detection."""
        try:
            logger.info("Starting extraction for URL", extra={"url": url})

            # Smart detection of extraction method
            if self._is_sitemap_url(url):
                return self._extract_sitemap(url)
            elif self._should_use_sitemap(url):
                sitemap_url = self._find_sitemap(url)
                if sitemap_url:
                    logger.info(
                        "Found sitemap, using sitemap extraction",
                        extra={"sitemap_url": sitemap_url},
                    )
                    return self._extract_sitemap(sitemap_url)

            # Fall back to single page extraction
            return self._extract_single_page(url)

        except Exception as e:
            logger.error("URL extraction failed", extra={"url": url, "error": str(e)})
            return ExtractionResult.error(f"Failed to extract from URL: {e}")

    def extract_from_pdf(self, file_path: Union[str, Path]) -> ExtractionResult:
        """Extract content from PDF file."""
        try:
            path = Path(file_path).absolute()
            logger.info("Extracting PDF content", extra={"file_path": str(path)})

            result = self.converter.convert(path)
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

    def _extract_single_page(self, url: str) -> ExtractionResult:
        """Extract content from single web page."""
        try:
            result = self.converter.convert(url)
            content = result.document.export_to_markdown()

            metadata = {
                "source_type": "url",
                "url": url,
                "extraction_time": datetime.now().isoformat(),
            }

            return ExtractionResult(success=True, content=content, metadata=metadata)

        except Exception as e:
            return ExtractionResult.error(f"Failed to extract page {url}: {e}")

    def _extract_sitemap(self, sitemap_url: str) -> ExtractionResult:
        """Extract content from all URLs in sitemap."""
        try:
            urls = get_sitemap_urls(sitemap_url)
            logger.info("Processing sitemap URLs", extra={"url_count": len(urls)})

            results = []
            conv_results = self.converter.convert_all(urls)

            for i, result in enumerate(conv_results):
                if result.document:
                    content = result.document.export_to_markdown()
                    url = urls[i]

                    # Save individual files if output directory specified
                    if self.output_dir:
                        filename = self._create_filename_from_url(url)
                        file_path = self.output_dir / filename
                        file_path.write_text(content, encoding="utf-8")
                        logger.debug(
                            "Saved extracted content",
                            extra={"file_path": str(file_path)},
                        )

                    results.append(
                        {
                            "url": url,
                            "content": content,
                            "filename": filename if self.output_dir else None,
                        }
                    )

            metadata = {
                "source_type": "sitemap",
                "sitemap_url": sitemap_url,
                "processed_urls": len(results),
                "total_urls": len(urls),
                "extraction_time": datetime.now().isoformat(),
            }

            # Combine all content with separators
            combined_content = "\n\n---\n\n".join([r["content"] for r in results])

            return ExtractionResult(
                success=True, content=combined_content, metadata=metadata
            )

        except Exception as e:
            return ExtractionResult.error(f"Failed to extract sitemap: {e}")

    def _is_sitemap_url(self, url: str) -> bool:
        """Check if URL points to a sitemap."""
        return url.lower().endswith(("sitemap.xml", "sitemap_index.xml"))

    def _should_use_sitemap(self, url: str) -> bool:
        """Determine if we should look for a sitemap for this URL."""
        # For now, always try to find sitemap for better coverage
        return True

    def _find_sitemap(self, url: str) -> Optional[str]:
        """Find sitemap URL for a given website."""
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # Check robots.txt first
        try:
            robots_url = f"{base_url}/robots.txt"
            response = requests.get(robots_url, timeout=10)
            if response.status_code == 200:
                for line in response.text.split("\n"):
                    if line.lower().startswith("sitemap:"):
                        sitemap_url = line.split(":", 1)[1].strip()
                        logger.debug(
                            "Found sitemap in robots.txt",
                            extra={"sitemap_url": sitemap_url},
                        )
                        return sitemap_url
        except Exception:
            pass

        # Check common sitemap locations
        common_paths = [
            "/sitemap.xml",
            "/sitemap_index.xml",
            "/sitemap.xml.gz",
            "/sitemap/sitemap.xml",
            "/sitemaps/sitemap.xml",
        ]

        for path in common_paths:
            try:
                sitemap_url = f"{base_url}{path}"
                response = requests.head(sitemap_url, timeout=10)
                if response.status_code == 200:
                    logger.debug(
                        "Found sitemap at common location",
                        extra={"sitemap_url": sitemap_url},
                    )
                    return sitemap_url
            except Exception:
                continue

        return None

    def _create_filename_from_url(self, url: str) -> str:
        """Create safe filename from URL."""
        parsed = urlparse(url)
        filename = re.sub(r"[^\w\-_.]", "_", parsed.path.strip("/") or "index")
        if not filename.endswith(".md"):
            filename += ".md"
        return filename
