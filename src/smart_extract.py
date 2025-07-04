#!/usr/bin/env python3
"""
Smart Document Extractor - Now using crawl4ai for all URL extraction
Simplified to use crawl4ai instead of complex tiered extraction system.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from .core.cleaning import MarkdownCleaner
from .core.crawl4ai_extraction import Crawl4aiExtractor
from .core.logging import get_logger, setup_logging


class SmartExtractor:
    """
    Simplified extractor using crawl4ai for all URL-based extraction.

    Much simpler than the previous tiered system - just uses crawl4ai
    which handles all the complexity of modern websites.
    """

    def __init__(
        self, output_dir: str = "output", clean: bool = True, max_pages: int = 50
    ):
        self.output_dir = Path(output_dir)
        self.clean = clean
        self.max_pages = max_pages
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Use simplified crawl4ai-based extractor
        self.extractor = Crawl4aiExtractor(self.output_dir)
        self.cleaner = MarkdownCleaner() if clean else None
        self.logger = get_logger(__name__)

    async def extract(self, url: str) -> bool:
        """Main extraction method using crawl4ai."""
        self.logger.info(
            "Starting crawl4ai extraction", url=url, max_pages=self.max_pages
        )

        # Use crawl4ai for all URL extraction
        result = await self.extractor.extract_from_url(url, max_pages=self.max_pages)

        if not result.success:
            self.logger.error("Extraction failed", url=url, error=result.error)
            return False

        # Apply cleaning if requested and we have combined content
        if self.clean and self.cleaner and result.content:
            self.logger.info("Applying markdown cleaning")
            cleaned_content = self.cleaner.clean_content(result.content)

            # Save cleaned combined content
            output_file = self.output_dir / "extracted_content.md"
            output_file.write_text(cleaned_content, encoding="utf-8")
            self.logger.info("Saved cleaned content", file_path=str(output_file))

        self.logger.info(
            "Extraction completed successfully",
            source_type=result.metadata.get("source_type"),
            successful_extractions=result.metadata.get("successful_extractions", 1),
        )
        return True

    def extract_sync(self, url: str) -> bool:
        """Synchronous wrapper for async extract method."""
        return asyncio.run(self.extract(url))


def main():
    """CLI interface for smart extraction."""
    parser = argparse.ArgumentParser(
        description="Smart document extractor using crawl4ai",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python smart_extract.py https://ratatui.rs/
  python smart_extract.py https://docs.rs/tokio/latest/tokio/ --max-pages 30
  python smart_extract.py https://example.com --output-dir my_docs --no-clean
        """,
    )

    parser.add_argument("url", help="URL to extract content from")

    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for extracted files (default: output)",
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Maximum number of pages to extract (default: 50)",
    )

    parser.add_argument(
        "--no-clean", action="store_true", help="Skip markdown cleaning step"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)

    # Create extractor and run
    extractor = SmartExtractor(
        output_dir=args.output_dir, clean=not args.no_clean, max_pages=args.max_pages
    )

    try:
        success = extractor.extract_sync(args.url)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nExtraction cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error("Unexpected error during extraction", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
