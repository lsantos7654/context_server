#!/usr/bin/env python3
"""
Smart Document Extractor - Automatically determines the best extraction method
Refactored to use consolidated core utilities following CLAUDE.md principles.
"""

import argparse
import sys
from pathlib import Path

from .core.cleaning import MarkdownCleaner
from .core.extraction import UnifiedDocumentExtractor
from .core.logging import get_logger, setup_logging


class SmartExtractor:
    """
    Intelligent extractor that automatically determines the best extraction method.

    Refactored to use consolidated utilities and follow CLAUDE.md principles:
    - Uses dependency injection for extractor and cleaner
    - Simplified interface with clear responsibilities
    - Structured logging instead of custom print statements
    - Delegates extraction logic to UnifiedDocumentExtractor
    """

    def __init__(self, output_dir: str = "output", clean: bool = True):
        self.output_dir = Path(output_dir)
        self.clean = clean
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Use consolidated utilities via dependency injection
        self.extractor = UnifiedDocumentExtractor(self.output_dir)
        self.cleaner = MarkdownCleaner() if clean else None
        self.logger = get_logger(__name__)

    def extract(self, url: str) -> bool:
        """
        Main extraction method using unified extraction system.

        Simplified to delegate to UnifiedDocumentExtractor, removing duplication.
        """
        self.logger.info("Starting smart extraction", url=url)

        # Use unified extractor for all extraction logic
        result = self.extractor.extract_from_url(url)

        if not result.success:
            self.logger.error("Extraction failed", url=url, error=result.error)
            return False

        # Apply cleaning if requested
        if self.clean and self.cleaner and result.content:
            self.logger.info("Applying markdown cleaning")
            cleaned_content = self.cleaner.clean_content(result.content)

            # Save cleaned content if it was sitemap extraction
            # (individual files already saved)
            if result.metadata.get("source_type") != "sitemap":
                output_file = self.output_dir / "extracted_content.md"
                output_file.write_text(cleaned_content, encoding="utf-8")
                self.logger.info("Saved cleaned content", file_path=str(output_file))

        self.logger.info(
            "Extraction completed successfully",
            source_type=result.metadata.get("source_type"),
            processed_urls=result.metadata.get("processed_urls", 1),
        )
        return True


def main():
    """CLI interface for smart extraction."""
    parser = argparse.ArgumentParser(
        description="Smart document extractor with automatic method detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python smart_extract.py https://docs.rs/tokio/latest/tokio/
  python smart_extract.py https://example.com --output-dir my_docs
  python smart_extract.py https://site.com --no-clean --verbose
        """,
    )

    parser.add_argument("url", help="URL to extract content from")

    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for extracted files (default: output)",
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
    extractor = SmartExtractor(output_dir=args.output_dir, clean=not args.no_clean)

    try:
        success = extractor.extract(args.url)
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
