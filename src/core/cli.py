"""
Unified CLI interface for Context Server.

Consolidates CLI functionality from extract.py and other modules into a
clean, extensible interface following CLAUDE.md principles.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .cleaning import MarkdownCleaner
from .extraction import UnifiedDocumentExtractor
from .logging import get_logger, setup_logging
from .processors import ProcessingOptions, ProcessorFactory


class ContextServerCLI:
    """
    Unified command-line interface for Context Server operations.

    Consolidates and replaces fragmented CLI interfaces with a single,
    clean interface following CLAUDE.md principles.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser."""
        parser = argparse.ArgumentParser(
            description="Context Server - Unified document processing and extraction",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Smart extraction (auto-detects method)
  context-server extract https://docs.rs/tokio/latest/tokio/

  # Process specific document types
  context-server process pdf document.pdf --extract-tables
  context-server process url https://example.com --output-dir docs
  context-server process git https://github.com/user/repo.git

  # Clean existing markdown files
  context-server clean output/ --no-backup

  # Check supported formats
  context-server info
            """,
        )

        # Global options
        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose logging"
        )

        parser.add_argument(
            "--output-dir",
            "-o",
            default="output",
            help="Output directory (default: output)",
        )

        # Subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Extract command (smart extraction)
        extract_parser = subparsers.add_parser(
            "extract", help="Smart extraction with automatic method detection"
        )
        extract_parser.add_argument("url", help="URL to extract from")
        extract_parser.add_argument(
            "--no-clean", action="store_true", help="Skip markdown cleaning"
        )

        # Process command (specific processors)
        process_parser = subparsers.add_parser(
            "process", help="Process documents using specific processors"
        )
        process_subparsers = process_parser.add_subparsers(
            dest="process_type", help="Document type to process"
        )

        # PDF processing
        pdf_parser = process_subparsers.add_parser("pdf", help="Process PDF files")
        pdf_parser.add_argument("file", help="Path to PDF file")
        pdf_parser.add_argument(
            "--extract-tables", action="store_true", help="Extract tables from PDF"
        )
        pdf_parser.add_argument(
            "--split-vertical",
            action="store_true",
            help="Apply vertical splitting to PDF",
        )

        # URL processing
        url_parser = process_subparsers.add_parser("url", help="Process URLs")
        url_parser.add_argument("url", help="URL to process")

        # Git processing
        git_parser = process_subparsers.add_parser(
            "git", help="Process Git repositories"
        )
        git_parser.add_argument("repository", help="Git repository URL or local path")

        # Clean command
        clean_parser = subparsers.add_parser(
            "clean", help="Clean existing markdown files"
        )
        clean_parser.add_argument(
            "directory", help="Directory containing markdown files"
        )
        clean_parser.add_argument(
            "--no-backup", action="store_true", help="Skip creating backups"
        )

        # Info command
        subparsers.add_parser(
            "info", help="Show information about supported formats and capabilities"
        )

        return parser

    def run_extract(self, args) -> int:
        """Run smart extraction command."""
        try:
            self.logger.info("Starting smart extraction", url=args.url)

            extractor = UnifiedDocumentExtractor(Path(args.output_dir))
            result = extractor.extract_from_url(args.url)

            if not result.success:
                self.logger.error("Extraction failed", error=result.error)
                return 1

            # Apply cleaning if requested
            if not args.no_clean and result.content:
                self.logger.info("Applying markdown cleaning")
                cleaner = MarkdownCleaner()
                cleaned_content = cleaner.clean_content(result.content)

                # Save cleaned content
                if result.metadata.get("source_type") != "sitemap":
                    output_file = Path(args.output_dir) / "extracted_content.md"
                    output_file.write_text(cleaned_content, encoding="utf-8")
                    self.logger.info(
                        "Saved cleaned content", file_path=str(output_file)
                    )

            self.logger.info("Extraction completed successfully")
            return 0

        except Exception as e:
            self.logger.error("Extraction failed", error=str(e))
            return 1

    def run_process(self, args) -> int:
        """Run document processing with specific processor."""
        try:
            # Create processing options
            options = ProcessingOptions(
                output_dir=Path(args.output_dir),
                extract_tables=getattr(args, "extract_tables", False),
                split_vertical=getattr(args, "split_vertical", False),
            )

            # Get the source based on process type
            if args.process_type == "pdf":
                source = Path(args.file)
                if not source.exists():
                    self.logger.error("PDF file not found", file_path=str(source))
                    return 1
            elif args.process_type == "url":
                source = args.url
            elif args.process_type == "git":
                source = args.repository
            else:
                self.logger.error(
                    "Unknown process type", process_type=args.process_type
                )
                return 1

            # Create factory and get processor
            factory = ProcessorFactory(options)
            processor = factory.get_processor(source)

            if not processor:
                self.logger.error(
                    "No processor available for source", source=str(source)
                )
                return 1

            # Process the source
            self.logger.info(
                "Processing document",
                processor_type=processor.get_source_type(),
                source=str(source),
            )

            result = processor.process(source)

            if not result.success:
                self.logger.error("Processing failed", error=result.error)
                return 1

            # Save results
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save main content
            content_file = output_dir / "processed_content.md"
            content_file.write_text(result.content, encoding="utf-8")

            # Save metadata
            metadata_file = output_dir / "metadata.json"
            import json

            metadata_file.write_text(
                json.dumps(result.metadata, indent=2), encoding="utf-8"
            )

            self.logger.info(
                "Processing completed successfully",
                content_file=str(content_file),
                chunk_count=len(result.chunks),
            )
            return 0

        except Exception as e:
            self.logger.error("Processing failed", error=str(e))
            return 1

    def run_clean(self, args) -> int:
        """Run markdown cleaning command."""
        try:
            directory = Path(args.directory)
            if not directory.exists():
                self.logger.error("Directory not found", directory=str(directory))
                return 1

            self.logger.info("Cleaning markdown files", directory=str(directory))

            cleaner = MarkdownCleaner()
            stats = cleaner.clean_directory(
                directory, create_backups=not args.no_backup
            )

            self.logger.info(
                "Cleaning completed successfully",
                files_processed=stats["files_processed"],
                files_backed_up=stats["files_backed_up"],
            )

            return 0

        except Exception as e:
            self.logger.error("Cleaning failed", error=str(e))
            return 1

    def run_info(self, args) -> int:
        """Show information about supported formats."""
        try:
            factory = ProcessorFactory()
            supported_types = factory.get_supported_types()

            print("Context Server - Document Processing System")
            print("=" * 45)
            print()
            print("Supported document types:")
            for doc_type in supported_types:
                print(f"  • {doc_type}")

            print()
            print("Available commands:")
            print("  • extract    - Smart extraction with auto-detection")
            print("  • process    - Process with specific processors")
            print("  • clean      - Clean existing markdown files")
            print("  • info       - Show this information")

            print()
            print("For detailed help: context-server --help")
            print("For command help: context-server <command> --help")

            return 0

        except Exception as e:
            self.logger.error("Failed to show info", error=str(e))
            return 1

    def run(self, argv: Optional[list] = None) -> int:
        """Main CLI entry point."""
        parser = self.create_parser()
        args = parser.parse_args(argv)

        # Setup logging
        log_level = "DEBUG" if args.verbose else "INFO"
        setup_logging(log_level)

        # Route to appropriate command
        if args.command == "extract":
            return self.run_extract(args)
        elif args.command == "process":
            return self.run_process(args)
        elif args.command == "clean":
            return self.run_clean(args)
        elif args.command == "info":
            return self.run_info(args)
        else:
            parser.print_help()
            return 1


def main():
    """CLI entry point."""
    cli = ContextServerCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
