"""Core utility functions for file handling, URL processing, and validation."""

import re
from pathlib import Path
from urllib.parse import urlparse


class FileUtils:
    """File handling utilities."""

    @staticmethod
    def create_safe_filename(filename: str) -> str:
        """Create a safe filename by replacing invalid characters."""
        # Replace invalid filesystem characters with underscores
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", filename)
        # Replace spaces and other problematic characters
        safe_name = re.sub(r"[ \t]+", "_", safe_name)
        # Remove multiple consecutive underscores
        safe_name = re.sub(r"_{2,}", "_", safe_name)
        # Strip leading/trailing underscores and dots
        safe_name = safe_name.strip("_.")
        # Ensure it's not empty
        return safe_name if safe_name else "untitled"

    @staticmethod
    def split_filename(filename: str) -> tuple[str, str]:
        """Split filename into name and extension."""
        path = Path(filename)
        return path.stem, path.suffix.lstrip(".")

    @staticmethod
    def ensure_directory(path: Path) -> Path:
        """Ensure directory exists, create if needed."""
        path.mkdir(parents=True, exist_ok=True)
        return path


class URLUtils:
    """URL processing utilities."""

    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize URL by adding scheme if missing."""
        if not url.startswith(("http://", "https://")):
            return f"https://{url}"
        return url

    @staticmethod
    def extract_domain(url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc

    @staticmethod
    def is_same_domain(url1: str, url2: str) -> bool:
        """Check if two URLs are from the same domain."""
        domain1 = URLUtils.extract_domain(url1)
        domain2 = URLUtils.extract_domain(url2)
        return domain1 == domain2


class TextUtils:
    """Text processing utilities."""

    @staticmethod
    def clean_whitespace(text: str) -> str:
        """Clean excessive whitespace from text."""
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in text.splitlines()]

        # Reduce excessive blank lines (max 2 consecutive)
        cleaned_lines = []
        consecutive_empty = 0

        for line in lines:
            if not line.strip():
                consecutive_empty += 1
                if consecutive_empty <= 2:  # Allow up to 2 consecutive empty lines
                    cleaned_lines.append(line)
            else:
                consecutive_empty = 0
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text to max length with optional suffix."""
        if len(text) <= max_length:
            return text

        truncated_length = max_length - len(suffix)
        return text[:truncated_length] + suffix

    @staticmethod
    def extract_title_from_content(content: str) -> str:
        """Extract title from markdown content."""
        lines = content.splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return ""

    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text."""
        return len(text.split())


class ValidationUtils:
    """Validation utilities."""

    @staticmethod
    def validate_url(url: str) -> str:
        """Validate and normalize URL."""
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")

        return URLUtils.normalize_url(url.strip())

    @staticmethod
    def validate_directory(path: Path, create_if_missing: bool = False) -> Path:
        """Validate directory path."""
        resolved_path = path.resolve()

        if not resolved_path.exists():
            if create_if_missing:
                resolved_path.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError(f"Directory does not exist: {resolved_path}")

        if not resolved_path.is_dir():
            raise ValueError(f"Path is not a directory: {resolved_path}")

        return resolved_path
