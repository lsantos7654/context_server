"""
Common utility functions and patterns.

Consolidates common patterns from across the codebase into reusable utilities
following CLAUDE.md principles.
"""

# Use standard logging to avoid circular import
import logging
import re
import shutil
from pathlib import Path

# No typing imports needed for Python 3.12+
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


class URLUtils:
    """
    URL-related utility functions.

    Consolidates URL handling patterns found across multiple modules.
    """

    @staticmethod
    def normalize_url(url: str) -> str:
        """
        Normalize URL by adding scheme if missing.

        Args:
            url: URL to normalize

        Returns:
            Normalized URL with scheme
        """
        if not url:
            raise ValueError("URL cannot be empty")

        parsed = urlparse(url)
        if not parsed.scheme:
            url = f"https://{url}"

        return url

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """
        Check if URL is valid and reachable.

        Args:
            url: URL to validate

        Returns:
            True if URL is valid and reachable
        """
        try:
            normalized = URLUtils.normalize_url(url)
            parsed = urlparse(normalized)

            # Basic validation
            if not all([parsed.scheme, parsed.netloc]):
                return False

            # Check if reachable (with timeout)
            response = requests.head(normalized, timeout=10, allow_redirects=True)
            return response.status_code < 400

        except Exception:
            return False

    @staticmethod
    def extract_domain(url: str) -> str:
        """
        Extract domain from URL.

        Args:
            url: URL to extract domain from

        Returns:
            Domain name
        """
        parsed = urlparse(URLUtils.normalize_url(url))
        return parsed.netloc

    @staticmethod
    def is_same_domain(url1: str, url2: str) -> bool:
        """
        Check if two URLs are from the same domain.

        Args:
            url1: First URL
            url2: Second URL

        Returns:
            True if both URLs are from the same domain
        """
        try:
            domain1 = URLUtils.extract_domain(url1)
            domain2 = URLUtils.extract_domain(url2)
            return domain1.lower() == domain2.lower()
        except Exception:
            return False


class FileUtils:
    """
    File and path utility functions.

    Consolidates file handling patterns found across multiple modules.
    """

    @staticmethod
    def create_safe_filename(text: str, max_length: int = 255) -> str:
        """
        Create safe filename from text.

        Args:
            text: Text to convert to filename
            max_length: Maximum filename length

        Returns:
            Safe filename
        """
        # Remove or replace unsafe characters
        safe_text = re.sub(r"[^\w\-_.]", "_", text)

        # Remove multiple consecutive underscores
        safe_text = re.sub(r"_+", "_", safe_text)

        # Remove leading/trailing underscores
        safe_text = safe_text.strip("_")

        # Ensure it's not empty
        if not safe_text:
            safe_text = "untitled"

        # Truncate if too long
        if len(safe_text) > max_length:
            name, ext = FileUtils.split_filename(safe_text)
            max_name_length = max_length - len(ext) - 1  # -1 for the dot
            safe_text = name[:max_name_length] + "." + ext if ext else name[:max_length]

        return safe_text

    @staticmethod
    def split_filename(filename: str) -> tuple[str, str]:
        """
        Split filename into name and extension.

        Args:
            filename: Filename to split

        Returns:
            Tuple of (name, extension)
        """
        path = Path(filename)
        return path.stem, path.suffix.lstrip(".")

    @staticmethod
    def ensure_directory(path: str | Path) -> Path:
        """
        Ensure directory exists, creating it if necessary.

        Args:
            path: Directory path

        Returns:
            Path object for the directory
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @staticmethod
    def backup_file(file_path: Path, backup_suffix: str = ".backup") -> Path | None:
        """
        Create backup of file.

        Args:
            file_path: File to backup
            backup_suffix: Suffix for backup file

        Returns:
            Path to backup file or None if failed
        """
        try:
            if not file_path.exists():
                return None

            backup_dir = file_path.parent / "backup"
            backup_dir.mkdir(exist_ok=True)

            backup_path = backup_dir / f"{file_path.name}{backup_suffix}"
            shutil.copy2(file_path, backup_path)

            logger.debug(
                "Created backup", original=str(file_path), backup=str(backup_path)
            )
            return backup_path

        except Exception as e:
            logger.warning(
                "Failed to create backup", file_path=str(file_path), error=str(e)
            )
            return None

    @staticmethod
    def get_file_size_mb(file_path: Path) -> float:
        """
        Get file size in megabytes.

        Args:
            file_path: Path to file

        Returns:
            File size in MB
        """
        try:
            size_bytes = file_path.stat().st_size
            return size_bytes / (1024 * 1024)
        except Exception:
            return 0.0


class TextUtils:
    """
    Text processing utility functions.

    Consolidates text handling patterns found across multiple modules.
    """

    @staticmethod
    def clean_whitespace(text: str) -> str:
        """
        Clean and normalize whitespace in text.

        Args:
            text: Text to clean

        Returns:
            Text with normalized whitespace
        """
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in text.split("\n")]

        # Remove excessive blank lines (max 2 consecutive)
        cleaned_lines = []
        blank_count = 0

        for line in lines:
            if not line.strip():
                blank_count += 1
                if blank_count <= 2:
                    cleaned_lines.append(line)
            else:
                blank_count = 0
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """
        Truncate text to maximum length with suffix.

        Args:
            text: Text to truncate
            max_length: Maximum length
            suffix: Suffix to add if truncated

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text

        truncate_length = max_length - len(suffix)
        return text[:truncate_length] + suffix

    @staticmethod
    def extract_title_from_content(content: str) -> str | None:
        """
        Extract title from markdown content.

        Args:
            content: Markdown content

        Returns:
            Extracted title or None
        """
        lines = content.strip().split("\n")

        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
            elif line.startswith("## "):
                return line[3:].strip()

        return None

    @staticmethod
    def count_words(text: str) -> int:
        """
        Count words in text.

        Args:
            text: Text to count words in

        Returns:
            Word count
        """
        return len(text.split())

    @staticmethod
    def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
        """
        Estimate reading time in minutes.

        Args:
            text: Text to estimate reading time for
            words_per_minute: Average reading speed

        Returns:
            Estimated reading time in minutes
        """
        word_count = TextUtils.count_words(text)
        return max(1, round(word_count / words_per_minute))


class ValidationUtils:
    """
    Input validation utility functions.

    Consolidates validation patterns found across multiple modules.
    """

    @staticmethod
    def validate_file_path(path: str | Path, must_exist: bool = True) -> Path:
        """
        Validate and normalize file path.

        Args:
            path: Path to validate
            must_exist: Whether file must exist

        Returns:
            Validated Path object

        Raises:
            ValueError: If path is invalid
        """
        if not path:
            raise ValueError("Path cannot be empty")

        path_obj = Path(path)

        if must_exist and not path_obj.exists():
            raise ValueError(f"Path does not exist: {path}")

        return path_obj.resolve()

    @staticmethod
    def validate_url(url: str, check_reachable: bool = False) -> str:
        """
        Validate URL format and optionally check if reachable.

        Args:
            url: URL to validate
            check_reachable: Whether to check if URL is reachable

        Returns:
            Normalized URL

        Raises:
            ValueError: If URL is invalid
        """
        if not url:
            raise ValueError("URL cannot be empty")

        try:
            normalized = URLUtils.normalize_url(url)
            parsed = urlparse(normalized)

            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError(f"Invalid URL format: {url}")

            if check_reachable and not URLUtils.is_valid_url(normalized):
                raise ValueError(f"URL is not reachable: {url}")

            return normalized

        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Invalid URL: {e}")

    @staticmethod
    def validate_directory(path: str | Path, create_if_missing: bool = False) -> Path:
        """
        Validate directory path.

        Args:
            path: Directory path to validate
            create_if_missing: Whether to create directory if missing

        Returns:
            Validated Path object

        Raises:
            ValueError: If directory is invalid
        """
        if not path:
            raise ValueError("Directory path cannot be empty")

        path_obj = Path(path)

        if path_obj.exists() and not path_obj.is_dir():
            raise ValueError(f"Path exists but is not a directory: {path}")

        if not path_obj.exists():
            if create_if_missing:
                path_obj.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError(f"Directory does not exist: {path}")

        return path_obj.resolve()


class MetadataUtils:
    """
    Metadata handling utility functions.

    Provides consistent metadata structure across the application.
    """

    @staticmethod
    def create_base_metadata(source: str | Path, source_type: str) -> dict[str, any]:
        """
        Create base metadata structure.

        Args:
            source: Source path or URL
            source_type: Type of source (pdf, url, git, etc.)

        Returns:
            Base metadata dictionary
        """
        from datetime import datetime

        metadata = {
            "source": str(source),
            "source_type": source_type,
            "processed_at": datetime.now().isoformat(),
            "processor_version": "1.0.0",  # Could be made configurable
        }

        # Add source-specific metadata
        if source_type == "url":
            metadata["domain"] = URLUtils.extract_domain(str(source))
        elif source_type in ["pdf", "file"]:
            file_path = Path(source)
            if file_path.exists():
                metadata["file_size_mb"] = FileUtils.get_file_size_mb(file_path)
                metadata["file_name"] = file_path.name

        return metadata

    @staticmethod
    def merge_metadata(
        base: dict[str, any], additional: dict[str, any]
    ) -> dict[str, any]:
        """
        Merge metadata dictionaries.

        Args:
            base: Base metadata
            additional: Additional metadata to merge

        Returns:
            Merged metadata
        """
        merged = base.copy()
        merged.update(additional)
        return merged
