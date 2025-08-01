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
        try:
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}"
        except Exception:
            return url

    @staticmethod
    def get_base_url(url: str) -> str:
        """Get base URL (scheme + netloc) from URL."""
        try:
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}"
        except Exception:
            return url

    @staticmethod
    def url_to_title(url: str) -> str:
        """Convert URL to a readable title."""
        try:
            parsed = urlparse(url)
            # Use the last part of the path, or the domain
            if parsed.path and parsed.path != "/":
                title = parsed.path.strip("/").split("/")[-1]
                # Clean up the title
                title = title.replace("-", " ").replace("_", " ")
                return title.title() if title else parsed.netloc
            else:
                return parsed.netloc
        except Exception:
            return "Untitled"

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    @staticmethod
    def join_url(base: str, path: str) -> str:
        """Join base URL with path."""
        base = base.rstrip("/")
        path = path.lstrip("/")
        return f"{base}/{path}"


__all__ = ["FileUtils", "URLUtils"]
