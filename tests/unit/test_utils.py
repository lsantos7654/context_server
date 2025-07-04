"""
Unit tests for core utility functions.
"""

import pytest

from src.core.utils import FileUtils, TextUtils, URLUtils, ValidationUtils


class TestURLUtils:
    """Test URL utility functions."""

    def test_normalize_url_adds_scheme(self):
        """Test that normalize_url adds https:// when missing."""
        result = URLUtils.normalize_url("example.com")
        assert result == "https://example.com"

    def test_normalize_url_preserves_scheme(self):
        """Test that normalize_url preserves existing scheme."""
        result = URLUtils.normalize_url("http://example.com")
        assert result == "http://example.com"

    def test_extract_domain(self):
        """Test domain extraction from URL."""
        result = URLUtils.extract_domain("https://docs.rs/tokio/latest/")
        assert result == "docs.rs"

    def test_is_same_domain(self):
        """Test domain comparison."""
        assert URLUtils.is_same_domain(
            "https://example.com/page1", "https://example.com/page2"
        )
        assert not URLUtils.is_same_domain("https://example.com", "https://other.com")


class TestFileUtils:
    """Test file utility functions."""

    def test_create_safe_filename(self):
        """Test safe filename creation."""
        result = FileUtils.create_safe_filename("My Document/Title: Test?")
        assert result == "My_Document_Title_Test"

    def test_split_filename(self):
        """Test filename splitting."""
        name, ext = FileUtils.split_filename("document.pdf")
        assert name == "document"
        assert ext == "pdf"

    def test_ensure_directory_creates_path(self, tmp_path):
        """Test directory creation."""
        test_dir = tmp_path / "test" / "nested"
        result = FileUtils.ensure_directory(test_dir)
        assert result.exists()
        assert result.is_dir()


class TestTextUtils:
    """Test text utility functions."""

    def test_clean_whitespace(self):
        """Test whitespace cleaning."""
        text = "Line 1   \n\n\n\nLine 2\n  \nLine 3  "
        result = TextUtils.clean_whitespace(text)
        lines = result.split("\n")

        # Check trailing whitespace is removed
        assert all(not line.endswith(" ") for line in lines)

        # Check excessive blank lines are reduced (max 2 consecutive)
        assert result.count("\n\n\n\n") == 0  # No quadruple newlines
        assert result.count("\n\n\n") <= 1  # Allow up to 1 triple newline

    def test_truncate_text(self):
        """Test text truncation."""
        text = "This is a long text that should be truncated"
        result = TextUtils.truncate_text(text, 20, "...")
        assert len(result) == 20
        assert result.endswith("...")

    def test_extract_title_from_content(self):
        """Test title extraction."""
        content = "# Main Title\n\nSome content here\n## Subtitle"
        result = TextUtils.extract_title_from_content(content)
        assert result == "Main Title"

    def test_count_words(self):
        """Test word counting."""
        text = "This is a test with five words"
        result = TextUtils.count_words(text)
        assert result == 7  # "This", "is", "a", "test", "with", "five", "words"


class TestValidationUtils:
    """Test validation utility functions."""

    def test_validate_url_valid(self):
        """Test URL validation with valid URL."""
        result = ValidationUtils.validate_url("https://example.com")
        assert result == "https://example.com"

    def test_validate_url_adds_scheme(self):
        """Test URL validation adds scheme."""
        result = ValidationUtils.validate_url("example.com")
        assert result == "https://example.com"

    def test_validate_url_empty_raises(self):
        """Test URL validation raises on empty URL."""
        with pytest.raises(ValueError, match="URL cannot be empty"):
            ValidationUtils.validate_url("")

    def test_validate_directory_exists(self, tmp_path):
        """Test directory validation with existing directory."""
        result = ValidationUtils.validate_directory(tmp_path)
        assert result == tmp_path.resolve()

    def test_validate_directory_creates_missing(self, tmp_path):
        """Test directory validation creates missing directory."""
        test_dir = tmp_path / "missing"
        result = ValidationUtils.validate_directory(test_dir, create_if_missing=True)
        assert result.exists()
        assert result.is_dir()
